
"""
流量生成器模块
基于条件生成架构，输入噪声+one-hot标签，输出TAM矩阵
使用预训练的RFNet作为代理模型提供语义监督
"""
import argparse
import os
from typing import Union, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加logger导入
from attacks.modules import RFNet
from utils.data import MyDataset
from utils.general import parse_trace, feature_transform
from utils.logger import init_logger


class ConditionalTrafficGenerator(nn.Module):
    """
    条件流量生成器
    输入: 噪声向量(500维) + one-hot标签(101维)
    输出: TAM矩阵 [batch, 2, seq_length]
    """
    
    def __init__(self, num_classes: int = 101, noise_dim: int = 500, 
                 seq_length: int = 5000, hidden_dim: int = 512):
        super(ConditionalTrafficGenerator, self).__init__()
        self.num_classes = num_classes
        self.noise_dim = noise_dim
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        
        # 添加logger (用于调试)
        self.logger = init_logger('ConditionalTrafficGenerator')
        
        # 输入维度: noise_dim + num_classes (one-hot)
        input_dim = noise_dim + num_classes
        
        # 初始编码层 (添加更多正则化防止过拟合)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 出站计数生成分支 (channel 0)
        self.outgoing_branch = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_length)
        )
        
        # 入站计数生成分支 (channel 1)
        self.incoming_branch = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_length)
        )
        
        # 时空融合层 (残差 refinement，添加正则化)
        self.fusion = nn.Sequential(
            nn.Linear(seq_length * 2, seq_length * 2),
            nn.BatchNorm1d(seq_length * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(seq_length * 2, seq_length * 2)
        )
                
        # 添加可学习的输出缩放因子，让生成器学习合适的输出范围
        # 降低初始值，配合tanh约束使用
        self.output_scale = nn.Parameter(torch.tensor(5.0))
    
    def forward(self, noise: torch.Tensor, labels_onehot: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            noise: [batch_size, noise_dim] 标准正态噪声
            labels_onehot: [batch_size, num_classes] one-hot编码标签
        Returns:
            tam_matrix: [batch_size, 2, seq_length] TAM矩阵，两通道均为非负计数
        """
        # 添加输入检查
        if torch.isnan(noise).any() or torch.isinf(noise).any():
            raise ValueError("Input noise contains NaN or Inf")
        if torch.isnan(labels_onehot).any() or torch.isinf(labels_onehot).any():
            raise ValueError("Input labels_onehot contains NaN or Inf")
        
        # 拼接噪声与标签
        x = torch.cat([noise, labels_onehot], dim=1)  # [B, noise_dim + num_classes]
        
        # 编码
        h = self.encoder(x)  # [B, hidden_dim * 2]
        
        # 分别生成出站与入站计数序列
        outgoing_raw = self.outgoing_branch(h)  # [B, seq_length]
        incoming_raw = self.incoming_branch(h)  # [B, seq_length]
        
        # 添加中间检查
        if torch.isnan(outgoing_raw).any() or torch.isinf(outgoing_raw).any():
            raise ValueError("Outgoing branch output contains NaN or Inf")
        if torch.isnan(incoming_raw).any() or torch.isinf(incoming_raw).any():
            raise ValueError("Incoming branch output contains NaN or Inf")
        
        # 改用ReLU施加非负约束，允许稀疏输出（很多位置为0）
        # softplus会让所有值都>0，导致100%非零，不符合真实TAM的稀疏性
        outgoing = torch.nn.functional.relu(outgoing_raw)
        incoming = torch.nn.functional.relu(incoming_raw)
        
        # 拼接并融合
        combined = torch.cat([outgoing, incoming], dim=1)  # [B, seq_length * 2]
        
        # 添加检查
        if torch.isnan(combined).any() or torch.isinf(combined).any():
            raise ValueError("Combined tensor contains NaN or Inf")
        
        fused = combined + self.fusion(combined)  # 残差连接
        
        # 添加检查
        if torch.isnan(fused).any() or torch.isinf(fused).any():
            raise ValueError("Fused tensor contains NaN or Inf")
        
        # 分离融合后的两通道并再次施加约束
        # 使用ReLU允许输出为0，保持稀疏性
        outgoing_final = torch.nn.functional.relu(fused[:, :self.seq_length])
        incoming_final = torch.nn.functional.relu(fused[:, self.seq_length:])
        
        # 使用可学习的缩放因子调整输出范围
        # softplus保证缩放因子始终为正，避免输出为负
        scale = torch.nn.functional.softplus(self.output_scale)
        outgoing_final = outgoing_final * scale
        incoming_final = incoming_final * scale
        
        # 添加检查
        if torch.isnan(outgoing_final).any() or torch.isinf(outgoing_final).any():
            raise ValueError("Outgoing final contains NaN or Inf")
        if torch.isnan(incoming_final).any() or torch.isinf(incoming_final).any():
            raise ValueError("Incoming final contains NaN or Inf")
        
        # 堆叠为TAM矩阵 [B, 2, seq_length]
        # channel 0: outgoing counts, channel 1: incoming counts
        tam_matrix = torch.stack([outgoing_final, incoming_final], dim=1)
        
        # 添加输出检查
        if torch.isnan(tam_matrix).any() or torch.isinf(tam_matrix).any():
            raise ValueError("Output TAM contains NaN or Inf")
        
        return tam_matrix


class ProxyRFClassifier(nn.Module):
    """
    代理分类器包装器
    加载预训练的RFNet并冻结参数，用于语义监督
    """
    
    def __init__(self, num_classes: int = 101, checkpoint_path: Optional[str] = None):
        super(ProxyRFClassifier, self).__init__()
        self.num_classes = num_classes
        self.rf_net = RFNet(num_classes=num_classes)
        
        # 加载预训练权重
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            # 支持直接state_dict或嵌套在'model_state_dict'下
            state = checkpoint.get('model_state_dict', checkpoint)
            self.rf_net.load_state_dict(state, strict=False)
            print(f"已加载代理模型权重: {checkpoint_path}")
        
        # 冻结所有参数
        for param in self.rf_net.parameters():
            param.requires_grad = False
        
        # 设为评估模式
        self.rf_net.eval()
    
    def forward(self, tam_matrix: torch.Tensor) -> torch.Tensor:
        """
        分类TAM矩阵
        Args:
            tam_matrix: [batch_size, 2, seq_length]
        Returns:
            logits: [batch_size, num_classes]
        """
        # 移除 no_grad，允许梯度传播用于生成器训练
        # 代理分类器参数已冻结（requires_grad=False），不会被更新
        logits = self.rf_net(tam_matrix)
        return logits


class TrafficGeneratorTrainer:
    """
    流量生成器训练器
    管理生成器训练流程，包括损失计算、优化与模型保存
    """
    
    def __init__(self, args: argparse.Namespace, proxy_checkpoint: Optional[str] = None):
        self.args = args
        self.logger = init_logger('TrafficGeneratorTrainer')
        
        # 在logger初始化后立即添加一行，用于快速验证logger工作状态
        self.logger.info("TrafficGeneratorTrainer initialized with enhanced stability settings.")
        
        # 设备
        self.device = self._acquire_device()
        
        # 类别数 (开放世界: mon_classes + 1)
        self.num_classes = args.mon_classes + 1 if args.open_world else args.mon_classes
        self.noise_dim = 500
        self.seq_length = args.seq_length
        
        # 创建生成器
        self.generator = ConditionalTrafficGenerator(
            num_classes=self.num_classes,
            noise_dim=self.noise_dim,
            seq_length=self.seq_length,
            hidden_dim=512
        ).to(self.device)
        
        # 创建并冻结代理分类器
        self.proxy_classifier = ProxyRFClassifier(
            num_classes=self.num_classes,
            checkpoint_path=proxy_checkpoint
        ).to(self.device)
        
        # 类别原型（在logits空间），用于InfoNCE对比学习
        # 初始化为零并在训练中用真实数据的logits移动平均更新
        self.prototypes = torch.zeros(self.num_classes, self.num_classes, device=self.device)
        self.prototype_momentum = 0.95
        self.prototype_initialized = torch.zeros(self.num_classes, dtype=torch.bool, device=self.device)
        
        # 优化器
        self.optimizer = optim.Adam(
            self.generator.parameters(),
            lr=args.lr0 if hasattr(args, 'lr0') else 0.0001,
            betas=(0.9, 0.999)
        )
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
        self.huber_loss = nn.HuberLoss(delta=1.0)
        
        # AMP混合精度
        self.use_amp = args.amp if hasattr(args, 'amp') else True
        self.scaler = GradScaler(enabled=self.use_amp and self.device.type == 'cuda')
        
        # 训练历史
        self.history = {
            'total_loss': [],
            'outgoing_loss': [],
            'incoming_loss': [],
            'count_loss': [],
            'ratio_loss': [],
            'semantic_loss': [],
            'contrastive_loss': [],
            'sparsity_loss': [],
            'mean_std_loss': [],
            'nonzero_loss': [],
            'proxy_accuracy': []
        }
    
    def _acquire_device(self):
        if self.args.use_gpu and torch.cuda.is_available():
            device = torch.device(f'cuda:{self.args.gpu}')
            self.logger.info(f'使用GPU: cuda:{self.args.gpu}')
        else:
            device = torch.device('cpu')
            self.logger.info('使用CPU')
        return device
    
    def validate_proxy_classifier(self, dataloader: DataLoader) -> float:
        """
        验证代理分类器对真实TAM的准确率
        """
        self.proxy_classifier.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for real_tam, labels in dataloader:
                real_tam = real_tam.to(self.device).float()
                labels = labels.to(self.device).long()
                
                logits = self.proxy_classifier(real_tam)
                _, predicted = torch.max(logits, dim=1)
                
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy
    
    @staticmethod
    def extract_tam(data_path: Union[str, os.PathLike], seq_length: int) -> np.ndarray:
        """
        提取TAM特征用于生成器训练
        注意：在当前环境中，feature_transform已返回[2, seq_length]格式
        """
        trace = parse_trace(data_path)
        feat = feature_transform(trace, feature_type='tam', seq_length=seq_length)
        # 返回原始格式（已经是[2, seq_length]）
        return feat
    
    def _labels_to_onehot(self, labels: torch.Tensor) -> torch.Tensor:
        """
        将标签索引转为one-hot编码
        Args:
            labels: [batch_size] 长整型标签索引
        Returns:
            onehot: [batch_size, num_classes]
        """
        onehot = torch.zeros(labels.size(0), self.num_classes, device=labels.device)
        onehot.scatter_(1, labels.unsqueeze(1), 1.0)
        return onehot
    
    def compute_losses(self, fake_tam: torch.Tensor, real_tam: torch.Tensor, 
                       labels: torch.Tensor) -> dict:
        """
        计算多项损失
        Args:
            fake_tam: [B, 2, T] 生成的TAM
            real_tam: [B, 2, T] 真实的TAM
            labels: [B] 标签索引
        Returns:
            losses: 损失字典
        """
        losses = {}
        
        # 1. 逐bin结构损失（归一化MSE）
        fake_norm = fake_tam / (fake_tam.max() + 1e-6)
        real_norm = real_tam / (real_tam.max() + 1e-6)
        outgoing_loss = self.mse_loss(fake_norm[:, 0, :], real_norm[:, 0, :])
        incoming_loss = self.mse_loss(fake_norm[:, 1, :], real_norm[:, 1, :])
        losses['outgoing_loss'] = outgoing_loss
        losses['incoming_loss'] = incoming_loss
        
        # 2. 总计数匹配（归一化相对误差）
        real_counts = real_tam.sum(dim=2)
        fake_counts = fake_tam.sum(dim=2)
        count_loss = self.mse_loss(
            fake_counts / (fake_counts.abs().max() + 1e-6),
            real_counts / (real_counts.abs().max() + 1e-6)
        )
        losses['count_loss'] = count_loss
        
        # 3. 方向比例损失（KL散度）
        real_total = real_counts.sum(dim=1, keepdim=True) + 1e-8
        fake_total = fake_counts.sum(dim=1, keepdim=True) + 1e-8
        real_ratio = real_counts / real_total
        fake_ratio = fake_counts / fake_total
        ratio_loss = self.kl_loss(torch.log(fake_ratio + 1e-8), real_ratio)
        losses['ratio_loss'] = ratio_loss
        
        # 4. 语义一致性损失（代理模型）
        # 使用每样本的均值/标准差进行可微缩放，使动态范围对齐真实分布
        B = real_tam.size(0)
        real_flat = real_tam.view(B, -1)
        fake_flat = fake_tam.view(B, -1)
        m_r = real_flat.mean(dim=1, keepdim=True).view(-1, 1, 1)
        s_r = real_flat.std(dim=1, keepdim=True).view(-1, 1, 1) + 1e-6
        m_f = fake_flat.mean(dim=1, keepdim=True).view(-1, 1, 1)
        s_f = fake_flat.std(dim=1, keepdim=True).view(-1, 1, 1) + 1e-6
        fake_tam_rescaled = (fake_tam - m_f) / s_f * s_r + m_r
        
        fake_logits = self.proxy_classifier(fake_tam_rescaled)
        
        if self.num_classes > 100:
            class_weights = torch.ones(self.num_classes, device=labels.device)
            class_weights[-1] = 0.1
            semantic_loss = nn.CrossEntropyLoss(weight=class_weights)(fake_logits, labels)
        else:
            semantic_loss = self.ce_loss(fake_logits, labels)
        losses['semantic_loss'] = semantic_loss
        
        # 使用真实样本的logits更新类别原型（移动平均）与InfoNCE对比损失
        with torch.no_grad():
            real_logits_batch = self.proxy_classifier(real_tam)
            uniq = labels.unique()
            for c in uniq:
                mask = (labels == c)
                if mask.any():
                    proto_update = real_logits_batch[mask].mean(dim=0)
                    if not self.prototype_initialized[c]:
                        self.prototypes[c] = proto_update
                        self.prototype_initialized[c] = True
                    else:
                        self.prototypes[c] = self.prototype_momentum * self.prototypes[c] + (1.0 - self.prototype_momentum) * proto_update
        
        # InfoNCE：将假样本嵌入（logits空间）拉近对应类别原型，远离其它类别
        temperature = 0.07
        fake_emb = torch.nn.functional.normalize(fake_logits, dim=1)
        proto_norm = torch.nn.functional.normalize(self.prototypes, dim=1)
        sims = torch.matmul(fake_emb, proto_norm.T) / temperature  # [B, C]
        contrastive_loss = nn.CrossEntropyLoss()(sims, labels)
        losses['contrastive_loss'] = contrastive_loss
        
        # 稀疏性与分布约束
        # 1) L1正则（鼓励稀疏）
        sparsity_loss = fake_tam.abs().mean()
        losses['sparsity_loss'] = sparsity_loss
        
        # 2) 非零比例匹配（用平滑近似 1 - exp(-x/τ)）
        tau = 1.0
        approx_nz_fake = 1.0 - torch.exp(-fake_tam / tau)
        approx_nz_real = 1.0 - torch.exp(-real_tam / tau)
        p_fake = approx_nz_fake.mean(dim=(1, 2))
        p_real = approx_nz_real.mean(dim=(1, 2))
        nonzero_loss = self.mse_loss(p_fake, p_real)
        losses['nonzero_loss'] = nonzero_loss
        
        # 3) 均值/方差匹配
        mean_match = self.mse_loss(m_f.view(-1), m_r.view(-1))
        std_match = self.mse_loss(s_f.view(-1), s_r.view(-1))
        mean_std_loss = mean_match + std_match
        losses['mean_std_loss'] = mean_std_loss
        
        # 计算准确率（准确率计算不需要梯度）
        with torch.no_grad():
            _, predicted = torch.max(fake_logits, dim=1)
            losses['proxy_accuracy'] = (predicted == labels).float().mean().item()
        
        return losses
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> dict:
        """
        训练一个epoch
        """
        self.generator.train()
        
        epoch_losses = {
            'total_loss': 0.0,
            'outgoing_loss': 0.0,
            'incoming_loss': 0.0,
            'count_loss': 0.0,
            'ratio_loss': 0.0,
            'semantic_loss': 0.0,
            'contrastive_loss': 0.0,
            'sparsity_loss': 0.0,
            'mean_std_loss': 0.0,
            'nonzero_loss': 0.0,
            'proxy_accuracy': 0.0
        }
        
        n_batches = 0
        # 每10轮打印一次详细损失分解
        detailed_log = epoch % 10 == 1
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for real_tam, labels in progress_bar:
            n_batches += 1
            batch_size = real_tam.size(0)
            
            # 检查真实数据是否包含无效值
            if torch.isnan(real_tam).any() or torch.isinf(real_tam).any():
                self.logger.warning(f"Skipping batch {n_batches} due to invalid real data")
                continue
            
            # 移动到设备
            # feature_transform已返回[2, seq_length]格式，DataLoader会添加batch维度变成[B, 2, T]
            real_tam = real_tam.to(self.device).float()  # [B, 2, T]
            labels = labels.to(self.device).long()       # [B]
            
            # 生成噪声与one-hot标签
            noise = torch.randn(batch_size, self.noise_dim, device=self.device)
            labels_onehot = self._labels_to_onehot(labels)
            
            # 检查生成输入是否有效
            if torch.isnan(noise).any() or torch.isinf(noise).any():
                self.logger.warning(f"Skipping batch {n_batches} due to invalid noise")
                continue
            
            # 清零梯度
            self.optimizer.zero_grad(set_to_none=True)
            
            # 前向传播 (AMP)
            with autocast(enabled=self.use_amp):
                fake_tam = self.generator(noise, labels_onehot)
                
                # 检查生成输出是否有效
                if torch.isnan(fake_tam).any() or torch.isinf(fake_tam).any():
                    self.logger.warning(f"Skipping batch {n_batches} due to invalid fake TAM")
                    continue
                
                # 计算各项损失
                losses = self.compute_losses(fake_tam, real_tam, labels)
                
                # 组合总损失（调整：降低语义loss权重，增加ratio loss权重，降低总轮数防止过拟合）
                if epoch < 5:
                    w_out = w_in = 0.5; w_count = 0.1; w_ratio = 0.8; w_sem = 0.5; w_con = 0.3; w_l1 = 0.01; w_ms = 0.1; w_nz = 0.05
                elif epoch < 15:
                    w_out = w_in = 0.3; w_count = 0.05; w_ratio = 0.6; w_sem = 1.0; w_con = 0.5; w_l1 = 0.01; w_ms = 0.1; w_nz = 0.05
                else:
                    w_out = w_in = 0.2; w_count = 0.05; w_ratio = 0.5; w_sem = 1.5; w_con = 0.8; w_l1 = 0.01; w_ms = 0.05; w_nz = 0.05
                
                total_loss = (
                    w_out * losses['outgoing_loss'] +
                    w_in * losses['incoming_loss'] +
                    w_count * losses['count_loss'] +
                    w_ratio * losses['ratio_loss'] +
                    w_sem * losses['semantic_loss'] +
                    w_con * losses['contrastive_loss'] +
                    w_l1 * losses['sparsity_loss'] +
                    w_ms * losses['mean_std_loss'] +
                    w_nz * losses['nonzero_loss']
                )
            
            # 在AMP上下文外保存未缩放的损失值（用于日志记录）
            total_loss_unscaled = total_loss.item()
            
            # 反向传播
            self.scaler.scale(total_loss).backward()
            
            # 梯度裁剪（放宽限制）
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
            
            # 更新参数
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # 累计损失（使用未缩放的值）
            epoch_losses['total_loss'] += total_loss_unscaled
            epoch_losses['outgoing_loss'] += losses['outgoing_loss'].item()
            epoch_losses['incoming_loss'] += losses['incoming_loss'].item()
            epoch_losses['count_loss'] += losses['count_loss'].item()
            epoch_losses['ratio_loss'] += losses['ratio_loss'].item()
            epoch_losses['semantic_loss'] += losses['semantic_loss'].item()
            epoch_losses['contrastive_loss'] += losses['contrastive_loss'].item()
            epoch_losses['sparsity_loss'] += losses['sparsity_loss'].item()
            epoch_losses['mean_std_loss'] += losses['mean_std_loss'].item()
            epoch_losses['nonzero_loss'] += losses['nonzero_loss'].item()
            epoch_losses['proxy_accuracy'] += losses['proxy_accuracy']
            
            # 更新进度条 (每10轮显示详细损失)
            if detailed_log:
                progress_bar.set_postfix({
                    'loss': f"{total_loss_unscaled:.2f}",
                    'out': f"{losses['outgoing_loss'].item():.4f}",
                    'in': f"{losses['incoming_loss'].item():.4f}",
                    'sem': f"{losses['semantic_loss'].item():.2f}",
                    'acc': f"{losses['proxy_accuracy']:.3f}"
                })
            else:
                progress_bar.set_postfix({
                    'loss': f"{total_loss_unscaled:.2f}",
                    'acc': f"{losses['proxy_accuracy']:.3f}"
                })
        
        # 计算平均
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
        
        # 每个epoch结束后打印诊断信息（所有epoch）
        # 获取一个batch用于统计对比
        with torch.no_grad():
            for real_tam_sample, labels_sample in dataloader:
                real_tam_sample = real_tam_sample.to(self.device).float()
                labels_sample = labels_sample.to(self.device).long()
                
                # 生成对应的假样本
                noise_sample = torch.randn(real_tam_sample.size(0), self.noise_dim, device=self.device)
                labels_onehot_sample = self._labels_to_onehot(labels_sample)
                fake_tam_sample = self.generator(noise_sample, labels_onehot_sample)
                
                # 统计信息
                real_mean = real_tam_sample.mean().item()
                real_std = real_tam_sample.std().item()
                real_max = real_tam_sample.max().item()
                real_min = real_tam_sample.min().item()
                real_nonzero_ratio = (real_tam_sample > 0).float().mean().item()
                
                fake_mean = fake_tam_sample.mean().item()
                fake_std = fake_tam_sample.std().item()
                fake_max = fake_tam_sample.max().item()
                fake_min = fake_tam_sample.min().item()
                fake_nonzero_ratio = (fake_tam_sample > 0).float().mean().item()
                
                scale_param = torch.nn.functional.softplus(self.generator.output_scale).item()
                
                self.logger.info(
                    f"\n[Epoch {epoch} Diagnostics]\n"
                    f"  Output Scale: {scale_param:.2f}\n"
                    f"  Real TAM  - mean: {real_mean:.4f}, std: {real_std:.4f}, "
                    f"min: {real_min:.4f}, max: {real_max:.2f}, nonzero: {real_nonzero_ratio:.2%}\n"
                    f"  Fake TAM  - mean: {fake_mean:.4f}, std: {fake_std:.4f}, "
                    f"min: {fake_min:.4f}, max: {fake_max:.2f}, nonzero: {fake_nonzero_ratio:.2%}\n"
                    f"  Ratio (Fake/Real) - mean: {fake_mean/real_mean if real_mean > 0 else 0:.2f}x, "
                    f"max: {fake_max/real_max if real_max > 0 else 0:.2f}x"
                )
                break  # 只统计第一个batch
        
        return epoch_losses
    
    def train(self, train_dataloader: DataLoader, epochs: int = 50):
        """
        完整训练流程
        """
        self.logger.info(f"开始训练生成器，总轮数: {epochs}")
        self.logger.info(f"类别数: {self.num_classes}, 噪声维度: {self.noise_dim}, 序列长度: {self.seq_length}")
        
        for epoch in range(1, epochs + 1):
            epoch_losses = self.train_epoch(train_dataloader, epoch)
            
            # 记录历史
            for key in self.history:
                self.history[key].append(epoch_losses[key])
            
            # 打印日志
            self.logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"Total: {epoch_losses['total_loss']:.4f} | "
                f"Out: {epoch_losses['outgoing_loss']:.4f} | "
                f"In: {epoch_losses['incoming_loss']:.4f} | "
                f"Cnt: {epoch_losses['count_loss']:.4f} | "
                f"Ratio: {epoch_losses['ratio_loss']:.4f} | "
                f"Sem: {epoch_losses['semantic_loss']:.4f} | "
                f"Con: {epoch_losses['contrastive_loss']:.4f} | "
                f"L1: {epoch_losses['sparsity_loss']:.4f} | "
                f"MS: {epoch_losses['mean_std_loss']:.4f} | "
                f"NZ: {epoch_losses['nonzero_loss']:.4f} | "
                f"Acc: {epoch_losses['proxy_accuracy']:.3f}"
            )
        
        self.logger.info("训练完成!")
    
    def save_generator(self, save_path: str):
        """
        保存生成器模型 (转至CPU提升可移植性)
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 转至CPU保存
        cpu_state = {k: v.cpu() for k, v in self.generator.state_dict().items()}
        
        torch.save({
            'generator_state_dict': cpu_state,
            'num_classes': self.num_classes,
            'noise_dim': self.noise_dim,
            'seq_length': self.seq_length,
            'history': self.history
        }, save_path)
        
        self.logger.info(f"生成器已保存至: {save_path}")
    
    def load_generator(self, load_path: str):
        """
        加载生成器模型
        """
        checkpoint = torch.load(load_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.history = checkpoint.get('history', self.history)
        self.logger.info(f"生成器已从 {load_path} 加载")
    
    @torch.no_grad()
    def generate_samples(self, labels: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """
        生成指定标签的TAM样本
        Args:
            labels: [n_samples] 标签索引
            n_samples: 样本数 (若labels为单个标签，会重复生成n_samples个)
        Returns:
            tam_samples: [n_samples, 2, seq_length]
        """
        self.generator.eval()
        
        if labels.dim() == 0:
            labels = labels.repeat(n_samples)
        
        labels = labels.to(self.device)
        noise = torch.randn(labels.size(0), self.noise_dim, device=self.device)
        labels_onehot = self._labels_to_onehot(labels)
        
        tam_samples = self.generator(noise, labels_onehot)
        
        return tam_samples.cpu()


def create_argparse_for_generator():
    """
    创建用于生成器训练的参数解析器
    """
    parser = argparse.ArgumentParser(description='流量生成器训练')
    
    # 数据相关
    parser.add_argument('--data-path', type=str, required=True, help='数据目录路径')
    parser.add_argument('--mon-classes', type=int, default=100, help='监控类别数')
    parser.add_argument('--mon-inst', type=int, default=100, help='每类监控实例数')
    parser.add_argument('--unmon-inst', type=int, default=10000, help='未监控实例数')
    parser.add_argument('--open-world', action='store_true', help='开放世界模式')
    parser.add_argument('--suffix', type=str, default='.cell', help='文件后缀')
    parser.add_argument('--seq-length', type=int, default=5000, help='序列长度')
    
    # 训练相关
    parser.add_argument('--batch-size', type=int, default=64, help='批次大小')
    parser.add_argument('--epochs', type=int, default=80, help='训练轮数')
    parser.add_argument('--lr0', type=float, default=0.0002, help='学习率')
    parser.add_argument('--workers', type=int, default=4, help='数据加载线程数')
    
    # 设备相关
    parser.add_argument('--use-gpu', action='store_true', help='使用GPU')
    parser.add_argument('--gpu', type=int, default=1, help='GPU设备ID')
    parser.add_argument('--amp', action='store_true', help='混合精度训练')
    
    # 模型相关
    parser.add_argument('--proxy-checkpoint', type=str, default=None, 
                        help='预训练RFNet代理模型权重路径')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/generator',
                        help='生成器保存目录')
    
    return parser
