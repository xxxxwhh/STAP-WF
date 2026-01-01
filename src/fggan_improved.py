"""
改进的FG-GAN实现: TAM → 时序扰动 → 哑包插入
基于ConvTGen架构,实现trace-level的对抗样本生成
"""
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from traffic_generator import ConditionalTrafficGenerator
from utils.logger import init_logger
from utils.gradcam import get_gradcam_for_model  # Grad-CAM引导


class Conv1DBlock(nn.Module):
    """
    一维卷积块 (ConvTGen基础组件)
    结构: Conv1d → BN → ReLU → Conv1d → BN → ReLU
    """
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 8, stride: int = 1, padding: int = 0):
        super(Conv1DBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)


class ConvTGen(nn.Module):
    """
    ConvTGen: TAM → 时序扰动生成器
    输入: TAM [B, 2, 5000]
    输出: 时序扰动 [B, 2500, 2] (timestamp, direction)
    """
    def __init__(self, seq_length: int = 5000, max_perturbations: int = 1500,
                 beta_alpha: float = 2.0, beta_beta: float = 5.0):
        """
        Args:
            seq_length: TAM序列长度
            max_perturbations: 最大扰动数量
            beta_alpha: Beta分布α参数 (控制前端集中度，越小越集中)
            beta_beta: Beta分布β参数 (控制尾部衰减，越大越快衰减)
        """
        super(ConvTGen, self).__init__()
        self.seq_length = seq_length
        self.max_perturbations = max_perturbations
        self.beta_alpha = beta_alpha
        self.beta_beta = beta_beta

        # 第一层卷积块: 2 → 32 channels
        self.conv_block1 = Conv1DBlock(2, 32, kernel_size=8, stride=1, padding=0)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=2)
        self.dropout1 = nn.Dropout(0.1)

        # 第二层卷积块: 32 → 32 channels
        self.conv_block2 = Conv1DBlock(32, 32, kernel_size=8, stride=1, padding=0)

        # 第三层卷积块: 32 → 64 channels
        self.conv_block3 = Conv1DBlock(32, 64, kernel_size=8, stride=1, padding=0)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=2)
        self.dropout2 = nn.Dropout(0.1)

        # 第四层卷积块: 64 → 64 channels
        self.conv_block4 = Conv1DBlock(64, 64, kernel_size=8, stride=1, padding=0)

        # 计算flatten后的维度
        # 5000 → conv(k8,p0) → 4993 → pool(k4,s2) → 2495
        # 2495 → conv(k8,p0) → 2488 → conv(k8,p0) → 2481
        # 2481 → conv(k8,p0) → 2474 → pool(k4,s2) → 1235
        # 1235 → conv(k8,p0) → 1228
        # 实际需要动态计算或手动调整
        self.L6 = self._calculate_output_length(seq_length)

        # Flatten + 全连接层
        self.fc = nn.Linear(64 * self.L6, max_perturbations * 2)
        self.tanh = nn.Tanh()

        # 时间处理网络 (不包含Sigmoid，因为后面统一处理)
        self.time_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            # nn.Sigmoid()
            # 注意: 不在这里加Sigmoid，由sample_beta_distribution统一处理
        )

        # 方向处理网络: 基准方向 + 残差扰动 (使用tanh保持连续性)
        self.dir_net_base = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Tanh()
        )
        self.dir_net_delta = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Tanh()
        )

    def _calculate_output_length(self, input_length: int) -> int:
        """计算经过所有卷积和池化后的序列长度"""
        L = input_length
        # Conv1 (k=8, p=0)
        L = L - 8 + 1
        # Pool1 (k=4, s=2)
        L = (L - 4) // 2 + 1
        # Conv2 (k=8, p=0)
        L = L - 8 + 1
        # Conv2 (k=3, p=1)
        L = L  # same padding
        # Conv3 (k=8, p=0)
        L = L - 8 + 1
        # Pool2 (k=4, s=2)
        L = (L - 4) // 2 + 1
        # Conv4 (k=8, p=0)
        L = L - 8 + 1
        # Conv4 (k=3, p=1)
        L = L  # same padding
        return L

    def sample_beta_distribution(self, time_channel: torch.Tensor) -> torch.Tensor:
        """
        基于网络输出生成前端偏向的时间戳
        使用Kumaraswamy分布作为Beta分布的替代（有闭式icdf，可微分）
        Args:
            time_channel: [B, max_perturbations, 1] 时间网络输出 (未经Sigmoid)
        Returns:
            time_samples: [B, max_perturbations] 时间戳 [0, 80]秒
        """
        # 1. 网络输出经过Sigmoid映射到(0,1)，作为“伪随机数”
        # 这是关键：网络控制了u的值，而不是完全随机
        u = torch.sigmoid(time_channel).squeeze(-1)  # [B, 2500], (0,1)
        
        # 2. 为避免icdf边界值数值不稳定，将u限制在(0,1)区间内
        u_safe = u.clamp(min=1e-6, max=1.0 - 1e-6)
        
        # 3. 使用Kumaraswamy分布的逆CDF (类似Beta但有闭式解)
        # Kumaraswamy(a, b) 的 icdf: x = (1 - (1 - u)^(1/b))^(1/a)
        # 当 a=beta_alpha, b=beta_beta 时，分布形状与 Beta(a,b) 相似
        a = self.beta_alpha
        b = self.beta_beta
        # icdf(u) = (1 - (1 - u)^(1/b))^(1/a)
        time_samples = torch.pow(1.0 - torch.pow(1.0 - u_safe, 1.0 / b), 1.0 / a)
        
        # 4. 映射到[0, 80]秒
        time_output = time_samples.mul(80.0)
        return time_output

    def forward(self, tam: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            tam: [B, 2, 5000] TAM矩阵
        Returns:
            perturbations: [B, 2500, 2] 时序扰动 (timestamp, direction)
        """
        B = tam.size(0)

        # 卷积特征提取
        x = self.conv_block1(tam)       # [B, 32, ~4993]
        x = self.pool1(x)                # [B, 32, ~2495]
        x = self.dropout1(x)

        x = self.conv_block2(x)          # [B, 32, ~2481]

        x = self.conv_block3(x)          # [B, 64, ~2474]
        x = self.pool2(x)                # [B, 64, ~1235]
        x = self.dropout2(x)

        x = self.conv_block4(x)          # [B, 64, L6]

        # Flatten
        x = x.flatten(1)                 # [B, 64*L6]

        # 全连接 + Tanh
        x = self.fc(x)                   # [B, 2500*2]
        x = self.tanh(x)                 # [-1, 1]
        x = x.view(B, self.max_perturbations, 2)  # [B, 2500, 2]

        # 分离时间和方向通道
        time_channel = x[:, :, 0].unsqueeze(-1)  # [B, 2500, 1]
        dir_channel = x[:, :, 1].unsqueeze(-1)   # [B, 2500, 1]

        # ===== 时间网络处理 + Beta分布变换 =====
        # 先经过时间网络学习特征，再用Beta分布偏向前端
        time_features = self.time_net(time_channel)  # [B, 2500, 1] - 网络学习
        time_output = self.sample_beta_distribution(time_features)  # [B, 2500] - Beta变换
        # ===== 方向网络处理（基准方向 + 残差，连续输出） =====
        base_dir = self.dir_net_base(dir_channel)    # [B, 2500, 1], ∈ [-1,1]
        delta_dir = self.dir_net_delta(dir_channel)  # [B, 2500, 1], ∈ [-1,1]
        dir_cont = (base_dir + 0.5 * delta_dir).squeeze(-1)  # [B, 2500]
        dir_output = torch.clamp(dir_cont, -1.0, 1.0)

        # 堆叠为最终输出
        perturbations = torch.stack([time_output, dir_output], dim=2)  # [B, 2500, 2]

        return perturbations


class TAMDiscriminator(nn.Module):
    """
    TAM判别器 (条件WGAN-GP)
    输入: TAM [B, 2, 5000] + 标签
    输出: 判别分数 (标量)
    """
    def __init__(self, seq_length: int = 5000, num_classes: int = 101):
        super(TAMDiscriminator, self).__init__()
        self.seq_length = seq_length
        self.num_classes = num_classes

        # 标签嵌入
        self.label_embedding = nn.Embedding(num_classes, 50)

        # 判别网络 (处理TAM + 标签)
        self.conv1 = nn.Conv1d(2, 64, kernel_size=8, stride=2, padding=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=8, stride=2, padding=3)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=8, stride=2, padding=3)

        # 计算flatten维度
        L_out = seq_length // 8  # 约625

        self.fc = nn.Sequential(
            nn.Linear(256 * L_out + 50, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 1)  # 无激活函数 (WGAN)
        )

    def forward(self, tam: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tam: [B, 2, 5000]
            labels: [B] 标签索引
        Returns:
            validity: [B, 1] 判别分数
        """
        B = tam.size(0)

        # 归一化TAM输入 (关键修复!)
        # TAM的值可能很大(几百),需要归一化到合理范围
        tam_normalized = tam / (tam.max() + 1e-6)

        # 卷积特征提取
        x = F.leaky_relu(self.conv1(tam_normalized), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)

        x = x.flatten(1)  # [B, 256*L_out]

        # 标签嵌入
        label_emb = self.label_embedding(labels)  # [B, 50]

        # 拼接并判别
        x = torch.cat([x, label_emb], dim=1)
        validity = self.fc(x)

        # 限制判别器输出范围 (关键修复!)
        # WGAN理论上无界,但实践中需要限制以防止训练崩溃
        validity = torch.clamp(validity, -100.0, 100.0)

        return validity


class NoiseAdapter(nn.Module):
    """噪声适配器: latent + 标签 → TAM-like 特征"""
    def __init__(self, num_classes: int, latent_dim: int = 256, seq_length: int = 2000):
        super(NoiseAdapter, self).__init__()
        self.seq_length = seq_length
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        input_dim = latent_dim + num_classes
        hidden_dim = 512

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * seq_length),
        )

    def forward(self, z: torch.Tensor, labels_onehot: torch.Tensor) -> torch.Tensor:
        """生成 TAM-like 原型
        Args:
            z: [B, latent_dim] 高斯噪声
            labels_onehot: [B, num_classes] one-hot 标签
        Returns:
            tam_proto: [B, 2, seq_length]
        """
        x = torch.cat([z, labels_onehot], dim=1)  # [B, latent_dim + num_classes]
        out = self.fc(x)  # [B, 2 * seq_length]
        tam_proto = out.view(-1, 2, self.seq_length)
        return tam_proto

class ImprovedFGGAN:
    """
    改进的FG-GAN训练器
    流程: G₁(frozen) → TAM → G₂(ConvTGen) → 时序扰动 → 对抗trace
    """
    def __init__(self, args, generator_g1_path: str, proxy_model_path: str):
        self.args = args
        self.logger = init_logger('ImprovedFGGAN')

        # 设备
        self.device = self._acquire_device()

        # 是否使用G₁流量生成器（对比实验：使用随机噪声vs使用G₁）
        self.use_g1 = getattr(args, 'use_g1', False)
        
        if self.use_g1:
            # 加载预训练G₁ (冻结)
            self.logger.info(f"加载预训练traffic_generator: {generator_g1_path}")
            checkpoint = torch.load(generator_g1_path, map_location=self.device)

            self.num_classes = checkpoint['num_classes']
            self.noise_dim = checkpoint['noise_dim']
            self.seq_length = checkpoint['seq_length']

            self.G1 = ConditionalTrafficGenerator(
                num_classes=self.num_classes,
                noise_dim=self.noise_dim,
                seq_length=self.seq_length
            ).to(self.device)
            self.G1.load_state_dict(checkpoint['generator_state_dict'])
            self.G1.eval()
            for p in self.G1.parameters():
                p.requires_grad = False
            self.logger.info("G₁ 加载完成并冻结")
        else:
            # 对比方案：不使用G₁，使用latent+标签经噪声适配器生成TAM-like特征
            self.logger.info("对比方案：不使用G₁，G₂输入为latent经NoiseAdapter映射的TAM-like特征")
            self.num_classes = getattr(args, 'num_classes', 95)
            self.noise_dim = getattr(args, 'noise_dim', 2000)
            self.seq_length = getattr(args, 'seq_length', 2000)
            self.G1 = None

            # 新增: latent 维度和噪声适配器
            self.latent_dim = getattr(args, 'latent_dim', 256)
            self.noise_adapter = NoiseAdapter(
                num_classes=self.num_classes,
                latent_dim=self.latent_dim,
                seq_length=self.seq_length
            ).to(self.device)

            self.logger.info(
                f"配置: num_classes={self.num_classes}, noise_dim={self.noise_dim}, "
                f"seq_length={self.seq_length}, latent_dim={self.latent_dim}"
            )
        # 创建G₂ (ConvTGen) - 使用Beta分布
        beta_alpha = getattr(args, 'beta_alpha', 1.0)
        beta_beta = getattr(args, 'beta_beta', 16.0)
        self.G2 = ConvTGen(
            seq_length=self.seq_length,
            beta_alpha=beta_alpha,
            beta_beta=beta_beta
        ).to(self.device)
        self.logger.info(f"G₂ (ConvTGen) 初始化完成 - Beta分布(α={beta_alpha}, β={beta_beta})")

        # 创建判别器
        self.D = TAMDiscriminator(
            seq_length=self.seq_length,
            num_classes=self.num_classes
        ).to(self.device)
        self.logger.info("判别器 D 初始化完成")

        # 代理模型类型
        self.proxy_type = getattr(args, 'proxy_type', 'rf')  # 默认rf
        self.logger.info(f"代理模型类型: {self.proxy_type}")
        
        # 加载代理模型
        if self.proxy_type == 'rf':
            from attacks.modules import RFNet
            self.proxy_model = RFNet(num_classes=self.num_classes).to(self.device)
            self.logger.info("使用RF作为代理模型 (TAM特征)")
        else:
            raise ValueError(f"不支持的代理模型类型: {self.proxy_type}")
        
        # 加载代理模型权重
        self.logger.info(f"加载代理模型权重: {proxy_model_path}")
        ckpt = torch.load(proxy_model_path, map_location=self.device)
        state = ckpt.get('model_state_dict', ckpt)
        self.proxy_model.load_state_dict(state, strict=False)
        self.proxy_model.eval()
        for p in self.proxy_model.parameters():
            p.requires_grad = False
        self.logger.info("代理模型加载完成并冻结")

        # 优化器 (仅训练时需要)
        if hasattr(args, 'lr0'):
            lr = args.lr0  # 进一步降低学习率 0.1 → 0.05
            self.optimizer_G2 = torch.optim.Adam(self.G2.parameters(), lr=lr * 0.1, betas=(0.5, 0.999))
            self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.999))
        else:
            # 推理模式,不需要优化器
            self.optimizer_G2 = None
            self.optimizer_D = None

        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

        # AMP
        self.use_amp = args.amp if hasattr(args, 'amp') else False
        self.scaler = GradScaler(enabled=self.use_amp)

        # 训练历史
        self.history = {
            'loss_D': [],
            'loss_G2_gan': [],
            'loss_G2_adv': [],
            'loss_G2_dist': [],
            'asr': []  # Attack Success Rate
        }

    def _acquire_device(self):
        if self.args.use_gpu and torch.cuda.is_available():
            device = torch.device(f'cuda:{self.args.gpu}')
            self.logger.info(f'使用GPU: cuda:{self.args.gpu}')
        else:
            device = torch.device('cpu')
            self.logger.info('使用CPU')
        return device

    def _labels_to_onehot(self, labels: torch.Tensor) -> torch.Tensor:
        """将标签索引转为one-hot编码"""
        onehot = torch.zeros(labels.size(0), self.num_classes, device=labels.device)
        onehot.scatter_(1, labels.unsqueeze(1), 1.0)
        return onehot

    def trace_to_tam(self, trace: np.ndarray) -> torch.Tensor:
        """
        将时序序列转换为TAM格式
        Args:
            trace: [N, 2] (timestamp, signed_size)
        Returns:
            tam: [2, seq_length] TAM矩阵
        """
        from utils.general import feature_transform
        tam = feature_transform(trace, feature_type='tam', seq_length=self.seq_length)
        return torch.from_numpy(tam).float()

    def perturbations_to_tam_differentiable(self, perturbations: torch.Tensor,
                                           original_trace: np.ndarray) -> torch.Tensor:
        """
        可微分的扰动→TAM转换（绕过NumPy操作）
        Args:
            perturbations: [2500, 2] Tensor (timestamp, direction)
            original_trace: [N, 2] 原始trace
        Returns:
            adv_tam: [2, seq_length] 对抗TAM
        """
        device = perturbations.device
        
        # 1. 将原始trace转为TAM基础
        base_tam = self.trace_to_tam(original_trace)  # [2, seq_length]
        base_tam = base_tam.to(device)
        
        # 2. 过滤有效扰动（使用soft mask以保持可微）
        time_valid = (perturbations[:, 0] >= 0) & (perturbations[:, 0] <= 80)
        dir_valid = torch.abs(perturbations[:, 1]) > 0.5
        valid_mask = (time_valid & dir_valid).float()  # [2500]
        
        # 3. 软量化时间到离散bins（使用soft binning）
        # 与真实TAM提取一致: time_window=0.044s, 时间[0,80]映射到bins
        time_window = 0.044  # 与utils/general.py保持一致
        time_bins = (perturbations[:, 0] / time_window).clamp(0, self.seq_length - 1)
        
        # 4. 使用scatter_add进行向量化累加（替代循环）
        bin_floor = torch.floor(time_bins).long()
        bin_ceil = torch.ceil(time_bins).long().clamp(max=self.seq_length - 1)
        weight_ceil = time_bins - bin_floor.float()
        weight_floor = 1.0 - weight_ceil
        
        # 5. 构建增量TAM（向量化版本）
        delta_tam = torch.zeros(2, self.seq_length, device=device)
        
        # 方向连续值（∈[-1,1]），正值视为发送，负值视为接收
        dir_values = perturbations[:, 1]
        send_contrib = F.relu(dir_values) * valid_mask
        recv_contrib = F.relu(-dir_values) * valid_mask

        delta_tam[0].scatter_add_(0, bin_floor, weight_floor * send_contrib)
        delta_tam[0].scatter_add_(0, bin_ceil, weight_ceil * send_contrib)

        delta_tam[1].scatter_add_(0, bin_floor, weight_floor * recv_contrib)
        delta_tam[1].scatter_add_(0, bin_ceil, weight_ceil * recv_contrib)
        
        # 6. 合并基础TAM和增量TAM
        adv_tam = base_tam + delta_tam
        
        return adv_tam

    def insert_dummy_packets(self, original_trace: np.ndarray,
                            perturbations: np.ndarray,
                            gradcam_importance: Optional[np.ndarray] = None,
                            gradcam_model_type: str = 'tam',
                            top_k: Optional[int] = None,  # None=不使用Top-K（训练时），数值=使用Top-K（推理时）
                            packet_size: int = 1) -> np.ndarray:
        """
        将扰动转换为哑包并插入到原始trace (支持Grad-CAM引导)
        Args:
            original_trace: [N, 2] 原始时序序列
            perturbations: [2500, 2] 扰动 (timestamp, direction)
            gradcam_importance: [seq_length] Grad-CAM重要性分数 (可选)
            gradcam_model_type: Grad-CAM来源模型类型 ('df'/'rf'/'tam')
            top_k: 保留最重要的K个位置。None=插入所有有效扰动（训练用），数值=Top-K过滤（推理用）
            packet_size: 哑包大小 (字节)
        Returns:
            adv_trace: [M, 2] 对抗时序序列
        """
        dummy_packets = []
        
        # 如果提供Grad-CAM且指定top_k，使用Top-K策略选择最重要的扰动（推理模式）
        if gradcam_importance is not None and top_k is not None:
            # 计算每个扰动对应的重要性
            perturbation_importance = []
            valid_indices = []
            
            for i in range(len(perturbations)):
                timestamp = perturbations[i, 0]
                direction = perturbations[i, 1]
                
                # 过滤无效扰动
                if abs(direction) > 0.5 and 0 <= timestamp <= 80:
                    # 根据模型类型进行不同的映射
                    if gradcam_model_type in ['rf', 'tam']:
                        # RF/TAM: 基于时间窗口 (0.044s per bin)
                        time_window = 0.044
                        tam_idx = int(timestamp / time_window)
                        tam_idx = min(tam_idx, len(gradcam_importance) - 1)
                    elif gradcam_model_type == 'df':
                        # DF: 基于数据包顺序索引
                        packet_idx = np.searchsorted(original_trace[:, 0], timestamp)
                        tam_idx = min(packet_idx, len(gradcam_importance) - 1)
                    else:
                        # 默认使用时间窗口映射
                        time_window = 0.044
                        tam_idx = int(timestamp / time_window)
                        tam_idx = min(tam_idx, len(gradcam_importance) - 1)
                    
                    importance = gradcam_importance[tam_idx]
                    perturbation_importance.append(importance)
                    valid_indices.append(i)
            
            # 按重要性排序，选择Top-K
            if len(valid_indices) > top_k:
                # 获取Top-K的索引
                top_k_mask = np.argsort(perturbation_importance)[-top_k:]  # 最大的K个
                selected_indices = [valid_indices[idx] for idx in top_k_mask]
            else:
                # 如果有效扰动少于K，全部保留
                selected_indices = valid_indices
            
            # 只处理选中的扰动
            selected_indices_set = set(selected_indices)
            for i in range(len(perturbations)):
                if i in selected_indices_set:
                    timestamp = perturbations[i, 0]
                    direction = perturbations[i, 1]
                    signed_size = int(np.sign(direction))
                    dummy_packets.append([timestamp, signed_size])
        else:
            # 没有Grad-CAM，插入所有有效扰动
            for i in range(len(perturbations)):
                timestamp = perturbations[i, 0]
                direction = perturbations[i, 1]

                # 过滤无效扰动
                if abs(direction) > 0.5 and 0 <= timestamp <= 80:
                    # 转换为trace格式
                    signed_size = int(np.sign(direction))
                    dummy_packets.append([timestamp, signed_size])

        if len(dummy_packets) == 0:
            return original_trace

        dummy_array = np.array(dummy_packets)

        # 合并并按时间排序
        adv_trace = np.vstack([original_trace, dummy_array])
        adv_trace = adv_trace[adv_trace[:, 0].argsort()]

        return adv_trace

    def compute_gradient_penalty(self, real_tam: torch.Tensor, fake_tam: torch.Tensor,
                                 labels: torch.Tensor) -> torch.Tensor:
        """计算WGAN-GP的梯度惩罚"""
        B = real_tam.size(0)
        alpha = torch.rand(B, 1, 1, device=self.device)

        interpolates = (alpha * real_tam + (1 - alpha) * fake_tam).requires_grad_(True)
        d_interpolates = self.D(interpolates, labels)

        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(B, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    @torch.no_grad()
    def generate_adversarial_trace(self, original_trace: np.ndarray,
                                   label: int,
                                   gradcam_importance: Optional[np.ndarray] = None,
                                   gradcam_model_type: str = 'tam') -> Tuple[np.ndarray, np.ndarray]:
        """
        生成对抗时序序列
        Args:
            original_trace: [N, 2] 原始trace
            label: 标签索引
            gradcam_importance: [seq_length] Grad-CAM重要性分数 (可选)
            gradcam_model_type: Grad-CAM来源模型类型 ('df'/'rf'/'tam')
        Returns:
            adv_trace: [M, 2] 对抗trace
            adv_tam: [2, seq_length] 对抗TAM
        """
        if self.use_g1:
            self.G1.eval()
        self.G2.eval()

        # 1. 生成G₂的输入
        if self.use_g1:
            # 使用G₁生成TAM原型
            noise = torch.randn(1, self.noise_dim, device=self.device)
            label_tensor = torch.tensor([label], device=self.device)
            label_onehot = self._labels_to_onehot(label_tensor)
            tam_proto = self.G1(noise, label_onehot)  # [1, 2, 5000]
        else:
            # 改进方案：使用latent+标签，经噪声适配器生成TAM-like特征
            z = torch.randn(1, self.latent_dim, device=self.device)
            label_tensor = torch.tensor([label], device=self.device)
            label_onehot = self._labels_to_onehot(label_tensor)
            tam_proto = self.noise_adapter(z, label_onehot)  # [1, 2, self.noise_dim]

        # 2. G₂生成时序扰动
        perturbations = self.G2(tam_proto)  # [1, 2500, 2]
        pert_np = perturbations.squeeze(0).cpu().numpy()  # [2500, 2]

        # 3. 插入哑包 (使用Grad-CAM引导 + Top-K策略)
        adv_trace = self.insert_dummy_packets(
            original_trace, 
            pert_np,
            gradcam_importance=gradcam_importance,  # 传入Grad-CAM重要性
            gradcam_model_type=gradcam_model_type,  # 传入模型类型
            top_k=2000  # 推理时使用Top-K，只保留500个最重要的扰动
        )

        # 4. 转换为TAM
        adv_tam = self.trace_to_tam(adv_trace)  # [2, 5000]

        return adv_trace, adv_tam.numpy()

    def save_model(self, save_path: str):
        """保存G₂和D的权重"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        save_dict = {
            'G2_state_dict': self.G2.state_dict(),
            'D_state_dict': self.D.state_dict(),
            'history': self.history,
            'num_classes': self.num_classes,
            'seq_length': self.seq_length
        }

        # 只在优化器存在时保存
        if self.optimizer_G2 is not None:
            save_dict['optimizer_G2'] = self.optimizer_G2.state_dict()
        if self.optimizer_D is not None:
            save_dict['optimizer_D'] = self.optimizer_D.state_dict()

        torch.save(save_dict, save_path)

        self.logger.info(f"模型已保存至: {save_path}")

    def load_model(self, load_path: str):
        """加载G₂和D的权重"""
        checkpoint = torch.load(load_path, map_location=self.device)

        self.G2.load_state_dict(checkpoint['G2_state_dict'])
        self.D.load_state_dict(checkpoint['D_state_dict'])

        # 只在优化器存在时加载
        if self.optimizer_G2 is not None and 'optimizer_G2' in checkpoint:
            self.optimizer_G2.load_state_dict(checkpoint['optimizer_G2'])
        if self.optimizer_D is not None and 'optimizer_D' in checkpoint:
            self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])

        self.history = checkpoint.get('history', self.history)

        self.logger.info(f"模型已从 {load_path} 加载")

    def train_batch(self, real_traces: list, labels: torch.Tensor,
                    n_critic: int = 3) -> dict:
        """
        训练一个batch
        Args:
            real_traces: 列表of原始traces (每个[N,2])
            labels: [B] 标签
            n_critic: 判别器更新次数
        Returns:
            losses: 损失字典
        """
        B = labels.size(0)
        losses = {}

        # 1. 生成G₂的输入
        if self.use_g1:
            # 使用G₁生成TAM原型
            with torch.no_grad():
                noise = torch.randn(B, self.noise_dim, device=self.device)
                labels_onehot = self._labels_to_onehot(labels)
                tam_proto = self.G1(noise, labels_onehot)  # [B, 2, 5000]
        else:
            # 改进方案：使用latent+标签，经噪声适配器生成TAM-like特征
            z = torch.randn(B, self.latent_dim, device=self.device)  # [B, latent_dim]
            labels_onehot = self._labels_to_onehot(labels)           # [B, num_classes]
            tam_proto = self.noise_adapter(z, labels_onehot)         # [B, 2, self.noise_dim]

        # 2. 生成时序扰动 (使用G₂)
        perturbations = self.G2(tam_proto)  # [B, 2500, 2]

        # 3. 根据代理模型类型转换为对应特征
        if self.proxy_type == 'rf':
            # RF: 转换为TAM（使用可微分方法保留梯度）
            adv_tams = []
            for i in range(B):
                adv_tam = self.perturbations_to_tam_differentiable(
                    perturbations[i], 
                    real_traces[i]
                )
                adv_tams.append(adv_tam)
            adv_tams = torch.stack(adv_tams)  # [B, 2, 5000]
            proxy_features = adv_tams

        # 4. 真实TAM
        real_tams = []
        for trace in real_traces:
            real_tam = self.trace_to_tam(trace)
            real_tams.append(real_tam)
        real_tams = torch.stack(real_tams).to(self.device)  # [B, 2, 5000]

        # ========== 训练判别器 D (n_critic次) ==========
        for _ in range(n_critic):
            self.optimizer_D.zero_grad()

            # 真实样本判别
            real_validity = self.D(real_tams, labels)

            # 生成样本判别
            fake_validity = self.D(adv_tams.detach(), labels)

            # WGAN-GP损失 (降低GP权重以稳定D)
            gp = self.compute_gradient_penalty(real_tams, adv_tams.detach(), labels)
            loss_D = -torch.mean(real_validity) + torch.mean(fake_validity) + 3.0 * gp  # GP权重 10.0 → 2.0

            loss_D.backward()
            # 梯度裁剪防止梯度爆炸 (增大裁剪阈值)
            torch.nn.utils.clip_grad_norm_(self.D.parameters(), max_norm=5.0)  # 1.0 → 5.0
            self.optimizer_D.step()

        losses['loss_D'] = loss_D.item()
        losses['gp'] = gp.item()

        # ========== 训练生成器 G₂ ==========
        self.optimizer_G2.zero_grad()

        # GAN损失
        fake_validity = self.D(adv_tams, labels)
        loss_G2_gan = -torch.mean(fake_validity)

        # 对抗损失 (使用代理模型)
        adv_logits = self.proxy_model(proxy_features)

        # C&W风格改进: 即使误分类也继续优化,拉大logits差距
        real_logits = adv_logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        # 排除正确类别,找到其他类别的最大logit
        max_other_logits = (adv_logits - 10000 * F.one_hot(labels, self.num_classes)).max(dim=1)[0]
        # 最小化: real_logit - max_other_logit
        # 目标: 让 real_logit < max_other_logit (即误分类)
        loss_G2_adv = torch.mean(torch.clamp(real_logits - max_other_logits, min=0))

        # 总损失 (大幅提高对抗损失权重，降低GAN损失)
        # 由于ASR很低，需要更强的对抗信号
        alpha, beta = 10, 0.5  # 对抗100x, GAN 0.1x (从 10x/0.5x 调整)
        loss_G2 = alpha * loss_G2_adv + beta * loss_G2_gan

        loss_G2.backward()
        # 梯度裁剪防止梯度爆炸 (增大裁剪阈值)
        torch.nn.utils.clip_grad_norm_(self.G2.parameters(), max_norm=5.0)  # 1.0 → 5.0
        self.optimizer_G2.step()

        losses['loss_G2_gan'] = loss_G2_gan.item()
        losses['loss_G2_adv'] = loss_G2_adv.item()
        losses['loss_G2_total'] = loss_G2.item()

        # 计算ASR (Attack Success Rate) - 基于真实哑包插入
        with torch.no_grad():
            pred = torch.argmax(adv_logits, dim=1)
            asr = (pred != labels).float().mean().item()
        losses['asr'] = asr
        
        return losses
