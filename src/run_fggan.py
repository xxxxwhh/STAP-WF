"""
改进FG-GAN训练主程序
包含完整的训练流程、评估与对抗样本生成
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from sympy import false
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from fggan_improved import ImprovedFGGAN
from utils.general import get_flist_label, seed_everything, parse_trace
from utils.logger import init_logger


class TraceDataset(Dataset):
    """时序序列数据集"""
    def __init__(self, file_list: np.ndarray, labels: np.ndarray):
        self.file_list = file_list
        self.labels = labels
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        fpath = self.file_list[idx]
        label = self.labels[idx]
        
        # 加载trace
        trace = parse_trace(fpath)
        
        return trace, label


def collate_fn(batch):
    """自定义collate函数 (trace长度不同)"""
    traces, labels = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.long)
    return list(traces), labels


def create_argparse():
    """创建参数解析器"""
    parser = argparse.ArgumentParser(description='改进FG-GAN训练')
    
    # 数据相关
    parser.add_argument('--data-path', type=str, required=True, help='数据目录路径')
    parser.add_argument('--mon-classes', type=int, default=100, help='监控类别数')
    parser.add_argument('--mon-inst', type=int, default=100, help='每类监控实例数')
    parser.add_argument('--unmon-inst', type=int, default=10000, help='未监控实例数')
    parser.add_argument('--open-world', action='store_true', help='开放世界模式')
    parser.add_argument('--suffix', type=str, default='.cell', help='文件后缀')
    
    # 模型相关
    parser.add_argument('--generator-g1', type=str, required=True,
                        help='预训练traffic_generator权重路径')
    parser.add_argument('--proxy-model', type=str, required=True,
                        help='预训练代理模型权重路径 (RF或DF)')
    parser.add_argument('--proxy-type', type=str, default='rf', choices=['rf', 'df'],
                        help='代理模型类型: rf(使用TAM特征) 或 df(使用方向序列特征)')
    parser.add_argument('--use-g1', action='store_true', default=False,
                        help='是否使用G₁流量生成器（False则使用随机正态分布作为对比）')
    parser.add_argument('--no-use-g1', action='store_false', dest='use_g1',
                        help='不使用G₁，直接用随机正态分布（对比实验）')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/fggan',
                        help='模型保存目录')
    
    # 训练相关
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr0', type=float, default=0.0002, help='学习率')
    parser.add_argument('--n-critic', type=int, default=1, help='判别器更新次数（降低以防止判别器过强）')
    parser.add_argument('--workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--monitor-real-insertion', action='store_true',
                        help='训练时监控真实哑包插入的ASR (insert_dummy_packets方法，会增加训练时间)')
    
    # Beta分布参数 (控制扰动在流量前端的集中度)
    parser.add_argument('--beta-alpha', type=float, default=1.0,
                        help='Beta分布α参数 (越小越集中在前端)')
    parser.add_argument('--beta-beta', type=float, default=16.0,
                        help='Beta分布β参数 (越大尾部衰减越快)')
    
    # 设备相关
    parser.add_argument('--use-gpu', action='store_true', help='使用GPU')
    parser.add_argument('--gpu', type=int, default=0, help='GPU设备ID')
    parser.add_argument('--amp', action='store_true', help='混合精度训练')
    
    # 其他
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--save-interval', type=int, default=10, help='保存间隔(epoch)')
    
    return parser


def main():
    # 解析参数
    parser = create_argparse()
    args = parser.parse_args()
    
    # 设置随机种子
    seed_everything(args.seed)
    
    # 初始化logger
    logger = init_logger('RunFGGAN')
    logger.info(f"开始FG-GAN训练: {args}")
    
    # 创建保存目录
    os.makedirs(args.checkpoints, exist_ok=True)
    
    # 加载数据
    logger.info("加载数据...")
    flist, labels = get_flist_label(
        args.data_path, 
        args.mon_classes,
        args.mon_inst,
        args.unmon_inst if args.open_world else 0,
        args.suffix,
        100
    )
    
    logger.info(f"数据集大小: {len(flist)}, 类别数: {len(np.unique(labels))}")
    
    # 创建数据集和加载器
    dataset = TraceDataset(flist, labels)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True if args.use_gpu else False
    )
    
    # 创建训练器
    logger.info("初始化FG-GAN训练器...")
    trainer = ImprovedFGGAN(
        args,
        generator_g1_path=args.generator_g1,
        proxy_model_path=args.proxy_model,
    )
    
    # 训练循环
    logger.info(f"开始训练，总轮数: {args.epochs}")
    
    for epoch in range(1, args.epochs + 1):
        # 训练一个epoch
        epoch_losses = {
            'loss_D': 0.0,
            'gp': 0.0,
            'loss_G2_gan': 0.0,
            'loss_G2_adv': 0.0,
            'loss_G2_total': 0.0,
            'asr': 0.0
        }
        
        n_batches = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        
        for traces, labels in progress_bar:
            labels = labels.to(trainer.device)
            
            # 训练一个batch
            batch_losses = trainer.train_batch(
                traces, 
                labels,
                n_critic=args.n_critic,
            )
            
            # 累积损失
            for key in epoch_losses:
                epoch_losses[key] += batch_losses[key]
            n_batches += 1
            
            # 更新进度条
            postfix_dict = {
                'D': f"{batch_losses['loss_D']:.3f}",
                'G': f"{batch_losses['loss_G2_total']:.3f}",
                'ASR': f"{batch_losses['asr']:.2%}"
            }
            # 如果有真实ASR，一并显示
            if 'asr_real' in batch_losses:
                postfix_dict['ASR_real'] = f"{batch_losses['asr_real']:.2%}"
            progress_bar.set_postfix(postfix_dict)
        
        # 计算平均
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
        
        # 记录历史
        for key in trainer.history:
            if key in epoch_losses:
                trainer.history[key].append(epoch_losses[key])
        
        # 打印统计
        log_msg = (
            f"Epoch {epoch}/{args.epochs} | "
            f"D: {epoch_losses['loss_D']:.4f} | "
            f"GP: {epoch_losses['gp']:.4f} | "
            f"G_GAN: {epoch_losses['loss_G2_gan']:.4f} | "
            f"G_ADV: {epoch_losses['loss_G2_adv']:.4f} | "
            f"ASR: {epoch_losses['asr']:.2%}"
        )
        
        # 如果有真实ASR，一并显示
        if 'asr_real' in epoch_losses:
            log_msg += f" | ASR_real: {epoch_losses['asr_real']:.2%}"
        
        logger.info(log_msg)
        
        # 定期保存
        if epoch % args.save_interval == 0:
            save_path = os.path.join(args.checkpoints, f'fggan_epoch_{epoch}.pth')
            trainer.save_model(save_path)
            logger.info(f"模型已保存: {save_path}")
        
        # 提前停止: 如果ASR超过96%，自动停止
        if epoch_losses['asr'] > 1.00:
            logger.info(f"\n⚠️ Epoch {epoch}达到，ASR={epoch_losses['asr']:.2%} > 96%，自动停止训练")
            # 保存最终模型
            asr_value = epoch_losses['asr'] * 100
            final_path = os.path.join(args.checkpoints, f'fggan_epoch_{epoch}_asr{asr_value:.1f}.pth')
            trainer.save_model(final_path)
            logger.info(f"最终模型已保存: {final_path}")
            break
    
    # 保存最终模型
    final_path = os.path.join(args.checkpoints, 'fggan_final.pth')
    trainer.save_model(final_path)
    logger.info(f"训练完成! 最终模型: {final_path}")
    
    # 保存训练历史
    import json
    history_path = os.path.join(args.checkpoints, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(trainer.history, f, indent=2)
    logger.info(f"训练历史已保存: {history_path}")


if __name__ == '__main__':
    main()
