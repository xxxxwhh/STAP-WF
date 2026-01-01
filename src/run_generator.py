"""
流量生成器训练与验证主程序
包含完整的训练流程、验证评估与可视化
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from traffic_generator import TrafficGeneratorTrainer, create_argparse_for_generator
from utils.data import MyDataset
from utils.general import get_flist_label, seed_everything
from utils.logger import init_logger


def split_dataset(dataset: MyDataset, val_ratio: float = 0.1, seed: int = 42):
    """
    划分训练集与验证集
    Args:
        dataset: 完整数据集
        val_ratio: 验证集比例
        seed: 随机种子
    Returns:
        train_dataset, val_dataset
    """
    n_samples = len(dataset)
    indices = np.arange(n_samples)
    labels = dataset.labels
    
    # 分层划分保持类别比例
    train_idx, val_idx = train_test_split(
        indices, 
        test_size=val_ratio, 
        stratify=labels,
        random_state=seed
    )
    
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    
    return train_dataset, val_dataset


def validate_generator(trainer: TrafficGeneratorTrainer, val_loader: DataLoader, epoch: int):
    """
    在验证集上评估生成器
    Args:
        trainer: 训练器实例
        val_loader: 验证集DataLoader
        epoch: 当前轮次
    Returns:
        val_metrics: 验证指标字典
    """
    trainer.generator.eval()
    
    val_metrics = {
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
    
    with torch.no_grad():
        for real_tam, labels in val_loader:
            n_batches += 1
            batch_size = real_tam.size(0)
            
            # 移动到设备
            # feature_transform已返回[2, seq_length]格式，DataLoader会添加batch维度变成[B, 2, T]
            real_tam = real_tam.to(trainer.device).float()  # [B, 2, T]
            labels = labels.to(trainer.device).long()
            
            # 生成样本
            noise = torch.randn(batch_size, trainer.noise_dim, device=trainer.device)
            labels_onehot = trainer._labels_to_onehot(labels)
            fake_tam = trainer.generator(noise, labels_onehot)
            
            # 计算损失
            losses = trainer.compute_losses(fake_tam, real_tam, labels)
            
            total_loss = (
                1.0 * losses['outgoing_loss'] +
                1.0 * losses['incoming_loss'] +
                0.1 * losses['count_loss'] +
                0.5 * losses['ratio_loss'] +
                0.7 * losses['semantic_loss']
            )
            
            # 累计
            val_metrics['total_loss'] += total_loss.item()
            val_metrics['outgoing_loss'] += losses['outgoing_loss'].item()
            val_metrics['incoming_loss'] += losses['incoming_loss'].item()
            val_metrics['count_loss'] += losses['count_loss'].item()
            val_metrics['ratio_loss'] += losses['ratio_loss'].item()
            val_metrics['semantic_loss'] += losses['semantic_loss'].item()
            val_metrics['contrastive_loss'] += losses.get('contrastive_loss', 0.0) if isinstance(losses.get('contrastive_loss', 0.0), float) else losses.get('contrastive_loss', torch.tensor(0.0)).item()
            val_metrics['sparsity_loss'] += losses.get('sparsity_loss', torch.tensor(0.0)).item()
            val_metrics['mean_std_loss'] += losses.get('mean_std_loss', torch.tensor(0.0)).item()
            val_metrics['nonzero_loss'] += losses.get('nonzero_loss', torch.tensor(0.0)).item()
            val_metrics['proxy_accuracy'] += losses['proxy_accuracy']
    
    # 平均
    for key in val_metrics:
        val_metrics[key] /= n_batches
    
    trainer.logger.info(
        f"[Validation Epoch {epoch}] "
        f"Loss: {val_metrics['total_loss']:.4f} | "
        f"Acc: {val_metrics['proxy_accuracy']:.3f} | "
        f"Out: {val_metrics['outgoing_loss']:.4f} | In: {val_metrics['incoming_loss']:.4f} | "
        f"Cnt: {val_metrics['count_loss']:.4f} | Ratio: {val_metrics['ratio_loss']:.4f} | "
        f"Sem: {val_metrics['semantic_loss']:.4f} | Con: {val_metrics['contrastive_loss']:.4f} | "
        f"L1: {val_metrics['sparsity_loss']:.4f} | MS: {val_metrics['mean_std_loss']:.4f} | NZ: {val_metrics['nonzero_loss']:.4f}"
    )
    
    return val_metrics


def train_with_validation(args):
    """
    带验证的完整训练流程
    """
    logger = init_logger('TrainGenerator')
    
    # 设置随机种子
    seed_everything(42)
    
    # 准备数据
    logger.info("正在加载数据...")
    unmon_inst = args.unmon_inst if args.open_world else 0
    flist, labels = get_flist_label(
        args.data_path, 
        mon_cls=args.mon_classes,
        mon_inst=args.mon_inst,
        unmon_inst=unmon_inst,
        suffix=args.suffix
    )
    # ss
    
    logger.info(f"总样本数: {len(flist)}, 类别数: {args.mon_classes + 1 if args.open_world else args.mon_classes}")
    
    # 创建数据集
    dataset = MyDataset(args, flist, labels, TrafficGeneratorTrainer.extract_tam)
    
    # 划分训练集与验证集
    train_dataset, val_dataset = split_dataset(dataset, val_ratio=0.1, seed=42)
    logger.info(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.workers,
        pin_memory=True if args.use_gpu else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True if args.use_gpu else False
    )
    
    # 创建训练器
    logger.info("初始化训练器...")
    trainer = TrafficGeneratorTrainer(args, proxy_checkpoint=args.proxy_checkpoint)
    
    # 验证代理分类器对真实TAM的准确率
    logger.info("验证代理分类器在真实TAM上的性能...")
    proxy_acc_train = trainer.validate_proxy_classifier(train_loader)
    proxy_acc_val = trainer.validate_proxy_classifier(val_loader)
    logger.info(f"代理分类器准确率 - 训练集: {proxy_acc_train:.4f}, 验证集: {proxy_acc_val:.4f}")
    
    if proxy_acc_train < 0.1:
        logger.warning("警告: 代理分类器在真实TAM上的准确率很低! 请检查checkpoint是否正确")
    
    # 训练循环
    logger.info(f"开始训练，总轮数: {args.epochs}")
    
    best_val_acc = 0.0
    best_epoch = 0
    val_history = []
    
    for epoch in range(1, args.epochs + 1):
        # 训练一个epoch
        epoch_losses = trainer.train_epoch(train_loader, epoch)
        
        # 记录训练历史
        for key in trainer.history:
            trainer.history[key].append(epoch_losses[key])
        
        # 验证
        val_metrics = validate_generator(trainer, val_loader, epoch)
        val_history.append(val_metrics)
        
        # 保存最佳模型
        if val_metrics['proxy_accuracy'] > best_val_acc:
            best_val_acc = val_metrics['proxy_accuracy']
            best_epoch = epoch
            best_model_path = os.path.join(args.checkpoints, 'best_generator.pth')
            trainer.save_generator(best_model_path)
            logger.info(f"*** 新的最佳模型保存于 Epoch {epoch}, 验证准确率: {best_val_acc:.4f} ***")
        
        # 定期保存checkpoint
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(args.checkpoints, f'generator_epoch_{epoch}.pth')
            trainer.save_generator(checkpoint_path)
    
    # 保存最终模型
    final_path = os.path.join(args.checkpoints, 'final_generator.pth')
    trainer.save_generator(final_path)
    
    # 保存验证历史
    val_history_path = os.path.join(args.checkpoints, 'val_history.npy')
    np.save(val_history_path, val_history)
    
    logger.info(f"训练完成! 最佳验证准确率: {best_val_acc:.4f} (Epoch {best_epoch})")
    
    return trainer, val_history, val_loader


def main():
    parser = create_argparse_for_generator()
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.checkpoints, exist_ok=True)
    
    # 训练
    trainer, val_history, val_loader = train_with_validation(args)
    
    # 可视化结果
    from visualize_generator import visualize_training_results, visualize_generated_samples, visualize_tsne_embeddings
    
    visualize_training_results(trainer, val_history, save_dir=args.checkpoints)
    visualize_generated_samples(trainer, num_classes=5, save_dir=args.checkpoints)
    # 新增：t-SNE嵌入可视化（使用验证集真实样本作为参照）
    try:
        visualize_tsne_embeddings(trainer, val_loader, num_classes=5, per_class_samples=50, save_dir=args.checkpoints)
    except Exception as e:
        print(f"t-SNE可视化失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("所有任务完成!")


if __name__ == '__main__':
    main()
