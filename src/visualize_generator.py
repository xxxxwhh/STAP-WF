"""
流量生成器可视化工具
包含训练历史可视化、生成样本质量评估、TAM矩阵对比等
"""
import os
from typing import List, Dict, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.manifold import TSNE

from traffic_generator import TrafficGeneratorTrainer


# 设置绘图样式
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False


def visualize_training_results(trainer: TrafficGeneratorTrainer, 
                               val_history: List[Dict], 
                               save_dir: str = './results'):
    """
    可视化训练历史与验证曲线
    Args:
        trainer: 训练器实例
        val_history: 验证历史列表
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 提取训练历史
    train_loss = trainer.history['total_loss']
    train_acc = trainer.history['proxy_accuracy']
    
    # 提取验证历史
    val_loss = [v['total_loss'] for v in val_history]
    val_acc = [v['proxy_accuracy'] for v in val_history]
    
    epochs = range(1, len(train_loss) + 1)
    
    # 创建图形
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. 总损失曲线
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, train_loss, label='Train Loss', linewidth=2, color='#1f77b4')
    ax1.plot(epochs, val_loss, label='Val Loss', linewidth=2, color='#ff7f0e', linestyle='--')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Total Loss', fontsize=12)
    ax1.set_title('Total Loss (Train vs Val)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2. 代理准确率曲线
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, train_acc, label='Train Acc', linewidth=2, color='#2ca02c')
    ax2.plot(epochs, val_acc, label='Val Acc', linewidth=2, color='#d62728', linestyle='--')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Proxy Accuracy', fontsize=12)
    ax2.set_title('Proxy Model Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # 3. 出站损失
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, trainer.history['outgoing_loss'], linewidth=2, color='#9467bd')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Outgoing Loss', fontsize=12)
    ax3.set_title('Outgoing Channel Loss', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. 入站损失
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(epochs, trainer.history['incoming_loss'], linewidth=2, color='#8c564b')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Incoming Loss', fontsize=12)
    ax4.set_title('Incoming Channel Loss', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. 计数匹配损失
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(epochs, trainer.history['count_loss'], linewidth=2, color='#e377c2')
    ax5.set_xlabel('Epoch', fontsize=12)
    ax5.set_ylabel('Count Loss', fontsize=12)
    ax5.set_title('Total Count Matching Loss', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. 方向比例损失
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(epochs, trainer.history['ratio_loss'], linewidth=2, color='#7f7f7f')
    ax6.set_xlabel('Epoch', fontsize=12)
    ax6.set_ylabel('Ratio Loss (KL)', fontsize=12)
    ax6.set_title('Direction Ratio Loss', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # 7. 语义损失
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(epochs, trainer.history['semantic_loss'], linewidth=2, color='#bcbd22')
    ax7.set_xlabel('Epoch', fontsize=12)
    ax7.set_ylabel('Semantic Loss (CE)', fontsize=12)
    ax7.set_title('Semantic Consistency Loss', fontsize=14, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    # 8. 组合损失对比
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(epochs, trainer.history['outgoing_loss'], label='Outgoing', linewidth=1.5, alpha=0.7)
    ax8.plot(epochs, trainer.history['incoming_loss'], label='Incoming', linewidth=1.5, alpha=0.7)
    ax8.plot(epochs, trainer.history['semantic_loss'], label='Semantic', linewidth=1.5, alpha=0.7)
    # 新增：对比损失与分布约束
    if 'contrastive_loss' in trainer.history:
        ax8.plot(epochs, trainer.history['contrastive_loss'], label='Contrastive', linewidth=1.5, alpha=0.7)
    if 'sparsity_loss' in trainer.history:
        ax8.plot(epochs, trainer.history['sparsity_loss'], label='L1', linewidth=1.5, alpha=0.7)
    if 'mean_std_loss' in trainer.history:
        ax8.plot(epochs, trainer.history['mean_std_loss'], label='MeanStd', linewidth=1.5, alpha=0.7)
    if 'nonzero_loss' in trainer.history:
        ax8.plot(epochs, trainer.history['nonzero_loss'], label='Nonzero', linewidth=1.5, alpha=0.7)
    ax8.set_xlabel('Epoch', fontsize=12)
    ax8.set_ylabel('Loss', fontsize=12)
    ax8.set_title('Loss Components Comparison', fontsize=14, fontweight='bold')
    ax8.legend(fontsize=10, ncol=2)
    ax8.grid(True, alpha=0.3)
    
    # 9. 训练vs验证准确率差异
    ax9 = fig.add_subplot(gs[2, 2])
    acc_gap = np.array(train_acc) - np.array(val_acc)
    ax9.plot(epochs, acc_gap, linewidth=2, color='#17becf')
    ax9.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax9.set_xlabel('Epoch', fontsize=12)
    ax9.set_ylabel('Acc Gap (Train - Val)', fontsize=12)
    ax9.set_title('Overfitting Indicator', fontsize=14, fontweight='bold')
    ax9.grid(True, alpha=0.3)
    
    plt.suptitle('Traffic Generator Training History', fontsize=18, fontweight='bold', y=0.995)
    
    save_path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"训练历史可视化已保存至: {save_path}")


def visualize_generated_samples(trainer: TrafficGeneratorTrainer,
                                num_classes: int = 5,
                                samples_per_class: int = 3,
                                save_dir: str = './results'):
    """
    可视化生成的TAM样本
    Args:
        trainer: 训练器实例
        num_classes: 可视化的类别数
        samples_per_class: 每类生成样本数
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 选择类别（均匀分布，包含未监控类）
    if trainer.num_classes > num_classes:
        class_indices = np.linspace(0, trainer.num_classes - 1, num_classes, dtype=int)
    else:
        class_indices = np.arange(trainer.num_classes)
    
    num_classes = len(class_indices)
    
    # 创建图形
    fig, axes = plt.subplots(num_classes, samples_per_class, 
                            figsize=(samples_per_class * 5, num_classes * 3))
    
    if num_classes == 1:
        axes = axes.reshape(1, -1)
    if samples_per_class == 1:
        axes = axes.reshape(-1, 1)
    
    for i, class_idx in enumerate(class_indices):
        # 生成样本
        labels = torch.tensor([class_idx] * samples_per_class)
        samples = trainer.generate_samples(labels, n_samples=samples_per_class)  # [N, 2, T]
        
        for j in range(samples_per_class):
            ax = axes[i, j]
            tam = samples[j].numpy()  # [2, T]
            
            # 计算非零区域
            outgoing = tam[0, :]
            incoming = tam[1, :]
            nonzero_mask = (outgoing > 0.01) | (incoming > 0.01)
            
            if nonzero_mask.sum() > 0:
                first_nonzero = np.where(nonzero_mask)[0][0]
                last_nonzero = np.where(nonzero_mask)[0][-1]
                plot_start = max(0, first_nonzero - 50)
                plot_end = min(len(outgoing), last_nonzero + 50)
            else:
                plot_start, plot_end = 0, min(500, len(outgoing))
            
            x = np.arange(plot_start, plot_end)
            
            # 绘制双通道
            ax.fill_between(x, 0, outgoing[plot_start:plot_end], 
                           alpha=0.6, color='#1f77b4', label='Outgoing')
            ax.fill_between(x, 0, -incoming[plot_start:plot_end], 
                           alpha=0.6, color='#ff7f0e', label='Incoming')
            
            # 标题
            class_type = 'Unmonitored' if class_idx == trainer.num_classes - 1 else f'Class {class_idx}'
            ax.set_title(f'{class_type} - Sample {j+1}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Time Slot', fontsize=10)
            ax.set_ylabel('Packet Count', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            if i == 0 and j == 0:
                ax.legend(fontsize=9, loc='upper right')
            
            # 统计信息
            total_out = outgoing.sum()
            total_in = incoming.sum()
            ax.text(0.02, 0.98, f'Out: {total_out:.0f}\nIn: {total_in:.0f}',
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Generated TAM Samples by Class', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    save_path = os.path.join(save_dir, 'generated_samples.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"生成样本可视化已保存至: {save_path}")


def compare_real_vs_generated(trainer: TrafficGeneratorTrainer,
                              real_loader,
                              num_samples: int = 3,
                              save_dir: str = './results'):
    """
    对比真实与生成的TAM样本
    Args:
        trainer: 训练器实例
        real_loader: 真实数据DataLoader
        num_samples: 对比样本数
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取真实样本
    real_tam_batch, real_labels = next(iter(real_loader))
    real_tam = real_tam_batch[:num_samples].to(trainer.device)
    labels = real_labels[:num_samples].to(trainer.device)
    
    # 生成对应标签的样本
    gen_tam = trainer.generate_samples(labels, n_samples=num_samples).to(trainer.device)
    
    # 创建对比图
    fig, axes = plt.subplots(num_samples, 2, figsize=(14, num_samples * 3))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        label = labels[i].item()
        
        # 真实TAM
        ax_real = axes[i, 0]
        real_data = real_tam[i].cpu().numpy()
        
        nonzero = (real_data[0, :] > 0) | (real_data[1, :] > 0)
        if nonzero.sum() > 0:
            first, last = np.where(nonzero)[0][0], np.where(nonzero)[0][-1]
            start, end = max(0, first - 30), min(len(real_data[0]), last + 30)
        else:
            start, end = 0, min(500, len(real_data[0]))
        
        x = np.arange(start, end)
        ax_real.fill_between(x, 0, real_data[0, start:end], alpha=0.6, color='#2ca02c', label='Outgoing')
        ax_real.fill_between(x, 0, -real_data[1, start:end], alpha=0.6, color='#d62728', label='Incoming')
        ax_real.set_title(f'Real - Class {label}', fontsize=12, fontweight='bold')
        ax_real.set_ylabel('Count', fontsize=10)
        ax_real.legend(fontsize=9)
        ax_real.grid(True, alpha=0.3)
        
        # 生成TAM
        ax_gen = axes[i, 1]
        gen_data = gen_tam[i].cpu().numpy()
        
        ax_gen.fill_between(x, 0, gen_data[0, start:end], alpha=0.6, color='#2ca02c', label='Outgoing')
        ax_gen.fill_between(x, 0, -gen_data[1, start:end], alpha=0.6, color='#d62728', label='Incoming')
        ax_gen.set_title(f'Generated - Class {label}', fontsize=12, fontweight='bold')
        ax_gen.set_ylabel('Count', fontsize=10)
        ax_gen.legend(fontsize=9)
        ax_gen.grid(True, alpha=0.3)
        
        # 统计对比
        real_total = real_data.sum()
        gen_total = gen_data.sum()
        ax_real.text(0.02, 0.98, f'Total: {real_total:.0f}', transform=ax_real.transAxes,
                    fontsize=9, va='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        ax_gen.text(0.02, 0.98, f'Total: {gen_total:.0f}', transform=ax_gen.transAxes,
                   fontsize=9, va='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.suptitle('Real vs Generated TAM Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    save_path = os.path.join(save_dir, 'real_vs_generated.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"真实vs生成对比图已保存至: {save_path}")


def visualize_class_distribution(trainer: TrafficGeneratorTrainer,
                                 num_samples_per_class: int = 50,
                                 save_dir: str = './results'):
    """
    可视化不同类别生成样本的统计分布
    Args:
        trainer: 训练器实例
        num_samples_per_class: 每类生成样本数
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 选择若干代表性类别
    num_vis_classes = min(10, trainer.num_classes)
    class_indices = np.linspace(0, trainer.num_classes - 1, num_vis_classes, dtype=int)
    
    # 统计特征
    total_counts = []
    outgoing_ratios = []
    class_labels = []
    
    for cls_idx in class_indices:
        labels = torch.tensor([cls_idx] * num_samples_per_class)
        samples = trainer.generate_samples(labels, n_samples=num_samples_per_class)  # [N, 2, T]
        
        for sample in samples:
            out_count = sample[0, :].sum().item()
            in_count = sample[1, :].sum().item()
            total = out_count + in_count
            
            total_counts.append(total)
            outgoing_ratios.append(out_count / (total + 1e-8))
            class_labels.append(cls_idx)
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. 总计数分布
    ax1 = axes[0]
    for cls_idx in class_indices:
        mask = np.array(class_labels) == cls_idx
        counts = np.array(total_counts)[mask]
        ax1.hist(counts, bins=30, alpha=0.5, label=f'Class {cls_idx}')
    
    ax1.set_xlabel('Total Packet Count', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Total Count Distribution by Class', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # 2. 出站比例分布
    ax2 = axes[1]
    for cls_idx in class_indices:
        mask = np.array(class_labels) == cls_idx
        ratios = np.array(outgoing_ratios)[mask]
        ax2.hist(ratios, bins=30, alpha=0.5, label=f'Class {cls_idx}')
    
    ax2.set_xlabel('Outgoing Ratio', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Outgoing/Incoming Ratio by Class', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Generated Sample Statistics by Class', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_path = os.path.join(save_dir, 'class_distribution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"类别分布统计图已保存至: {save_path}")


def plot_confusion_matrix(trainer: TrafficGeneratorTrainer,
                          test_loader,
                          save_dir: str = './results'):
    """
    生成混淆矩阵评估代理模型在生成样本上的表现
    Args:
        trainer: 训练器实例
        test_loader: 测试集DataLoader
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for real_tam, labels in test_loader:
            batch_size = real_tam.size(0)
            labels = labels.to(trainer.device)
            
            # 生成样本
            noise = torch.randn(batch_size, trainer.noise_dim, device=trainer.device)
            labels_onehot = trainer._labels_to_onehot(labels)
            fake_tam = trainer.generator(noise, labels_onehot)
            
            # 代理预测
            logits = trainer.proxy_classifier(fake_tam)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算混淆矩阵
    from sklearn.metrics import confusion_matrix
    
    # 若类别太多，只显示前20个
    unique_labels = np.unique(all_labels)
    if len(unique_labels) > 20:
        top_classes = unique_labels[:20]
        mask = np.isin(all_labels, top_classes)
        all_labels = np.array(all_labels)[mask]
        all_preds = np.array(all_preds)[mask]
    
    cm = confusion_matrix(all_labels, all_preds)
    
    # 归一化
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
    
    # 绘制
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=False, fmt='.2f', cmap='Blues', 
               xticklabels=unique_labels[:len(cm)], 
               yticklabels=unique_labels[:len(cm)],
               cbar_kws={'label': 'Normalized Count'})
    
    ax.set_xlabel('Predicted Class', fontsize=12)
    ax.set_ylabel('True Class', fontsize=12)
    ax.set_title('Confusion Matrix (Generated Samples on Proxy Model)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    accuracy = np.trace(cm) / cm.sum()
    print(f"混淆矩阵已保存至: {save_path}")
    print(f"代理模型在生成样本上的准确率: {accuracy:.4f}")


def visualize_tsne_embeddings(trainer: TrafficGeneratorTrainer,
                              real_loader,
                              num_classes: int = 5,
                              per_class_samples: int = 50,
                              save_dir: str = './results'):
    """
    使用代理模型logits作为嵌入，对真实与生成样本做t-SNE降维可视化。
    """
    os.makedirs(save_dir, exist_ok=True)
    trainer.proxy_classifier.eval()
    X = []
    Y = []
    T = []  # 类型: 'Real'/'Fake'
    with torch.no_grad():
        # 采样若干类别
        class_indices = np.linspace(0, trainer.num_classes - 1, num_classes, dtype=int)
        # 从真实数据收集嵌入
        collected = 0
        for real_tam, labels in real_loader:
            real_tam = real_tam.to(trainer.device).float()
            labels = labels.to(trainer.device).long()
            logits = trainer.proxy_classifier(real_tam)
            X.append(logits.cpu().numpy())
            Y.append(labels.cpu().numpy())
            T.append(np.array(['Real'] * labels.size(0)))
            collected += labels.size(0)
            if collected >= num_classes * per_class_samples:
                break
        # 生成对应标签的假样本嵌入
        labels_all = np.concatenate(Y)
        labels_all = labels_all[:num_classes * per_class_samples]
        labels_tensor = torch.tensor(labels_all, device=trainer.device)
        noise = torch.randn(labels_tensor.size(0), trainer.noise_dim, device=trainer.device)
        labels_onehot = trainer._labels_to_onehot(labels_tensor)
        fake_tam = trainer.generator(noise, labels_onehot)
        fake_logits = trainer.proxy_classifier(fake_tam)
        X.append(fake_logits.cpu().numpy())
        Y.append(labels_all)
        T.append(np.array(['Fake'] * labels_tensor.size(0)))
    # 合并
    X = np.concatenate(X)
    Y = np.concatenate(Y)
    T = np.concatenate(T)
    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
    X2 = tsne.fit_transform(X)
    # 绘制
    fig, ax = plt.subplots(figsize=(10, 8))
    for t_type, color, marker in [('Real', '#1f77b4', 'o'), ('Fake', '#ff7f0e', 'x')]:
        mask = (T == t_type)
        ax.scatter(X2[mask, 0], X2[mask, 1], c=[color], s=12, marker=marker, alpha=0.7, label=t_type)
    ax.set_title('t-SNE of Proxy Logits (Real vs Generated)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_path = os.path.join(save_dir, 'tsne_embeddings.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"t-SNE嵌入图已保存至: {save_path}")

if __name__ == '__main__':
    # 示例用法
    import argparse
    from torch.utils.data import DataLoader
    from traffic_generator import TrafficGeneratorTrainer, create_argparse_for_generator
    from utils.data import MyDataset
    from utils.general import get_flist_label
    
    parser = create_argparse_for_generator()
    args = parser.parse_args()
    
    # 加载训练好的生成器
    trainer = TrafficGeneratorTrainer(args, proxy_checkpoint=args.proxy_checkpoint)
    checkpoint_path = os.path.join(args.checkpoints, 'best_generator.pth')
    
    if os.path.exists(checkpoint_path):
        trainer.load_generator(checkpoint_path)
        
        # 准备测试数据
        flist, labels = get_flist_label(args.data_path, args.mon_classes, 
                                       args.mon_inst, args.unmon_inst if args.open_world else 0,
                                       args.suffix)
        dataset = MyDataset(args, flist, labels, TrafficGeneratorTrainer.extract_tam)
        test_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
        
        # 生成所有可视化
        visualize_generated_samples(trainer, num_classes=5, save_dir=args.checkpoints)
        compare_real_vs_generated(trainer, test_loader, num_samples=3, save_dir=args.checkpoints)
        visualize_class_distribution(trainer, num_samples_per_class=50, save_dir=args.checkpoints)
        plot_confusion_matrix(trainer, test_loader, save_dir=args.checkpoints)
        
        print("所有可视化完成!")
    else:
        print(f"未找到模型权重: {checkpoint_path}")
