"""诊断 G₁ 生成质量 - 类间多样性评估"""
import torch
import numpy as np
from traffic_generator import ConditionalTrafficGenerator
from attacks.modules import RFNet
from utils.general import get_flist_label, feature_transform, parse_trace
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 加载 G₁
g1_path = input("输入 G₁ checkpoint 路径: ").strip()
checkpoint = torch.load(g1_path, map_location='cpu')
num_classes = checkpoint['num_classes']
noise_dim = checkpoint['noise_dim']
seq_length = checkpoint['seq_length']

G1 = ConditionalTrafficGenerator(num_classes, noise_dim, seq_length)
G1.load_state_dict(checkpoint['generator_state_dict'])
G1.eval()

# 加载 RFNet
rf_path = input("输入 RFNet checkpoint 路径: ").strip()
rf = RFNet(num_classes=num_classes)
ckpt = torch.load(rf_path, map_location='cpu')
rf.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
rf.eval()

print("\n" + "="*60)
print("🔍 类间多样性评估（不同类别是否区分开）")
print("="*60)

# 随机选择30个类别
n_classes = 30
n_samples_per_class = 100
np.random.seed(42)
selected_classes = np.random.choice(min(num_classes, 100), size=n_classes, replace=False)
print(f"\n选中的 {n_classes} 个类别: {selected_classes.tolist()}")
print(f"每个类别生成 {n_samples_per_class} 个样本，总计 {n_classes * n_samples_per_class} 个样本")

# 为每个类别生成100个样本
print(f"\n生成 {n_classes * n_samples_per_class} 个假样本...")
fake_tams_list = []
fake_labels_list = []

for cls in selected_classes:
    cls_labels = torch.full((n_samples_per_class,), cls, dtype=torch.long)
    cls_noise = torch.randn(n_samples_per_class, noise_dim)
    cls_onehot = torch.zeros(n_samples_per_class, num_classes)
    cls_onehot.scatter_(1, cls_labels.unsqueeze(1), 1.0)
    
    with torch.no_grad():
        cls_tams = G1(cls_noise, cls_onehot)
    
    fake_tams_list.append(cls_tams)
    fake_labels_list.extend([cls] * n_samples_per_class)

fake_tams = torch.cat(fake_tams_list, dim=0)
fake_labels = np.array(fake_labels_list)

with torch.no_grad():
    fake_logits = rf(fake_tams)

# 加载对应的真实样本（为每个选中类别加载样本）
data_path = input("输入数据路径: ").strip()
print(f"\n为 {n_classes} 个类别加载真实样本...")

real_tams_list = []
real_labels_list = []

for cls in selected_classes:
    # 加载该类别的样本
    flist, labels = get_flist_label(data_path, min(num_classes, 100), 90, 0, '.cell', 200)
    cls_files = [f for f, l in zip(flist, labels) if l == cls][:n_samples_per_class]
    
    for f in cls_files:
        trace = parse_trace(f)
        tam = feature_transform(trace, 'tam', seq_length)
        real_tams_list.append(torch.from_numpy(tam).float())
        real_labels_list.append(cls)

if len(real_tams_list) > 0:
    real_tams = torch.stack(real_tams_list)
    real_labels = np.array(real_labels_list)
    print(f"加载了 {len(real_tams)} 个真实样本")
else:
    print("⚠️ 警告: 没有加载到真实样本")
    real_tams = torch.zeros(1, 2, seq_length)
    real_labels = np.array([0])

with torch.no_grad():
    real_logits = rf(real_tams)

# t-SNE 可视化
print("\n计算 t-SNE...")
all_logits = torch.cat([real_logits, fake_logits], dim=0).numpy()
all_labels = np.concatenate([real_labels, fake_labels])

tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_logits)-1))
embedded = tsne.fit_transform(all_logits)

# 绘制类间分布图
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

n_real = len(real_labels)
n_fake = len(fake_labels)

# 左图: 真实样本的类间分布
colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
for i, cls in enumerate(selected_classes):
    mask = real_labels == cls
    if mask.sum() > 0:
        axes[0].scatter(embedded[:n_real][mask, 0], embedded[:n_real][mask, 1],
                       color=colors[i], label=f'C{cls}', alpha=0.6, s=30)
axes[0].set_title(f'Real Samples - Inter-Class Distribution\n({n_classes} classes, {len(real_labels)} samples)', 
                 fontsize=12, fontweight='bold')
axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, ncol=2)
axes[0].grid(alpha=0.3)

# 右图: 生成样本的类间分布
for i, cls in enumerate(selected_classes):
    mask = fake_labels == cls
    if mask.sum() > 0:
        axes[1].scatter(embedded[n_real:][mask, 0], embedded[n_real:][mask, 1],
                       color=colors[i], label=f'C{cls}', alpha=0.6, s=30)
axes[1].set_title(f'Generated Samples - Inter-Class Distribution\n({n_classes} classes, {len(fake_labels)} samples)', 
                 fontsize=12, fontweight='bold')
axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, ncol=2)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('g1_inter_class_diversity.png', dpi=150, bbox_inches='tight')
print("\n✅ 类间分布图已保存: g1_inter_class_diversity.png")

# 统计分析
print("\n📊 类间分布统计:")
print(f"真实样本 - 类别数: {n_classes}, 样本数: {len(real_labels)}")
print(f"生成样本 - 类别数: {n_classes}, 样本数: {len(fake_labels)}")

# 计算类间方差（越大说明不同类别越分散）
real_inter_var = embedded[:n_real].var(axis=0).mean()
fake_inter_var = embedded[n_real:].var(axis=0).mean()
print(f"\n类间方差:")
print(f"  Real: {real_inter_var:.4f}")
print(f"  Fake: {fake_inter_var:.4f}")
print(f"  Ratio (Fake/Real): {fake_inter_var/real_inter_var:.2f}x")

if fake_inter_var / real_inter_var > 0.5:
    print("  ✅ 生成样本类间分布合理")
else:
    print("  ⚠️ 警告: 生成样本类间分布过于集中")

# 计算每个类别的中心距离（衡量类别可区分性）
print("\n类别中心距离分析:")
fake_centers = []
for cls in selected_classes:
    mask = fake_labels == cls
    if mask.sum() > 0:
        center = embedded[n_real:][mask].mean(axis=0)
        fake_centers.append(center)

if len(fake_centers) > 1:
    fake_centers = np.array(fake_centers)
    # 计算所有类别中心之间的平均距离
    from scipy.spatial.distance import pdist
    avg_dist = pdist(fake_centers).mean()
    print(f"  生成样本类别中心平均距离: {avg_dist:.4f}")
    if avg_dist > 5.0:
        print("  ✅ 类别间区分明显")
    else:
        print("  ⚠️ 警告: 类别间区分不明显")

print("\n" + "="*60)

# ===== 新增：更多可视化图表 =====

# 1. 混淆矩阵（Confusion Matrix）- 评估代理模型对生成样本的分类质量
from sklearn.metrics import confusion_matrix
import seaborn as sns

print("\n生成混淆矩阵...")
fake_pred = torch.argmax(fake_logits, dim=1).numpy()
cm = confusion_matrix(fake_labels, fake_pred, labels=selected_classes)

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=selected_classes, 
            yticklabels=selected_classes, ax=ax, cbar_kws={'label': 'Count'})
ax.set_xlabel('预测类别', fontsize=12)
ax.set_ylabel('真实类别', fontsize=12)
ax.set_title(f'G₁ 生成样本混淆矩阵\n对角线越亮说明分类越准', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('g1_confusion_matrix.png', dpi=150, bbox_inches='tight')
print("✅ 混淆矩阵已保存: g1_confusion_matrix.png")

# 2. 类别准确率条形图 - 展示每个类别的生成质量
print("\n生成类别准确率图...")
class_accs = []
for cls in selected_classes:
    mask = fake_labels == cls
    if mask.sum() > 0:
        acc = (fake_pred[mask] == cls).mean()
        class_accs.append(acc)
    else:
        class_accs.append(0)

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(range(n_classes), class_accs, color='steelblue', alpha=0.7)
ax.axhline(y=np.mean(class_accs), color='red', linestyle='--', linewidth=2, label=f'平均: {np.mean(class_accs):.2%}')
ax.set_xlabel('类别索引', fontsize=12)
ax.set_ylabel('准确率', fontsize=12)
ax.set_title('G₁ 生成样本的各类别准确率', fontsize=14, fontweight='bold')
ax.set_xticks(range(n_classes))
ax.set_xticklabels([f'C{c}' for c in selected_classes], rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('g1_class_accuracy.png', dpi=150, bbox_inches='tight')
print("✅ 类别准确率图已保存: g1_class_accuracy.png")

# 3. TAM分布对比（直方图）- 对比真实与生成样本的TAM统计分布
print("\n生成TAM分布对比图...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 3.1 幅度分布
axes[0, 0].hist(real_tams.flatten().numpy(), bins=50, alpha=0.6, label='Real', color='blue', density=True)
axes[0, 0].hist(fake_tams.flatten().numpy(), bins=50, alpha=0.6, label='Fake', color='orange', density=True)
axes[0, 0].set_xlabel('TAM 值', fontsize=11)
axes[0, 0].set_ylabel('密度', fontsize=11)
axes[0, 0].set_title('TAM 幅度分布', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].set_xlim(0, min(real_tams.max().item(), fake_tams.max().item()))

# 3.2 非零比例
real_nonzero = (real_tams > 0).float().mean(dim=(1, 2)).numpy()
fake_nonzero = (fake_tams > 0).float().mean(dim=(1, 2)).numpy()
axes[0, 1].hist(real_nonzero, bins=30, alpha=0.6, label='Real', color='blue', density=True)
axes[0, 1].hist(fake_nonzero, bins=30, alpha=0.6, label='Fake', color='orange', density=True)
axes[0, 1].set_xlabel('非零比例', fontsize=11)
axes[0, 1].set_ylabel('密度', fontsize=11)
axes[0, 1].set_title('TAM 稀疏性分布', fontsize=12, fontweight='bold')
axes[0, 1].legend()

# 3.3 均值分布
real_mean = real_tams.mean(dim=(1, 2)).numpy()
fake_mean = fake_tams.mean(dim=(1, 2)).numpy()
axes[1, 0].hist(real_mean, bins=30, alpha=0.6, label='Real', color='blue', density=True)
axes[1, 0].hist(fake_mean, bins=30, alpha=0.6, label='Fake', color='orange', density=True)
axes[1, 0].set_xlabel('TAM 均值', fontsize=11)
axes[1, 0].set_ylabel('密度', fontsize=11)
axes[1, 0].set_title('TAM 平均值分布', fontsize=12, fontweight='bold')
axes[1, 0].legend()

# 3.4 最大值分布
real_max = real_tams.max(dim=2)[0].max(dim=1)[0].numpy()
fake_max = fake_tams.max(dim=2)[0].max(dim=1)[0].numpy()
axes[1, 1].hist(real_max, bins=30, alpha=0.6, label='Real', color='blue', density=True)
axes[1, 1].hist(fake_max, bins=30, alpha=0.6, label='Fake', color='orange', density=True)
axes[1, 1].set_xlabel('TAM 最大值', fontsize=11)
axes[1, 1].set_ylabel('密度', fontsize=11)
axes[1, 1].set_title('TAM 峰值分布', fontsize=12, fontweight='bold')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('g1_tam_distribution.png', dpi=150, bbox_inches='tight')
print("✅ TAM分布图已保存: g1_tam_distribution.png")

# 4. Logits分布对比（箱线图）- 查看生成样本的置信度分布
print("\n生成Logits分布图...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 4.1 正确类别的logit分布
real_correct_logits = []
fake_correct_logits = []
for i, cls in enumerate(selected_classes):
    real_mask = real_labels == cls
    fake_mask = fake_labels == cls
    if real_mask.sum() > 0:
        real_correct_logits.extend(real_logits[real_mask, cls].numpy())
    if fake_mask.sum() > 0:
        fake_correct_logits.extend(fake_logits[fake_mask, cls].numpy())

axes[0].boxplot([real_correct_logits, fake_correct_logits], labels=['Real', 'Fake'])
axes[0].set_ylabel('Logit 值', fontsize=12)
axes[0].set_title('正确类别的 Logit 分布', fontsize=12, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# 4.2 最大错误类别的logit分布
real_max_wrong = []
fake_max_wrong = []
for i, cls in enumerate(selected_classes):
    real_mask = real_labels == cls
    fake_mask = fake_labels == cls
    if real_mask.sum() > 0:
        real_logits_cls = real_logits[real_mask].clone()
        real_logits_cls[:, cls] = -1e9
        real_max_wrong.extend(real_logits_cls.max(dim=1)[0].numpy())
    if fake_mask.sum() > 0:
        fake_logits_cls = fake_logits[fake_mask].clone()
        fake_logits_cls[:, cls] = -1e9
        fake_max_wrong.extend(fake_logits_cls.max(dim=1)[0].numpy())

axes[1].boxplot([real_max_wrong, fake_max_wrong], labels=['Real', 'Fake'])
axes[1].set_ylabel('Logit 值', fontsize=12)
axes[1].set_title('最大错误类别的 Logit 分布', fontsize=12, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('g1_logits_distribution.png', dpi=150, bbox_inches='tight')
print("✅ Logits分布图已保存: g1_logits_distribution.png")

# 5. 综合评估报告
print("\n" + "="*60)
print("📊 G₁ 生成器综合评估报告")
print("="*60)
print(f"1. 总体准确率: {(fake_pred == fake_labels).mean():.2%}")
print(f"2. 类别准确率范围: {np.min(class_accs):.2%} ~ {np.max(class_accs):.2%}")
print(f"3. TAM 均值 - Real: {real_mean.mean():.4f}, Fake: {fake_mean.mean():.4f}")
print(f"4. TAM 最大值 - Real: {real_max.mean():.2f}, Fake: {fake_max.mean():.2f}")
print(f"5. TAM 非零比 - Real: {real_nonzero.mean():.2%}, Fake: {fake_nonzero.mean():.2%}")
print(f"6. 类间方差比例: {fake_inter_var/real_inter_var:.2f}x")
print("\n✅ 所有评估图表已生成！")
