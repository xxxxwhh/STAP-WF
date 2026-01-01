"""
使用训练好的FG-GAN生成对抗样本并评估
"""
import argparse
import os

import numpy as np
import torch
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib

# 配置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

from fggan_improved import ImprovedFGGAN
from attacks.modules.df import DFNet
from attacks.modules.rf import RFNet
from attacks.modules.varcnn import VarCNNNet
from attacks.modules.awf import AWFNet
from attacks.modules.ares import ARESNet
from utils.general import get_flist_label, parse_trace, feature_transform
from utils.logger import init_logger


def extract_bursts(trace):
    """从trace中提取burst序列
    Args:
        trace: [N, 2] 数组，列0=时间，列1=包大小（带符号）
    Returns:
        bursts: List[dict]，每个burst包含：
            - 'direction': 1(出站) 或 -1(入站)
            - 'size': burst总字节数（绝对值）
            - 'length': burst包含的包数量
            - 'start_time': burst开始时间
            - 'end_time': burst结束时间
            - 'duration': burst持续时间
    """
    if len(trace) == 0:
        return []
    
    bursts = []
    current_direction = np.sign(trace[0, 1])
    current_start = 0
    current_size = 0
    
    for i in range(len(trace)):
        pkt_direction = np.sign(trace[i, 1])
        pkt_size = abs(trace[i, 1])
        
        # 方向改变或最后一个包
        if pkt_direction != current_direction or i == len(trace) - 1:
            # 保存当前burst
            if i == len(trace) - 1 and pkt_direction == current_direction:
                # 最后一个包且方向相同
                end_idx = i
                current_size += pkt_size
            else:
                # 方向改变
                end_idx = i - 1
            
            if end_idx >= current_start:
                bursts.append({
                    'direction': int(current_direction),
                    'size': int(current_size),
                    'length': end_idx - current_start + 1,
                    'start_time': float(trace[current_start, 0]),
                    'end_time': float(trace[end_idx, 0]),
                    'duration': float(trace[end_idx, 0] - trace[current_start, 0])
                })
            
            # 开始新burst
            if i < len(trace) - 1:
                current_direction = pkt_direction
                current_start = i
                current_size = pkt_size
    
    return bursts


def create_argparse():
    """创建参数解析器"""
    parser = argparse.ArgumentParser(description='FG-GAN对抗样本生成与评估')
    
    # 数据相关
    parser.add_argument('--data-path', type=str, required=True, help='数据目录路径')
    parser.add_argument('--mon-classes', type=int, default=100, help='监控类别数')
    parser.add_argument('--mon-inst', type=int, default=100, help='每类监控实例数')
    parser.add_argument('--unmon-inst', type=int, default=10000, help='未监控实例数')
    parser.add_argument('--open-world', action='store_true', help='开放世界模式')
    parser.add_argument('--suffix', type=str, default='.cell', help='文件后缀')
    
    # 模型相关
    parser.add_argument('--fggan-model', type=str, required=True,
                        help='训练好的FG-GAN模型路径')
    parser.add_argument('--generator-g1', type=str, required=True,
                        help='预训练traffic_generator权重路径')
    parser.add_argument('--proxy-model', type=str, required=True,
                        help='预训练代理模型权重路径 (RF或DF)')
    
    # 评估目标模型
    parser.add_argument('--target-models', type=str, nargs='+', 
                        default=['rf', 'tiktak', 'df', 'varcnn', 'awf', 'ares'],
                        help='需要评估的放击模型 (rf/tiktak/df/varcnn/awf/ares)')
    parser.add_argument('--model-rf', type=str, default='./src/checkpoints/best_model.pth',
                        help='RFNet模型路径')
    parser.add_argument('--model-tiktak', type=str, default='./src/checkpoints/tiktok_best.pth',
                        help='Tik-Tok模型路径')
    parser.add_argument('--model-df', type=str, default='./src/checkpoints/df_best.pth',
                        help='DF模型路径')
    parser.add_argument('--model-varcnn', type=str, default='./src/checkpoints/varcnn_best.pth',
                        help='VarCNN模型路径')
    parser.add_argument('--model-awf', type=str, default='./src/checkpoints/awf_best.pth',
                        help='AWF模型路径')
    parser.add_argument('--model-ares', type=str, default='./src/checkpoints/ares_best.pth',
                        help='ARES模型路径')
    
    # 输出相关
    parser.add_argument('--output-dir', type=str, default='./adversarial_samples',
                        help='对抗样本保存目录')
    parser.add_argument('--num-samples', type=int, default=1000,
                        help='生成样本数量 (0=全部)')
    parser.add_argument('--seq-length', type=int, default=5000,
                        help='序列长度')
    
    # 设备相关
    parser.add_argument('--use-gpu', action='store_true', help='使用GPU')
    parser.add_argument('--gpu', type=int, default=0, help='GPU设备ID')
    parser.add_argument('--amp', action='store_true', help='混合精度')
    
    return parser


def load_attack_models(args, device, num_classes=101, seq_length=5000, enable_gradcam=True):
    """加载多个攻击模型用于评估
    Args:
        enable_gradcam: 是否启用Grad-CAM (如果启用,RF模型将保持梯度)
    """
    models = {}
    logger = init_logger('LoadModels')
    
    for model_name in args.target_models:
        model_name_lower = model_name.lower()
        
        try:
            if model_name_lower == 'rf':
                model = RFNet(num_classes=num_classes).to(device)
                ckpt = torch.load(args.model_rf, map_location=device)
                state = ckpt.get('model_state_dict', ckpt)
                model.load_state_dict(state, strict=False)
                logger.info(f"已加载RF模型: {args.model_rf}")
                
            elif model_name_lower in ['tiktak', 'tiktok']:
                # Tik-Tok使用DFNet架构
                model = DFNet(length=8000, num_classes=num_classes, in_channels=1).to(device)
                ckpt = torch.load(args.model_tiktak, map_location=device)
                state = ckpt.get('model_state_dict', ckpt)
                model.load_state_dict(state, strict=False)
                logger.info(f"已加载Tik-Tok模型: {args.model_tiktak}")
                model_name_lower = 'tiktak'  # 统一命名
                
            elif model_name_lower == 'df':
                model = DFNet(length=8000, num_classes=num_classes, in_channels=1).to(device)
                ckpt = torch.load(args.model_df, map_location=device)
                state = ckpt.get('model_state_dict', ckpt)
                model.load_state_dict(state, strict=False)
                logger.info(f"已加载DF模型: {args.model_df}")
                
            elif model_name_lower == 'varcnn':
                model = VarCNNNet(num_classes=num_classes).to(device)
                ckpt = torch.load(args.model_varcnn, map_location=device)
                state = ckpt.get('model_state_dict', ckpt)
                model.load_state_dict(state, strict=False)
                logger.info(f"已加载VarCNN模型: {args.model_varcnn}")
                
            elif model_name_lower == 'awf':
                model = AWFNet(num_classes=num_classes).to(device)
                ckpt = torch.load(args.model_awf, map_location=device)
                state = ckpt.get('model_state_dict', ckpt)
                model.load_state_dict(state, strict=False)
                logger.info(f"已加载AWF模型: {args.model_awf}")
                
            elif model_name_lower == 'ares':
                model = ARESNet(num_classes=num_classes).to(device)
                ckpt = torch.load(args.model_ares, map_location=device)
                state = ckpt.get('model_state_dict', ckpt)
                model.load_state_dict(state, strict=False)
                logger.info(f"已加载ARES模型: {args.model_ares}")
                
            else:
                raise ValueError(f"不支持的模型: {model_name}")
            
            model.eval()
            
            # 关键修改: RF和DF模型如果用于Grad-CAM则保持梯度
            if model_name_lower in ['rf', 'df'] and enable_gradcam:
                # RF/DF模型用于Grad-CAM,需要梯度
                logger.info(f"{model_name_lower.upper()}模型保持梯度以支持Grad-CAM")
            else:
                # 其他模型禁用梯度 (仅用于评估)
                for p in model.parameters():
                    p.requires_grad = False
            
            models[model_name_lower] = model
            
        except Exception as e:
            logger.warning(f"加载模型 {model_name} 失败: {e}")
            continue
    
    return models


def main():
    # 解析参数
    parser = create_argparse()
    args = parser.parse_args()
    
    # 初始化logger
    logger = init_logger('GenerateAdversarial')
    logger.info(f"开始生成对抗样本: {args}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    
    # 限制样本数
    if args.num_samples > 0 and args.num_samples < len(flist):
        indices = np.random.choice(len(flist), args.num_samples, replace=False)
        flist = flist[indices]
        labels = labels[indices]
    
    logger.info(f"将生成 {len(flist)} 个对抗样本")
    
    # 创建训练器并加载模型
    logger.info("加载FG-GAN模型...")
    trainer = ImprovedFGGAN(
        args,
        generator_g1_path=args.generator_g1,
        proxy_model_path=args.proxy_model,
    )
    trainer.load_model(args.fggan_model)
    trainer.G2.eval()
    trainer.D.eval()
    
    # 加载目标攻击模型
    logger.info("加载目标攻击模型...")
    # 使用正确的类别数 (不加1)
    num_classes = args.mon_classes if not args.open_world else args.mon_classes + 1
    
    # 检查是否启用Grad-CAM
    use_gradcam = getattr(args, 'use_gradcam', True)
    
    target_models = load_attack_models(
        args, 
        trainer.device, 
        num_classes=num_classes,
        seq_length=args.seq_length if hasattr(args, 'seq_length') else 5000,
        enable_gradcam=use_gradcam  # 传递Grad-CAM启用状态
    )
    
    if len(target_models) == 0:
        logger.error("没有成功加载任何目标模型!")
        return
    
    logger.info(f"成功加载 {len(target_models)} 个目标模型: {list(target_models.keys())}")
    
    # 初始化Grad-CAM (用于关键位置引导)
    gradcam = None
    gradcam_model_type = 'rf'  # 使用DF模型 (更强大)
    
    # 优先使用DF模型,其次RF
    if use_gradcam:
        gradcam_model = None
        if 'rf' in target_models:
            gradcam_model = target_models['rf']
            gradcam_model_type = 'rf'
            logger.info("使用RF模型进行Grad-CAM分析")

        if gradcam_model is not None:
            try:
                from utils.gradcam import get_gradcam_for_model
                gradcam = get_gradcam_for_model(gradcam_model, model_type=gradcam_model_type)
                logger.info(f"Grad-CAM初始化完成 (模型: {gradcam_model_type.upper()})")
            except Exception as e:
                logger.warning(f"Grad-CAM初始化失败: {e}, 将不使用Grad-CAM引导")
                gradcam = None
        else:
            logger.info("没有可用的模型用于Grad-CAM")
    else:
        logger.info("Grad-CAM已禁用")
    
    # 生成对抗样本
    logger.info("开始生成对抗样本...")
    
    adv_traces_list = []
    adv_tams_list = []
    orig_tams_list = []
    orig_traces_uncropped_list = []  # 保存未裁剪的原始trace用于可视化
    original_labels = []
    # 为每个模型创建预测列表
    adv_predictions = {model_name: [] for model_name in target_models.keys()}
    
    # 数据包开销统计
    overhead_stats = {
        'num_dummy_packets': [],      # 插入的dummy包数量 (|P'| - |P|)
        'num_original_packets': [],   # 原始包数量 (|P|)
        'num_total_packets': [],      # 总包数 (|P'|)
        'data_overhead': [],          # 数据开销 O(D) = (|P'| - |P|) / |P|
        'time_overhead': [],          # 时间开销 (秒)
    }
    
    for i, (fpath, label) in enumerate(tqdm(zip(flist, labels), total=len(flist))):
        # 加载原始trace (同时获取裁剪前的长度和未裁剪的trace)
        original_trace, original_length_before_crop = parse_trace(fpath, return_original_length=True)
        
        # 保存未裁剪的原始trace用于后续计算和可视化
        original_trace_uncropped = np.loadtxt(fpath)  # 直接加载未裁剪的原始数据
        orig_traces_uncropped_list.append(original_trace_uncropped)
        
        # 计算原始TAM (用于t-SNE可视化)
        orig_tam = feature_transform(original_trace, feature_type='tam', seq_length=args.seq_length if hasattr(args, 'seq_length') else 5000)
        orig_tams_list.append(orig_tam)
        
        # 计算Grad-CAM重要性 (可选)
        gradcam_importance = None
        if gradcam is not None:
            try:
                # 根据模型类型提取不同特征
                if gradcam_model_type == 'df':
                    # DF使用方向序列
                    df_feat = feature_transform(original_trace, feature_type='df', seq_length=5000)
                    input_tensor = torch.from_numpy(df_feat).float().unsqueeze(0).to(trainer.device)
                elif gradcam_model_type == 'rf':
                    # RF使用TAM
                    original_tam = feature_transform(original_trace, feature_type='tam', seq_length=2000)
                    input_tensor = torch.from_numpy(original_tam).float().unsqueeze(0).to(trainer.device)
                else:
                    raise ValueError(f"Unsupported model type: {gradcam_model_type}")
                
                # 生成Grad-CAM热力图
                cam, _ = gradcam.generate_cam(input_tensor, target_class=label)
                
                # 直接使用原始CAM (已在gradcam.py中归一化到[0,1])
                gradcam_importance = cam
            except Exception as e:
                if i < 10:  # 只显示前10个警告
                    logger.warning(f"样本 {i} Grad-CAM计算失败: {e}")
                gradcam_importance = None
        
        # 生成对抗trace和TAM (使用Grad-CAM引导)
        adv_trace, adv_tam = trainer.generate_adversarial_trace(
            original_trace,
            label,
            gradcam_importance=gradcam_importance,  # 传入Grad-CAM重要性
            gradcam_model_type=gradcam_model_type  # 传入模型类型
        )
        
        # 保存
        adv_traces_list.append(adv_trace)
        adv_tams_list.append(adv_tam)
        original_labels.append(label)
        
        # 计算开销统计
        # 关键修复: 使用裁剪前的原始长度计算数据开销
        num_original = original_length_before_crop  # 裁剪前的实际长度
        num_total = len(adv_trace)  # 对抗trace的长度
        num_dummy = num_total - original_length_before_crop  # dummy包数 = 对抗-裁剪前原始长度
        
        overhead_stats['num_dummy_packets'].append(num_dummy)
        overhead_stats['num_original_packets'].append(num_original)
        overhead_stats['num_total_packets'].append(num_total)
        
        # 数据开销: O(D) = (|P'| - |P|) / |P| = num_dummy / num_original
        data_overhead = num_dummy / num_original if num_original > 0 else 0.0
        overhead_stats['data_overhead'].append(data_overhead)
        
        # 时间开销 (秒) - 使用未裁剪的原始trace计算
        original_end_time = original_trace_uncropped[-1, 0]  # 使用未裁剪的原始trace
        adv_end_time = adv_trace[-1, 0]
        time_overhead = adv_end_time - original_end_time
        overhead_stats['time_overhead'].append(time_overhead)
        
        # 对所有目标模型进行预测
        with torch.no_grad():
            for model_name, model in target_models.items():
                # 根据模型类型提取不同特征
                if model_name == 'rf':
                    # RF使用TAM
                    feat = torch.from_numpy(adv_tam).float().unsqueeze(0).to(trainer.device)
                elif model_name == 'df':
                    # DF使用方向序列
                    df_feat = feature_transform(adv_trace, feature_type='df', seq_length=8000)
                    feat = torch.from_numpy(df_feat).float().unsqueeze(0).to(trainer.device)
                elif model_name == 'tiktak':
                    # Tik-Tok使用时间×方向
                    tt_feat = feature_transform(adv_trace, feature_type='tiktok', seq_length=8000)
                    feat = torch.from_numpy(tt_feat).float().unsqueeze(0).to(trainer.device)
                elif model_name == 'varcnn':
                    # VarCNN使用DT2特征 [2, seq_length]
                    # 通道0: 方向, 通道1: 时间差分
                    varcnn_feat = feature_transform(adv_trace, feature_type='dt2', seq_length=8000)
                    feat = torch.from_numpy(varcnn_feat).float().unsqueeze(0).to(trainer.device)
                elif model_name == 'awf':
                    # AWF使用方向序列 [1, 3000]
                    awf_feat = feature_transform(adv_trace, feature_type='df', seq_length=3000)
                    feat = torch.from_numpy(awf_feat).float().unsqueeze(0).to(trainer.device)
                elif model_name == 'ares':
                    # ARES使用TAF特征 [8, 8000]
                    ares_feat = feature_transform(adv_trace, feature_type='taf', seq_length=8000)
                    feat = torch.from_numpy(ares_feat).float().unsqueeze(0).to(trainer.device)
                else:
                    continue
                
                logits = model(feat)
                pred = torch.argmax(logits, dim=1).item()
                adv_predictions[model_name].append(pred)
    
    # 保存所有对抗样本
    save_path = os.path.join(args.output_dir, 'adversarial_samples.npz')
    save_dict = {
        'labels': np.array(original_labels),
        'tams': np.stack([tam for tam in adv_tams_list])
    }
    # 保存每个模型的预测结果
    for model_name, preds in adv_predictions.items():
        save_dict[f'predictions_{model_name}'] = np.array(preds)
    
    np.savez(save_path, **save_dict)
    logger.info(f"对抗TAM已保存至: {save_path}")
    
    # 保存对抗traces (文本格式)
    traces_dir = os.path.join(args.output_dir, 'traces')
    os.makedirs(traces_dir, exist_ok=True)
    
    for i, (trace, label) in enumerate(zip(adv_traces_list, original_labels)):
        trace_path = os.path.join(traces_dir, f'{label}_{i}.cell')
        np.savetxt(trace_path, trace, fmt='%.6f\t%d', delimiter='\t')
    
    logger.info(f"对抗traces已保存至: {traces_dir}")
    
    # t-SNE 可视化 (基于TAM，仅选择5个类别)
    if getattr(args, 'plot_tsne', True):
        try:
            logger.info("开始基于TAM的t-SNE可视化...")
            orig_tams = np.stack([tam for tam in orig_tams_list])  # [N, 2, T]
            adv_tams = np.stack([tam for tam in adv_tams_list])    # [N, 2, T]
            labels_np = np.array(original_labels)

            # 选出出现频率最高的5个类别
            unique, counts = np.unique(labels_np, return_counts=True)
            sorted_idx = np.argsort(counts)[::-1]
            selected_classes = unique[sorted_idx[:5]]

            feats = []
            ys = []
            domains = []  # 0=original, 1=adversarial

            n_orig_per_class = 200  # 增加到200个原始样本
            n_adv_per_class = 200   # 增加到200个对抗样本

            for cls in selected_classes:
                cls_indices = np.where(labels_np == cls)[0]
                if len(cls_indices) == 0:
                    continue

                # 原始样本
                choose_orig = np.random.choice(
                    cls_indices,
                    size=min(n_orig_per_class, len(cls_indices)),
                    replace=False
                )
                for idx in choose_orig:
                    feats.append(orig_tams[idx].reshape(-1))
                    ys.append(cls)
                    domains.append(0)

                # 对抗样本
                choose_adv = np.random.choice(
                    cls_indices,
                    size=min(n_adv_per_class, len(cls_indices)),
                    replace=False
                )
                for idx in choose_adv:
                    feats.append(adv_tams[idx].reshape(-1))
                    ys.append(cls)
                    domains.append(1)

            if len(feats) > 0:
                X = np.stack(feats)
                ys_arr = np.array(ys)
                domains_arr = np.array(domains)

                tsne = TSNE(
                    n_components=2,
                    perplexity=30,
                    learning_rate=200,
                    n_iter=2000,
                    metric='euclidean',
                    random_state=42,
                    init='pca'
                )
                X_2d = tsne.fit_transform(X)

                plt.figure(figsize=(8, 6))
                colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
                cls_to_color = {int(cls): colors[i] for i, cls in enumerate(selected_classes)}

                for cls in selected_classes:
                    cls_int = int(cls)
                    mask_orig = (ys_arr == cls_int) & (domains_arr == 0)
                    mask_adv = (ys_arr == cls_int) & (domains_arr == 1)
                    c = cls_to_color[cls_int]

                    # 原始: 圆点
                    plt.scatter(
                        X_2d[mask_orig, 0],
                        X_2d[mask_orig, 1],
                        c=c,
                        marker='o',
                        edgecolors='k',
                        s=25,
                        alpha=0.7,
                        label=f'class {cls_int} original'
                    )

                    # 对抗: 叉号
                    plt.scatter(
                        X_2d[mask_adv, 0],
                        X_2d[mask_adv, 1],
                        c=c,
                        marker='x',
                        s=35,
                        alpha=0.9,
                        label=f'class {cls_int} adversarial'
                    )

                plt.xlabel('t-SNE dim 1')
                plt.ylabel('t-SNE dim 2')
                plt.title('t-SNE: Original vs Adversarial TAM Distribution (Top 5 Classes)')

                handles, labels_plt = plt.gca().get_legend_handles_labels()
                unique_legend = dict(zip(labels_plt, handles))
                plt.legend(unique_legend.values(), unique_legend.keys(), fontsize=8, loc='best')

                plt.tight_layout()
                tsne_path = os.path.join(args.output_dir, 'tsne_tam_5classes.png')
                plt.savefig(tsne_path, dpi=200)
                plt.close()
                logger.info(f"t-SNE 可视化已保存至: {tsne_path}")

                # 为每个类别分别画一张放大的 t-SNE 子图
                for cls in selected_classes:
                    cls_int = int(cls)
                    mask_orig = (ys_arr == cls_int) & (domains_arr == 0)
                    mask_adv = (ys_arr == cls_int) & (domains_arr == 1)

                    if not (mask_orig.any() or mask_adv.any()):
                        continue

                    plt.figure(figsize=(6, 5))
                    c = cls_to_color[cls_int]

                    # 原始: 圆点
                    plt.scatter(
                        X_2d[mask_orig, 0],
                        X_2d[mask_orig, 1],
                        c=c,
                        marker='o',
                        edgecolors='k',
                        s=30,
                        alpha=0.8,
                        label='original'
                    )

                    # 对抗: 叉号
                    plt.scatter(
                        X_2d[mask_adv, 0],
                        X_2d[mask_adv, 1],
                        c=c,
                        marker='x',
                        s=40,
                        alpha=0.9,
                        label='adversarial'
                    )

                    plt.xlabel('t-SNE dim 1')
                    plt.ylabel('t-SNE dim 2')
                    plt.title(f't-SNE: Class {cls_int} Original vs Adversarial (Zoomed)')
                    plt.legend(fontsize=8, loc='best')
                    plt.tight_layout()

                    tsne_cls_path = os.path.join(args.output_dir, f'tsne_tam_class{cls_int}.png')
                    plt.savefig(tsne_cls_path, dpi=200)
                    plt.close()
                    logger.info(f"t-SNE 子图(类别 {cls_int})已保存至: {tsne_cls_path}")
            else:
                logger.warning("t-SNE 可视化: 无可用样本，跳过")
        except Exception as e:
            logger.warning(f"t-SNE 可视化失败: {e}")
    
    # 基于TAM特征的原始 vs 对抗可视化（类似论文图3-5）
    if getattr(args, 'plot_tsne', True):
        try:
            logger.info("开始基于TAM的原始 vs 对抗时间序列可视化...")
            labels_np = np.array(original_labels)

            unique, counts = np.unique(labels_np, return_counts=True)
            if len(unique) > 0:
                sorted_idx = np.argsort(counts)[::-1]
                selected_classes_trace = unique[sorted_idx[:4]]  # 选4个类别

                n_per_class = 1  # 每类1个样本
                n_cols = len(selected_classes_trace)

                # ====== 第一张图：TAM累积曲线 + 时间间隔对比（上下两行）======
                fig1, axes1 = plt.subplots(2, n_cols, figsize=(4 * n_cols, 6), squeeze=False)

                for col, cls in enumerate(selected_classes_trace):
                    cls_indices = np.where(labels_np == cls)[0]
                    if len(cls_indices) == 0:
                        continue

                    # 随机选择一个样本
                    idx = np.random.choice(cls_indices)

                    # 获取TAM特征 [2, seq_length]
                    orig_tam = orig_tams_list[idx]  # [2, T]
                    adv_tam = adv_tams_list[idx]    # [2, T]

                    # 上图：累积大小序列（TAM通道0：累积入包数，通道1：累积出包数）
                    ax_top = axes1[0][col]
                    x_range = np.arange(orig_tam.shape[1])
                    
                    # 绘制累积曲线
                    ax_top.plot(x_range, orig_tam[0, :], 'k-', linewidth=1.5, alpha=0.8, label='Original (In)')
                    ax_top.plot(x_range, orig_tam[1, :], 'k--', linewidth=1.5, alpha=0.8, label='Original (Out)')
                    ax_top.plot(x_range, adv_tam[0, :], 'r-', linewidth=1.2, alpha=0.7, label='Adversarial (In)')
                    ax_top.plot(x_range, adv_tam[1, :], 'b--', linewidth=1.2, alpha=0.7, label='Adversarial (Out)')
                    
                    ax_top.set_ylabel('Cumulative packets')
                    ax_top.set_title(f'Class {int(cls)}')
                    ax_top.grid(True, alpha=0.3)
                    if col == 0:
                        ax_top.legend(fontsize=7, loc='upper left')

                    # 下图：时间间隔差分（显示扰动模式）
                    ax_bottom = axes1[1][col]
                    
                    # 计算差分（间隔）
                    orig_diff_in = np.diff(orig_tam[0, :], prepend=0)
                    orig_diff_out = np.diff(orig_tam[1, :], prepend=0)
                    adv_diff_in = np.diff(adv_tam[0, :], prepend=0)
                    adv_diff_out = np.diff(adv_tam[1, :], prepend=0)
                    
                    # 只显示非零位置（有效包位置）
                    ax_bottom.plot(x_range, orig_diff_in, 'k-', linewidth=0.8, alpha=0.6, label='Original (In)')
                    ax_bottom.plot(x_range, orig_diff_out, 'k--', linewidth=0.8, alpha=0.6, label='Original (Out)')
                    ax_bottom.plot(x_range, adv_diff_in, 'r-', linewidth=0.8, alpha=0.7, label='Adversarial (In)')
                    ax_bottom.plot(x_range, adv_diff_out, 'b--', linewidth=0.8, alpha=0.7, label='Adversarial (Out)')
                    
                    ax_bottom.set_ylabel('Packet interval')
                    ax_bottom.set_xlabel('Time index')
                    ax_bottom.grid(True, alpha=0.3)
                    if col == 0:
                        ax_bottom.legend(fontsize=7, loc='upper left')

                fig1.suptitle('TAM Feature Comparison: Original vs Adversarial (Top row: Cumulative, Bottom row: Interval)', fontsize=12)
                plt.tight_layout(rect=[0, 0, 1, 0.96])

                trace_fig_path1 = os.path.join(args.output_dir, 'tam_comparison_4classes.png')
                plt.savefig(trace_fig_path1, dpi=200)
                plt.close(fig1)
                logger.info(f"TAM 对比图已保存至: {trace_fig_path1}")

                # ====== 第二张图：只显示插入的 dummy 包（基于原始trace）======
                selected_classes_dummy = unique[sorted_idx[:5]]  # 选5个类别
                n_per_class_dummy = 3
                n_rows_dummy = len(selected_classes_dummy)
                n_cols_dummy = n_per_class_dummy
                
                fig2, axes2 = plt.subplots(n_rows_dummy, n_cols_dummy, figsize=(4 * n_cols_dummy, 3 * n_rows_dummy), squeeze=False, sharey=True)

                for row, cls in enumerate(selected_classes_dummy):
                    cls_indices = np.where(labels_np == cls)[0]
                    if len(cls_indices) == 0:
                        continue

                    choose_idx = np.random.choice(
                        cls_indices,
                        size=min(n_per_class_dummy, len(cls_indices)),
                        replace=False
                    )

                    for col, idx in enumerate(choose_idx):
                        ax = axes2[row][col]

                        orig_trace = orig_traces_uncropped_list[idx]
                        adv_trace = adv_traces_list[idx]

                        # 计算插入的 dummy 包：对抗 trace 中不在原始 trace 中的点
                        # 简化策略：先画原始（灰色，低透明度），再画所有对抗包（红色高亮）
                        t_orig = orig_trace[:, 0]
                        d_orig = np.sign(orig_trace[:, 1])
                        t_adv = adv_trace[:, 0]
                        d_adv = np.sign(adv_trace[:, 1])

                        # 背景：原始 trace（灰色，作为参考）
                        ax.scatter(
                            t_orig,
                            d_orig,
                            s=3,
                            alpha=0.3,
                            color='lightgray',
                            marker='.',
                            label='original (background)' if row == 0 and col == 0 else ''
                        )

                        # 找出 dummy 包：对比对抗和原始的差异
                        # 简化方法：如果对抗 trace 长度 > 原始 trace 长度，额外的包就是 dummy
                        num_dummy = len(adv_trace) - len(orig_trace)
                        if num_dummy > 0:
                            # 高亮显示所有对抗包（包括原始+dummy）
                            ax.scatter(
                                t_adv,
                                d_adv,
                                s=15,
                                alpha=0.8,
                                color='red',
                                marker='o',
                                edgecolors='darkred',
                                linewidths=0.5,
                                label=f'dummy packets ({num_dummy})' if row == 0 and col == 0 else ''
                            )
                        else:
                            # 如果没有 dummy 包，显示提示
                            ax.text(0.5, 0, 'No dummy packets inserted', 
                                   ha='center', va='center', transform=ax.transAxes,
                                   fontsize=10, color='red', style='italic')

                        if col == 0:
                            ax.set_ylabel(f'class {int(cls)}')
                        ax.set_xlabel('time (s)')

                handles2, labels_plt2 = axes2[0][0].get_legend_handles_labels()
                if handles2:
                    fig2.legend(handles2, labels_plt2, loc='upper right', fontsize=8)

                fig2.suptitle('Dummy Packets Visualization (5 Classes x 3 Samples)', fontsize=12)
                plt.tight_layout(rect=[0, 0, 0.9, 0.95])

                trace_fig_path2 = os.path.join(args.output_dir, 'trace_dummy_only_5classes.png')
                plt.savefig(trace_fig_path2, dpi=200)
                plt.close(fig2)
                logger.info(f"dummy 包可视化图已保存至: {trace_fig_path2}")

                # ====== 第三张图：基于Burst特征的对比（类似论文图3-5）======
                selected_classes_burst = unique[sorted_idx[:4]]  # 选4个类别
                
                fig3, axes3 = plt.subplots(2, len(selected_classes_burst), figsize=(4 * len(selected_classes_burst), 6), squeeze=False)

                for col, cls in enumerate(selected_classes_burst):
                    cls_indices = np.where(labels_np == cls)[0]
                    if len(cls_indices) == 0:
                        continue

                    # 随机选择一个样本
                    idx = np.random.choice(cls_indices)

                    orig_trace = orig_traces_uncropped_list[idx]
                    adv_trace = adv_traces_list[idx]

                    # 提取burst特征
                    orig_bursts = extract_bursts(orig_trace)
                    adv_bursts = extract_bursts(adv_trace)

                    # 上图：包大小序列（Size）
                    ax_top = axes3[0][col]
                    
                    # 原始流量：黑色
                    orig_sizes = orig_trace[:, 1]  # 包大小（带符号）
                    x_orig = np.arange(len(orig_sizes))
                    ax_top.plot(x_orig, orig_sizes, 'k-', linewidth=0.8, alpha=0.8, label='Original')
                    
                    # 对抗流量：红色
                    adv_sizes = adv_trace[:, 1]
                    x_adv = np.arange(len(adv_sizes))
                    ax_top.plot(x_adv, adv_sizes, 'r-', linewidth=0.6, alpha=0.7, label='Adversarial')
                    
                    ax_top.set_ylabel('Packet size')
                    ax_top.set_title(f'Class {int(cls)}')
                    ax_top.grid(True, alpha=0.3)
                    ax_top.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
                    if col == 0:
                        ax_top.legend(fontsize=7, loc='upper left')

                    # 下图：Burst间隔时间序列
                    ax_bottom = axes3[1][col]
                    
                    # 计算burst间隔（相邻两个burst之间的时间差）
                    orig_intervals = []
                    for i in range(1, len(orig_bursts)):
                        interval = orig_bursts[i]['start_time'] - orig_bursts[i-1]['end_time']
                        orig_intervals.append(interval)
                    
                    adv_intervals = []
                    for i in range(1, len(adv_bursts)):
                        interval = adv_bursts[i]['start_time'] - adv_bursts[i-1]['end_time']
                        adv_intervals.append(interval)
                    
                    # 绘制burst间隔
                    if len(orig_intervals) > 0:
                        ax_bottom.plot(np.arange(len(orig_intervals)), orig_intervals, 'k-', 
                                      linewidth=0.8, alpha=0.8, label='Original')
                    
                    if len(adv_intervals) > 0:
                        ax_bottom.plot(np.arange(len(adv_intervals)), adv_intervals, 'r-', 
                                      linewidth=0.6, alpha=0.7, label='Adversarial')
                    
                    ax_bottom.set_ylabel('Burst interval (s)')
                    ax_bottom.set_xlabel('Burst index')
                    ax_bottom.grid(True, alpha=0.3)
                    if col == 0:
                        ax_bottom.legend(fontsize=7, loc='upper left')

                fig3.suptitle('Burst Feature Comparison: Original vs Adversarial (Top: Packet Size, Bottom: Burst Interval)', fontsize=12)
                plt.tight_layout(rect=[0, 0, 1, 0.96])

                burst_fig_path = os.path.join(args.output_dir, 'burst_comparison_4classes.png')
                plt.savefig(burst_fig_path, dpi=200)
                plt.close(fig3)
                logger.info(f"Burst 对比图已保存至: {burst_fig_path}")
        except Exception as e:
            logger.warning(f"trace 时间序列可视化失败: {e}")
    
    # 评估所有模型的ASR
    original_labels = np.array(original_labels)
    
    logger.info(f"\n========== 评估结果 ==========")
    logger.info(f"总样本数: {len(original_labels)}")
    
    # 1. 数据包开销统计
    logger.info(f"\n--- 数据包开销统计 ---")
    
    # 使用总体统计 (更准确)
    total_dummy = np.sum(overhead_stats['num_dummy_packets'])
    total_original = np.sum(overhead_stats['num_original_packets'])
    total_total = np.sum(overhead_stats['num_total_packets'])
    
    # 平均值 (每个样本)
    avg_dummy = np.mean(overhead_stats['num_dummy_packets'])
    avg_original = np.mean(overhead_stats['num_original_packets'])
    avg_total = np.mean(overhead_stats['num_total_packets'])
    
    # 数据开销: 使用总体统计
    overall_data_overhead = total_dummy / total_original if total_original > 0 else 0.0
    avg_time = np.mean(overhead_stats['time_overhead'])
    
    logger.info(f"平均原始包数 |P|: {avg_original:.1f}")
    logger.info(f"平均插入dummy包数 (|P'| - |P|): {avg_dummy:.1f}")
    logger.info(f"平均总包数 |P'|: {avg_total:.1f}")
    logger.info(f"总体数据开销 O(D) = Σ(|P'|-|P|)/Σ|P|: {overall_data_overhead:.4f} ({overall_data_overhead*100:.2f}%)")
    logger.info(f"平均时间开销: {avg_time:.3f} 秒")
    
    overhead_summary = {
        'avg_original_packets': float(avg_original),           # 平均|P|
        'avg_dummy_packets': float(avg_dummy),                 # 平均dummy包数
        'avg_total_packets': float(avg_total),                 # 平均|P'|
        'overall_data_overhead': float(overall_data_overhead),  # 总体O(D) = Σdummy/Σoriginal
        'overall_data_overhead_percentage': float(overall_data_overhead * 100),
        'avg_time_overhead_seconds': float(avg_time),
        'median_dummy_packets': float(np.median(overhead_stats['num_dummy_packets'])),
        'max_dummy_packets': int(np.max(overhead_stats['num_dummy_packets'])),
        'min_dummy_packets': int(np.min(overhead_stats['num_dummy_packets']))
    }
    
    # 2. ASR评估
    all_results = {}
    
    for model_name, preds_list in adv_predictions.items():
        preds = np.array(preds_list)
        
        # 计算总体ASR
        asr = (preds != original_labels).mean()
        acc = (preds == original_labels).mean()
        
        logger.info(f"\n--- {model_name.upper()} 模型 ---")
        logger.info(f"攻击成功率 (ASR): {asr:.2%}")
        logger.info(f"剩余准确率: {acc:.2%}")
        
        # 按类别统计ASR
        per_class_asr = {}
        for cls in np.unique(original_labels):
            mask = original_labels == cls
            if mask.sum() > 0:
                class_asr = (preds[mask] != original_labels[mask]).mean()
                per_class_asr[int(cls)] = class_asr
        
        # 显示前10类
        logger.info(f"各类别ASR (前10):")
        for cls, cls_asr in sorted(per_class_asr.items())[:10]:
            logger.info(f"  Class {cls}: {cls_asr:.2%}")
        
        # 画每个网站类别的平均dummy包数量与ASR的关系图
        if getattr(args, 'plot_tsne', True):
            try:
                labels_np = original_labels
                dummy_np = np.array(overhead_stats['num_dummy_packets'])
                classes_sorted = sorted(per_class_asr.keys())

                mean_dummy = []
                asr_vals = []
                for c in classes_sorted:
                    mask_c = labels_np == c
                    if mask_c.sum() > 0:
                        mean_dummy.append(float(dummy_np[mask_c].mean()))
                    else:
                        mean_dummy.append(0.0)
                    asr_vals.append(per_class_asr[c] * 100.0)

                fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

                # Upper plot: Average dummy packets
                axes[0].plot(classes_sorted, mean_dummy, '-o', color='red', markersize=3)
                axes[0].set_ylabel('Avg dummy packets')
                axes[0].set_title(f'{model_name.upper()} Dummy Packets vs ASR per Class')
                axes[0].grid(True, alpha=0.3)

                # Lower plot: ASR
                axes[1].plot(classes_sorted, asr_vals, '-o', color='blue', markersize=3)
                axes[1].set_ylabel('ASR (%)')
                axes[1].set_xlabel('Website class')
                axes[1].grid(True, alpha=0.3)

                plt.tight_layout()
                rel_path = os.path.join(args.output_dir, f'per_class_dummy_vs_asr_{model_name}.png')
                plt.savefig(rel_path, dpi=200)
                plt.close(fig)
                logger.info(f"每类 dummy 包与 ASR 关系图已保存至: {rel_path}")
            except Exception as e:
                logger.warning(f"绘制每类 dummy vs ASR 图失败 ({model_name}): {e}")
        
        all_results[model_name] = {
            'asr': float(asr),
            'accuracy': float(acc),
            'per_class_asr': {k: float(v) for k, v in per_class_asr.items()}
        }
    
    # 保存评估结果
    import json
    eval_path = os.path.join(args.output_dir, 'evaluation.json')
    with open(eval_path, 'w') as f:
        json.dump({
            'total_samples': len(original_labels),
            'overhead': overhead_summary,
            'models': all_results
        }, f, indent=2)
    
    logger.info(f"\n评估结果已保存至: {eval_path}")
    logger.info("完成!")


if __name__ == '__main__':
    main()
