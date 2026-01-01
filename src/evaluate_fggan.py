"""
评估FGGAN防御在开放世界下的效果
实时生成对抗样本并针对不同模型评估
"""
import argparse
import os
import json
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from fggan_improved import ImprovedFGGAN
from attacks.modules.rf import RFNet
from attacks.modules.df import DFNet
from attacks.modules.varcnn import VarCNNNet
from attacks.modules.awf import AWFNet
from attacks.modules.ares import ARESNet
from utils.general import get_flist_label, parse_trace, feature_transform
from utils.logger import init_logger


def evaluate_open_world(predictions, true_labels, mon_classes):
    """
    开放世界评估指标
    Args:
        predictions: 模型预测 [N]
        true_labels: 真实标签 [N] (0~mon_classes-1为监控类, mon_classes为非监控类)
        mon_classes: 监控类别数
    Returns:
        dict: 评估指标
    """
    # 区分监控类和非监控类
    is_monitored_true = true_labels < mon_classes
    is_monitored_pred = predictions < mon_classes
    
    # 1. 整体准确率
    overall_acc = accuracy_score(true_labels, predictions)
    
    # 2. 监控类准确率
    mon_indices = np.where(is_monitored_true)[0]
    mon_acc = accuracy_score(
        true_labels[mon_indices], 
        predictions[mon_indices]
    ) if len(mon_indices) > 0 else 0.0
    
    # 3. 非监控类准确率
    unmon_indices = np.where(~is_monitored_true)[0]
    unmon_acc = accuracy_score(
        true_labels[unmon_indices],
        predictions[unmon_indices]
    ) if len(unmon_indices) > 0 else 0.0
    
    # 4. TPR (True Positive Rate): 真实监控类被正确分类为监控类
    tp = np.sum(is_monitored_true & is_monitored_pred)
    fn = np.sum(is_monitored_true & ~is_monitored_pred)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # 5. FPR (False Positive Rate): 非监控类被误判为监控类
    fp = np.sum(~is_monitored_true & is_monitored_pred)
    tn = np.sum(~is_monitored_true & ~is_monitored_pred)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    # 6. Precision/Recall/F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tpr
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # 7. ASR (Attack Success Rate): 分类错误率
    asr = 1.0 - overall_acc
    
    return {
        'overall_accuracy': overall_acc,
        'monitored_accuracy': mon_acc,
        'unmonitored_accuracy': unmon_acc,
        'tpr': tpr,
        'fpr': fpr,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'asr': asr,
        'num_monitored': int(np.sum(is_monitored_true)),
        'num_unmonitored': int(np.sum(~is_monitored_true))
    }


def load_model(model_name, num_classes, checkpoint_path, device):
    """加载攻击模型"""
    if model_name == 'rf':
        model = RFNet(num_classes=num_classes)
    elif model_name == 'df':
        model = DFNet(num_classes=num_classes)
    elif model_name == 'varcnn':
        model = VarCNNNet(num_classes=num_classes)
    elif model_name == 'awf':
        model = AWFNet(num_classes=num_classes)
    elif model_name == 'ares':
        model = ARESNet(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description='评估FGGAN防御在开放世界下的效果')
    
    # 数据路径
    parser.add_argument('--data-path', type=str, required=True,
                        help='原始数据目录')
    
    # 数据配置
    parser.add_argument('--mon-classes', type=int, default=100)
    parser.add_argument('--mon-inst', type=int, default=100)
    parser.add_argument('--unmon-inst', type=int, default=10000)
    parser.add_argument('--open-world', action='store_true',
                        help='是否为开放世界(计算TPR/FPR等指标)')
    parser.add_argument('--suffix', type=str, default='.cell')
    parser.add_argument('--seq-length', type=int, default=5000)
    
    # FGGAN模型
    parser.add_argument('--fggan-model', type=str, required=True,
                        help='训练好的FGGAN模型路径')
    parser.add_argument('--generator-g1', type=str, required=True,
                        help='预训练traffic_generator权重路径')
    parser.add_argument('--proxy-model', type=str, required=True,
                        help='预训练代理模型权重路径')
    
    # 评估目标模型
    parser.add_argument('--models', type=str, nargs='+',
                        default=['rf', 'df', 'varcnn'],
                        help='评估的攻击模型')
    parser.add_argument('--model-checkpoints', type=str, nargs='+',
                        required=True,
                        help='模型checkpoint路径(顺序对应--models)')
    
    # 输出
    parser.add_argument('--output-dir', type=str, default='./evaluation_results')
    
    # 设备
    parser.add_argument('--use-gpu', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    
    args = parser.parse_args()
    
    logger = init_logger('EvaluateFGGAN')
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device(f'cuda:{args.gpu}' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载原始数据
    logger.info(f"加载原始数据: {args.data_path}")
    flist, labels = get_flist_label(
        args.data_path,
        args.mon_classes,
        args.mon_inst,
        args.unmon_inst if args.open_world else 0,
        args.suffix
    )
    
    logger.info(f"数据样本数: {len(flist)}")
    
    num_classes = args.mon_classes + 1 if args.open_world else args.mon_classes
    
    # 加载FGGAN模型
    logger.info("加载FGGAN模型...")
    fggan = ImprovedFGGAN(
        args,
        generator_g1_path=args.generator_g1,
        proxy_model_path=args.proxy_model,
    )
    fggan.load_model(args.fggan_model)
    fggan.G2.eval()
    fggan.D.eval()
    
    # 加载所有攻击模型
    models = {}
    for model_name, ckpt_path in zip(args.models, args.model_checkpoints):
        logger.info(f"加载模型: {model_name} from {ckpt_path}")
        models[model_name] = load_model(model_name, num_classes, ckpt_path, device)
    
    # 评估每个模型
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"\n========== 评估模型: {model_name.upper()} ==========")
        
        # 对所有样本实时生成对抗样本并预测
        predictions = []
        for i, (fpath, label) in enumerate(tqdm(zip(flist, labels), total=len(flist), desc=f"生成+推理 {model_name}")):
            # 读取原始trace
            original_trace = parse_trace(fpath)
            
            # 实时生成对抗样本
            adv_trace, _ = fggan.generate_adversarial_trace(original_trace, label)
            
            # 根据模型类型提取对应特征
            if model_name == 'rf':
                feat = feature_transform(adv_trace, 'tam', 5000)
            elif model_name == 'df':
                feat = feature_transform(adv_trace, 'df', 5000)
            elif model_name == 'varcnn':
                feat = feature_transform(adv_trace, 'dt2', 5000)
            elif model_name == 'awf':
                feat = feature_transform(adv_trace, 'df', 3000)
            elif model_name == 'ares':
                feat = feature_transform(adv_trace, 'taf', 8000)
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            
            feat = torch.from_numpy(feat).float().unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = model(feat)
                pred = torch.argmax(logits, dim=1).item()
                predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # 计算指标
        if args.open_world:
            # 开放世界评估
            metrics = evaluate_open_world(predictions, labels, args.mon_classes)
            
            logger.info(f"\n--- {model_name.upper()} 开放世界指标 ---")
            logger.info(f"总体准确率: {metrics['overall_accuracy']:.4f}")
            logger.info(f"监控类准确率: {metrics['monitored_accuracy']:.4f}")
            logger.info(f"非监控类准确率: {metrics['unmonitored_accuracy']:.4f}")
            logger.info(f"TPR (真阳性率): {metrics['tpr']:.4f}")
            logger.info(f"FPR (假阳性率): {metrics['fpr']:.4f}")
            logger.info(f"Precision: {metrics['precision']:.4f}")
            logger.info(f"Recall: {metrics['recall']:.4f}")
            logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
            logger.info(f"ASR (攻击成功率): {metrics['asr']:.4f}")
            logger.info(f"监控样本数: {metrics['num_monitored']}")
            logger.info(f"非监控样本数: {metrics['num_unmonitored']}")
        else:
            # 封闭世界评估
            accuracy = accuracy_score(labels, predictions)
            asr = 1.0 - accuracy
            
            metrics = {
                'overall_accuracy': accuracy,
                'asr': asr
            }
            
            logger.info(f"\n--- {model_name.upper()} 封闭世界指标 ---")
            logger.info(f"准确率: {accuracy:.4f}")
            logger.info(f"ASR (攻击成功率): {asr:.4f}")
        
        results[model_name] = metrics
    
    # 汇总结果
    summary = {
        'world_setting': 'open_world' if args.open_world else 'closed_world',
        'num_samples': len(labels),
        'num_classes': num_classes,
        'mon_classes': args.mon_classes,
        'model_results': results
    }
    
    # 保存JSON结果
    results_path = os.path.join(args.output_dir, 'fggan_evaluation.json')
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n========== 评估完成 ===========")
    logger.info(f"评估结果已保存至: {results_path}")
    
    # 打印对比总结
    if args.open_world:
        logger.info(f"\n--- 开放世界 vs 封闭世界对比 ---")
        logger.info("开放世界新增指标:")
        logger.info("  - TPR: 监控类被正确识别的比率(越低越好)")
        logger.info("  - FPR: 非监控类被误判为监控类的比率(越低越好)")
        logger.info("  - F1-Score: 综合评估指标")
        logger.info("\n理想的FGGAN防御应该:")
        logger.info("  - 高ASR: 成功欺骗攻击模型")
        logger.info("  - 低TPR: 监控类难以被识别")
        logger.info("  - 低FPR: 不影响非监控类的识别")
    
    logger.info("\n注意: 本评估实时生成对抗样本,无需预先生成")


if __name__ == '__main__':
    main()
