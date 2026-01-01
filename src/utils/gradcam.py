"""
Grad-CAM for 1D/2D models (RF, DF, Tik-Tok)
用于识别对分类器决策最重要的时间位置
"""
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM implementation for website fingerprinting models
    """
    def __init__(self, model, target_layer):
        """
        Args:
            model: 分类模型 (RFNet, DFNet, etc.)
            target_layer: 目标层 (通常是最后一个卷积层)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 注册hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """注册hooks (使用full_backward_hook避免warning)"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        # 使用register_full_backward_hook代替register_backward_hook
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        生成Grad-CAM热力图
        Args:
            input_tensor: [1, C, L] 或 [1, 1, C, L] 输入
            target_class: 目标类别 (None表示预测类别)
        Returns:
            cam: [L] 1D热力图
            pred_class: 预测类别
        """
        self.model.eval()
        
        # 前向传播
        output = self.model(input_tensor)
        pred_class = output.argmax(dim=1).item()
        
        if target_class is None:
            target_class = pred_class
        
        # 反向传播
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # 计算CAM
        # gradients: [1, C, L]
        # activations: [1, C, L]
        gradients = self.gradients
        activations = self.activations
        
        # 使用RMS (均方根) 代替平均值，保留梯度的强度信息
        # RMS能更好地捕捉梯度的显著性，避免正负抵消
        weights = torch.sqrt((gradients ** 2).mean(dim=(0, 2), keepdim=True) + 1e-8)  # [1, C, 1]
        
        # Weighted combination
        cam = (weights * activations).sum(dim=1)  # [1, L]
        cam = F.relu(cam)  # ReLU
        cam = cam.squeeze(0)  # [L]
        
        # 归一化到[0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy(), pred_class
    
    def get_top_k_positions(self, cam, k=500, min_distance=5):
        """
        从CAM中提取Top-K重要位置
        Args:
            cam: [L] 1D热力图
            k: 返回的位置数量
            min_distance: 位置之间的最小距离
        Returns:
            positions: [K] 重要位置的索引
            importances: [K] 对应的重要性分数
        """
        # 使用非极大值抑制避免位置聚集
        positions = []
        importances = []
        cam_copy = cam.copy()
        
        for _ in range(k):
            if cam_copy.max() == 0:
                break
            
            # 找到最大值位置
            max_idx = cam_copy.argmax()
            max_val = cam_copy[max_idx]
            
            positions.append(max_idx)
            importances.append(max_val)
            
            # 抑制周围区域
            start = max(0, max_idx - min_distance)
            end = min(len(cam_copy), max_idx + min_distance + 1)
            cam_copy[start:end] = 0
        
        return np.array(positions), np.array(importances)


class TAMGradCAM(GradCAM):
    """
    针对TAM格式 (RFNet) 的Grad-CAM
    输入: [1, 2, seq_length]
    """
    def generate_cam(self, input_tensor, target_class=None):
        """
        生成TAM的Grad-CAM
        Args:
            input_tensor: [1, 2, L] TAM矩阵
            target_class: 目标类别
        Returns:
            cam: [L] 时间维度的热力图
            pred_class: 预测类别
        """
        cam, pred_class = super().generate_cam(input_tensor, target_class)
        return cam, pred_class


class SequenceGradCAM(GradCAM):
    """
    针对1D序列 (DFNet, Tik-Tok) 的Grad-CAM
    输入: [1, 1, seq_length]
    """
    def generate_cam(self, input_tensor, target_class=None):
        """
        生成序列的Grad-CAM
        Args:
            input_tensor: [1, 1, L] 1D序列
            target_class: 目标类别
        Returns:
            cam: [L] 时间维度的热力图
            pred_class: 预测类别
        """
        cam, pred_class = super().generate_cam(input_tensor, target_class)
        return cam, pred_class


def get_gradcam_for_model(model, model_type='rf'):
    """
    为不同模型创建对应的Grad-CAM
    Args:
        model: 分类模型
        model_type: 'rf', 'df', 'tiktak'
    Returns:
        gradcam: GradCAM实例
    """
    if model_type == 'rf':
        # RFNet: 最后一个1D卷积层
        target_layer = model.features[-3]  # 倒数第3层 (最后的Conv1d)
        return TAMGradCAM(model, target_layer)
    
    elif model_type in ['df', 'tiktak']:
        # DFNet/Tik-Tok: layer4的最后一个卷积
        target_layer = model.layer4[0]  # Conv1d in layer4
        return SequenceGradCAM(model, target_layer)
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
