import numpy as np
import cv2
import torch
import torch.nn.functional as F

def bilateral_depth_filter(depth_map, rgb_image, sigma_space=10, sigma_color=0.1, kernel_size=15):
    """
    使用双边滤波对深度图进行边缘保留滤波
    
    参数:
        depth_map: 输入的深度图
        rgb_image: 参考RGB图像，用于提取边缘信息
        sigma_space: 空间距离权重参数
        sigma_color: 颜色差异权重参数
        kernel_size: 滤波器核大小
        
    返回:
        refined_depth: 边缘优化后的深度图
    """
    # 确保深度图为单通道浮点数类型
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.detach().cpu().numpy()
    
    # 归一化深度图以便于处理
    depth_min = np.min(depth_map)
    depth_max = np.max(depth_map)
    depth_normalized = (depth_map - depth_min) / (depth_max - depth_min)
    
    # 将深度图转换为CV2可处理的格式
    depth_normalized = (depth_normalized * 255).astype(np.uint8)
    
    # 提取RGB图像边缘信息
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY) if len(rgb_image.shape) > 2 else rgb_image
    edges = cv2.Canny(gray_image, 50, 150)
    
    # 应用双边滤波，保留边缘
    refined_depth = cv2.bilateralFilter(depth_normalized, kernel_size, sigma_color * 255, sigma_space)
    
    # 在边缘区域特别处理，可以提高边缘处的准确性
    edge_mask = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1) / 255.0
    
    # 将边缘信息融合到优化后的深度图中
    refined_depth = refined_depth * (1 - edge_mask) + depth_normalized * edge_mask
    
    # 恢复原始深度值范围
    refined_depth = refined_depth / 255.0 * (depth_max - depth_min) + depth_min
    
    return refined_depth

def guided_depth_filter(depth_map, rgb_image, radius=8, eps=0.0001):
    """
    使用引导滤波对深度图进行边缘保留优化
    
    参数:
        depth_map: 输入的深度图
        rgb_image: 引导图像
        radius: 滤波半径
        eps: 正则化参数
        
    返回:
        refined_depth: 引导滤波优化后的深度图
    """
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.detach().cpu().numpy()
    
    # 转换RGB图像为灰度图作为引导图
    if len(rgb_image.shape) > 2:
        guide = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    else:
        guide = rgb_image.copy()
    
    # 深度图归一化
    depth_min = np.min(depth_map)
    depth_max = np.max(depth_map)
    depth_normalized = (depth_map - depth_min) / (depth_max - depth_min)
    
    # 应用引导滤波
    depth_normalized = depth_normalized.astype(np.float32)
    guide = guide.astype(np.float32) / 255.0
    refined_depth = cv2.ximgproc.guidedFilter(guide, depth_normalized, radius, eps)
    
    # 恢复原始深度值范围
    refined_depth = refined_depth * (depth_max - depth_min) + depth_min
    
    return refined_depth

def edge_aware_depth_refinement(depth_pred, rgb_image):
    """
    结合多种边缘保留技术对深度图进行综合优化
    
    参数:
        depth_pred: 初始预测的深度图
        rgb_image: 对应的RGB图像
        
    返回:
        refined_depth: 优化后的深度图
    """
    # 1. 应用双边滤波进行初步优化
    bilateral_refined = bilateral_depth_filter(depth_pred, rgb_image)
    
    # 2. 应用引导滤波进一步优化
    guided_refined = guided_depth_filter(bilateral_refined, rgb_image)
    
    # 3. 边缘检测，用于识别需要特别处理的区域
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY) if len(rgb_image.shape) > 2 else rgb_image
    edges = cv2.Canny(gray, 50, 150)
    edge_mask = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1) / 255.0
    
    # 4. 在边缘区域应用自适应权重融合
    weight_map = 1.0 - 0.7 * edge_mask  # 边缘区域权重更低，更信任原始深度值
    
    refined_depth = guided_refined * weight_map + bilateral_refined * (1.0 - weight_map)
    
    return refined_depth