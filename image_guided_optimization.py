import numpy as np
import cv2
import torch
import torch.nn.functional as F
from skimage.segmentation import felzenszwalb, slic

class ImageGuidedDepthOptimizer:
    """
    基于图像引导的深度图优化类，整合多种优化策略
    """
    
    def __init__(self, use_superpixel=True, use_structure_tensor=True, use_cross_bilateral=True):
        """
        初始化优化器
        
        参数:
            use_superpixel: 是否使用超像素分割辅助优化
            use_structure_tensor: 是否使用结构张量分析
            use_cross_bilateral: 是否使用交叉双边滤波
        """
        self.use_superpixel = use_superpixel
        self.use_structure_tensor = use_structure_tensor
        self.use_cross_bilateral = use_cross_bilateral
    
    def optimize(self, depth_map, rgb_image):
        """
        执行深度图优化
        
        参数:
            depth_map: 输入深度图
            rgb_image: 参考RGB图像
            
        返回:
            optimized_depth: 优化后的深度图
        """
        if isinstance(depth_map, torch.Tensor):
            depth_map = depth_map.detach().cpu().numpy()
        
        # 1. 应用引导滤波的基础优化
        optimized_depth = self._apply_guided_filter(depth_map, rgb_image)
        
        # 2. 超像素分割辅助优化
        if self.use_superpixel:
            optimized_depth = self._enhance_with_superpixel(optimized_depth, rgb_image)
        
        # 3. 基于结构张量的优化
        if self.use_structure_tensor:
            optimized_depth = self._enhance_with_structure_tensor(optimized_depth, rgb_image)
        
        # 4. 交叉双边滤波优化
        if self.use_cross_bilateral:
            optimized_depth = self._apply_cross_bilateral(optimized_depth, rgb_image)
        
        return optimized_depth
    
    def _apply_guided_filter(self, depth_map, rgb_image, radius=10, eps=1e-4):
        """应用引导滤波进行基础优化"""
        # 转换为灰度图像作为引导图
        if len(rgb_image.shape) > 2:
            guide = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        else:
            guide = rgb_image.copy()
        
        # 归一化深度图
        depth_min = np.min(depth_map)
        depth_max = np.max(depth_map)
        depth_normalized = (depth_map - depth_min) / (depth_max - depth_min)
        
        # 应用引导滤波
        depth_float32 = depth_normalized.astype(np.float32)
        guide_float32 = guide.astype(np.float32) / 255.0
        refined = cv2.ximgproc.guidedFilter(guide_float32, depth_float32, radius, eps)
        
        # 恢复原始深度范围
        refined = refined * (depth_max - depth_min) + depth_min
        
        return refined
    
    def _enhance_with_superpixel(self, depth_map, rgb_image, n_segments=500):
        """使用超像素分割进行深度优化"""
        # 执行超像素分割
        segments = slic(rgb_image, n_segments=n_segments, compactness=10)
        
        # 为每个超像素区域计算深度中值
        refined_depth = depth_map.copy()
        for segment_id in np.unique(segments):
            mask = segments == segment_id
            depth_values = depth_map[mask]
            
            # 使用中值滤波去除离群值
            if len(depth_values) > 0:
                median_depth = np.median(depth_values)
                
                # 应用平滑过渡，避免超像素边界处的突变
                weight = 0.7  # 控制原始深度和中值之间的权重
                refined_depth[mask] = refined_depth[mask] * (1 - weight) + median_depth * weight
        
        return refined_depth
    
    def _enhance_with_structure_tensor(self, depth_map, rgb_image):
        """基于结构张量的深度优化"""
        if len(rgb_image.shape) > 2:
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = rgb_image.copy()
            
        # 计算结构张量的特征值和特征向量
        dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        dxx = dx * dx
        dxy = dx * dy
        dyy = dy * dy
        
        # 使用高斯滤波平滑结构张量
        window_size = 5
        sigma = 1.5
        dxx = cv2.GaussianBlur(dxx, (window_size, window_size), sigma)
        dxy = cv2.GaussianBlur(dxy, (window_size, window_size), sigma)
        dyy = cv2.GaussianBlur(dyy, (window_size, window_size), sigma)
        
        # 计算结构张量的特征值
        sqrt_term = np.sqrt((dxx - dyy)**2 + 4 * dxy**2)
        lambda1 = 0.5 * (dxx + dyy + sqrt_term)
        lambda2 = 0.5 * (dxx + dyy - sqrt_term)
        
        # 计算各向异性比例，用于检测边缘和结构
        coherence = np.zeros_like(lambda1)
        mask = lambda1 + lambda2 > 1e-10
        coherence[mask] = (lambda1[mask] - lambda2[mask]) / (lambda1[mask] + lambda2[mask])
        
        # 根据结构信息自适应调整深度
        refined_depth = depth_map.copy()
        
        # 在强结构区域保持原始细节，在弱结构区域进行平滑
        smooth_depth = cv2.GaussianBlur(depth_map, (5, 5), 1.0)
        weight_map = coherence  # 结构强度作为权重
        
        refined_depth = depth_map * weight_map + smooth_depth * (1 - weight_map)
        
        return refined_depth
    
    def _apply_cross_bilateral(self, depth_map, rgb_image, d=9, sigma_color=75, sigma_space=75):
        """应用交叉双边滤波优化深度图"""
        # 归一化深度图以用于滤波
        depth_min = np.min(depth_map)
        depth_max = np.max(depth_map)
        depth_normalized = (depth_map - depth_min) / (depth_max - depth_min)
        depth_normalized = (depth_normalized * 255).astype(np.uint8)
        
        # 应用交叉双边滤波，以RGB图像为引导
        refined = cv2.jointBilateralFilter(rgb_image, depth_normalized, d, sigma_color, sigma_space)
        
        # 恢复原始深度范围
        refined = refined.astype(float) / 255.0 * (depth_max - depth_min) + depth_min
        
        return refined

def optimize_depth_with_rgb(depth_map, rgb_image, iterations=2):
    """
    多迭代优化深度图质量
    
    参数:
        depth_map: 初始深度图
        rgb_image: RGB引导图像
        iterations: 优化迭代次数
        
    返回:
        optimized_depth: 优化后的深度图
    """
    optimizer = ImageGuidedDepthOptimizer(
        use_superpixel=True, 
        use_structure_tensor=True,
        use_cross_bilateral=True
    )
    
    current_depth = depth_map.copy()
    
    for i in range(iterations):
        # 每次迭代应用完整的优化流程
        current_depth = optimizer.optimize(current_depth, rgb_image)
        
        # 可选：在迭代间应用其他滤波器
        if i < iterations - 1:
            # 使用中值滤波消除可能引入的噪点
            current_depth = cv2.medianBlur(current_depth.astype(np.float32), 3)
    
    return current_depth


# ZoeDepth集成示例
def integrate_with_zoedepth(model, rgb_image):
    """
    将优化器集成到ZoeDepth框架中
    
    参数:
        model: ZoeDepth模型
        rgb_image: 输入RGB图像
        
    返回:
        refined_depth: 优化后的深度图
    """
    # 1. 使用ZoeDepth获取初始深度估计
    with torch.no_grad():
        rgb_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        rgb_tensor = F.interpolate(rgb_tensor, (384, 512), mode='bilinear', align_corners=True)
        
        # 假设模型输出深度图
        depth_pred = model(rgb_tensor)
        
        # 将深度图调整回原始尺寸
        depth_pred = F.interpolate(depth_pred, (rgb_image.shape[0], rgb_image.shape[1]), 
                                  mode='bilinear', align_corners=True)
        
        depth_pred = depth_pred.squeeze().cpu().numpy()
    
    # 2. 应用图像引导优化
    optimizer = ImageGuidedDepthOptimizer()
    refined_depth = optimizer.optimize(depth_pred, rgb_image)
    
    return refined_depth