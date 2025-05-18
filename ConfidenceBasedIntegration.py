import torch
import torch.nn.functional as F

class ConfidenceBasedIntegration:
    def __init__(self, sigma_1=0.1, sigma_2=0.3):
        """
        初始化基于置信度的深度集成
        
        参数:
        - sigma_1: 深度梯度的控制参数
        - sigma_2: 深度方差的控制参数
        """
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        
    def compute_confidence(self, depth_map):
        """
        计算深度图的置信度
        
        参数:
        - depth_map: 输入深度图
        
        返回:
        - confidence: 置信度图
        """
        # 计算深度梯度
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=depth_map.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=depth_map.device)
        
        sobel_x = sobel_x.view(1, 1, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3)
        
        depth_pad = F.pad(depth_map.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
        
        grad_x = F.conv2d(depth_pad, sobel_x)
        grad_y = F.conv2d(depth_pad, sobel_y)
        
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        
        # 基于梯度的置信度（梯度大的区域置信度低）
        confidence_gradient = torch.exp(-gradient_magnitude / self.sigma_1)
        
        # 返回置信度图
        return confidence_gradient.squeeze()
    
    def integrate_depths(self, depth_maps, confidence_maps=None):
        """
        基于置信度集成多视角深度
        
        参数:
        - depth_maps: 多视角深度图列表
        - confidence_maps: 预计算的置信度图列表（如果为None则自动计算）
        
        返回:
        - integrated_depth: 集成后的深度图
        """
        if confidence_maps is None:
            confidence_maps = [self.compute_confidence(d) for d in depth_maps]
        
        # 归一化置信度图
        total_confidence = sum(confidence_maps)
        epsilon = 1e-10  # 避免除零
        normalized_confidence = [conf / (total_confidence + epsilon) for conf in confidence_maps]
        
        # 加权融合深度图
        integrated_depth = sum(d * conf for d, conf in zip(depth_maps, normalized_confidence))
        
        # 应用中值滤波平滑处理
        integrated_depth = self._median_filter(integrated_depth)
        
        return integrated_depth
    
    def _median_filter(self, depth_map, kernel_size=3):
        """
        应用中值滤波
        
        参数:
        - depth_map: 输入深度图
        - kernel_size: 滤波核大小
        
        返回:
        - filtered_depth: 滤波后的深度图
        """
        depth_pad = F.pad(
            depth_map.unsqueeze(0).unsqueeze(0), 
            (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2), 
            mode='reflect'
        )
        
        patches = F.unfold(depth_pad, kernel_size=kernel_size, stride=1)
        patches = patches.view(1, kernel_size*kernel_size, -1)
        
        # 计算中值
        median_values, _ = torch.median(patches, dim=1)
        
        # 重塑为深度图
        filtered_depth = median_values.view(depth_map.shape)
        
        return filtered_depth