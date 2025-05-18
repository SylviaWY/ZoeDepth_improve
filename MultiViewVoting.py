import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

class MultiViewVoting:
    def __init__(self, bin_size=0.05, max_depth=10.0):
        """
        初始化多视图投票机制
        
        参数:
        - bin_size: 深度直方图的箱宽
        - max_depth: 最大深度值
        """
        self.bin_size = bin_size
        self.max_depth = max_depth
        self.num_bins = int(max_depth / bin_size) + 1
        
    def project_depth_to_reference(self, depth_map, K, T_src_to_ref):
        """
        将源视角的深度图投影到参考视角
        
        参数:
        - depth_map: 源视角的深度图
        - K: 相机内参矩阵
        - T_src_to_ref: 从源视角到参考视角的变换矩阵
        
        返回:
        - projected_depth: 投影后的深度图
        - valid_mask: 有效投影区域的掩码
        """
        height, width = depth_map.shape
        
        # 创建像素坐标网格
        y_coords, x_coords = torch.meshgrid(
            torch.arange(height, device=depth_map.device),
            torch.arange(width, device=depth_map.device)
        )
        
        # 将像素坐标转换为归一化相机坐标
        z = depth_map
        x = (x_coords - K[0, 2]) * z / K[0, 0]
        y = (y_coords - K[1, 2]) * z / K[1, 1]
        
        # 构建3D点云
        points_3d = torch.stack([x, y, z, torch.ones_like(z)], dim=-1)
        
        # 应用变换
        T = torch.tensor(T_src_to_ref, device=depth_map.device)
        points_transformed = torch.matmul(points_3d.view(-1, 4), T.T).view(height, width, 4)
        
        # 投影回参考视角
        depth_projected = points_transformed[..., 2]
        x_projected = points_transformed[..., 0] / points_transformed[..., 2] * K[0, 0] + K[0, 2]
        y_projected = points_transformed[..., 1] / points_transformed[..., 2] * K[1, 1] + K[1, 2]
        
        # 创建网格坐标用于插值
        grid_x = 2.0 * (x_projected / (width - 1)) - 1.0
        grid_y = 2.0 * (y_projected / (height - 1)) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1)
        
        # 有效投影判断
        valid_mask = (grid_x >= -1) & (grid_x <= 1) & (grid_y >= -1) & (grid_y <= 1) & (depth_projected > 0)
        
        return depth_projected, valid_mask
    
    def vote(self, depth_maps, Ks, transforms):
        """
        执行多视图投票
        
        参数:
        - depth_maps: 多视角的深度图列表
        - Ks: 相机内参矩阵列表
        - transforms: 从各视角到参考视角的变换矩阵列表
        
        返回:
        - final_depth: 投票后的最终深度图
        """
        # 使用第一个视角作为参考
        ref_depth = depth_maps[0]
        height, width = ref_depth.shape
        
        # 初始化直方图
        histograms = torch.zeros((height, width, self.num_bins), device=ref_depth.device)
        
        # 将每个视角的深度投影到参考视角并更新直方图
        for i in range(len(depth_maps)):
            if i == 0:  # 参考视角
                depths = ref_depth
                valid_mask = depths > 0
            else:
                depths, valid_mask = self.project_depth_to_reference(
                    depth_maps[i], Ks[i], transforms[i]
                )
            
            # 更新直方图
            bin_indices = (depths / self.bin_size).long()
            bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)
            
            # 仅考虑有效区域
            y_coords, x_coords = torch.where(valid_mask)
            bin_indices_valid = bin_indices[valid_mask]
            
            for j in range(len(y_coords)):
                y, x = y_coords[j], x_coords[j]
                bin_idx = bin_indices_valid[j]
                histograms[y, x, bin_idx] += 1
        
        # 找出每个像素的最高峰值
        max_bin_indices = torch.argmax(histograms, dim=2)
        final_depth = max_bin_indices.float() * self.bin_size
        
        return final_depth