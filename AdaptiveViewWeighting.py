class AdaptiveViewWeighting:
    def __init__(self, alpha=0.4, beta=0.3, gamma=0.3):
        """
        初始化视角权重自适应调整
        
        参数:
        - alpha: 视角角度差异的权重系数
        - beta: 深度完整性的权重系数
        - gamma: 平均置信度的权重系数
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.confidence_estimator = ConfidenceBasedIntegration()
        
    def compute_view_weight(self, depth_map, ref_pose, view_pose, valid_mask=None):
        """
        计算视角权重
        
        参数:
        - depth_map: 当前视角的深度图
        - ref_pose: 参考视角的位姿
        - view_pose: 当前视角的位姿
        - valid_mask: 有效深度区域掩码（如果为None则自动计算）
        
        返回:
        - weight: 该视角的权重
        """
        # 计算视角角度差异
        rot_ref = Rotation.from_matrix(ref_pose[:3, :3])
        rot_view = Rotation.from_matrix(view_pose[:3, :3])
        rel_rot = rot_ref.inv() * rot_view
        angle_diff = np.abs(rel_rot.magnitude())
        angle_weight = np.exp(-angle_diff / np.pi)
        
        # 计算深度完整性
        if valid_mask is None:
            valid_mask = depth_map > 0
        completion_ratio = torch.sum(valid_mask) / (depth_map.shape[0] * depth_map.shape[1])
        completion_weight = completion_ratio.item()
        
        # 计算平均置信度
        confidence = self.confidence_estimator.compute_confidence(depth_map)
        avg_confidence = torch.mean(confidence[valid_mask]).item()
        
        # 综合计算权重
        weight = self.alpha * angle_weight + self.beta * completion_weight + self.gamma * avg_confidence
        
        return weight
    
    def adjust_weights(self, depth_maps, poses, ref_index=0):
        """
        调整多视角的权重
        
        参数:
        - depth_maps: 多视角深度图列表
        - poses: 相机位姿列表
        - ref_index: 参考视角的索引
        
        返回:
        - weights: 调整后的权重列表
        """
        ref_pose = poses[ref_index]
        weights = []
        
        for i, (depth_map, pose) in enumerate(zip(depth_maps, poses)):
            if i == ref_index:
                # 参考视角给予较高权重
                weight = 1.0
            else:
                weight = self.compute_view_weight(depth_map, ref_pose, pose)
            
            weights.append(weight)
        
        # 归一化权重
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        return normalized_weights

class MultiViewFusion:
    def __init__(self):
        """
        初始化多视图融合系统
        """
        self.voter = MultiViewVoting()
        self.integrator = ConfidenceBasedIntegration()
        self.weight_adjuster = AdaptiveViewWeighting()
        
    def fuse_depth_maps(self, depth_maps, camera_params, poses, ref_index=0):
        """
        融合多视角深度图
        
        参数:
        - depth_maps: 多视角深度图列表
        - camera_params: 相机参数列表
        - poses: 相机位姿列表
        - ref_index: 参考视角索引
        
        返回:
        - fused_depth: 融合后的深度图
        """
        # 计算各视角权重
        weights = self.weight_adjuster.adjust_weights(depth_maps, poses, ref_index)
        
        # 投影转换矩阵计算
        ref_pose = poses[ref_index]
        transforms = []
        for pose in poses:
            T = np.linalg.inv(ref_pose) @ pose
            transforms.append(T)
        
        # 多视图投票
        voted_depth = self.voter.vote(depth_maps, camera_params, transforms)
        
        # 计算置信度图
        confidence_maps = [self.integrator.compute_confidence(d) for d in depth_maps]
        
        # 应用权重到置信度图
        weighted_confidence = [conf * w for conf, w in zip(confidence_maps, weights)]
        
        # 基于置信度和权重的深度集成
        fused_depth = self.integrator.integrate_depths(depth_maps, weighted_confidence)
        
        return fused_depth