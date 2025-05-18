import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from MVSNet.models import *

class ScaleConsistencyOptimizer(nn.Module):
    def __init__(self, config):
        super(ScaleConsistencyOptimizer, self).__init__()
        self.config = config
        
        # 加载ZoeDepth模型
        zoe_config = get_config("zoedepth_nk", "infer")
        self.zoe_model = build_model(zoe_config)
        self.zoe_model.eval()  # 设置为评估模式
        
        # 加载MVSNet模型
        self.mvs_model = MVSNet(self.config.mvs_config)
        checkpoint = torch.load(self.config.mvs_checkpoint)
        self.mvs_model.load_state_dict(checkpoint['model'])
        self.mvs_model.eval()  # 设置为评估模式
        
        # 初始化尺度和偏移参数（可学习）
        self.scales = nn.Parameter(torch.ones(self.config.num_views))
        self.biases = nn.Parameter(torch.zeros(self.config.num_views))
        
        # 将模型转移到GPU
        if torch.cuda.is_available():
            self.zoe_model = self.zoe_model.cuda()
            self.mvs_model = self.mvs_model.cuda()

def compute_scale_consistency_loss(self, zoe_depths, mvs_depth, valid_mask):
    """
    计算尺度一致性损失
    
    Args:
        zoe_depths: ZoeDepth输出的深度图列表 [N, H, W]
        mvs_depth: MVSNet输出的深度图 [H, W]
        valid_mask: 有效像素掩码 [H, W]
        
    Returns:
        scale_loss: 尺度一致性损失
    """
    ref_idx = 0  # 参考视图索引
    ref_depth_scaled = self.scales[ref_idx] * zoe_depths[ref_idx] + self.biases[ref_idx]
    
    # 计算尺度一致性损失
    scale_loss = torch.sum(torch.abs(ref_depth_scaled - mvs_depth) * valid_mask) / (torch.sum(valid_mask) + 1e-7)
    
    return scale_loss

def compute_multi_view_consistency_loss(self, zoe_depths, cam_params, ref_idx=0):
    """
    计算多视图一致性损失
    
    Args:
        zoe_depths: ZoeDepth输出的深度图列表 [N, H, W]
        cam_params: 相机参数列表，包含内参和外参
        ref_idx: 参考视图索引
        
    Returns:
        mv_loss: 多视图一致性损失
    """
    N = len(zoe_depths)
    device = zoe_depths[0].device
    H, W = zoe_depths[0].shape
    
    # 获取参考视图的缩放深度
    ref_depth_scaled = self.scales[ref_idx] * zoe_depths[ref_idx] + self.biases[ref_idx]
    
    # 参考视图相机参数
    ref_K = cam_params[ref_idx]['K']  # 内参
    ref_R = cam_params[ref_idx]['R']  # 旋转矩阵
    ref_t = cam_params[ref_idx]['t']  # 平移向量
    
    # 创建像素坐标
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device))
    xy_homo = torch.stack([x, y, torch.ones_like(x)], dim=0).reshape(3, -1)  # [3, H*W]
    
    # 反投影到3D空间
    ref_depth_flat = ref_depth_scaled.reshape(-1)  # [H*W]
    ref_points3d = ref_depth_flat.unsqueeze(0) * torch.matmul(torch.inverse(ref_K), xy_homo)  # [3, H*W]
    ref_points3d = torch.cat([ref_points3d, torch.ones(1, H*W, device=device)], dim=0)  # [4, H*W]
    
    mv_loss = 0.0
    valid_count = 0
    
    # 对每个非参考视图计算一致性损失
    for i in range(N):
        if i == ref_idx:
            continue
            
        # 目标视图相机参数
        tgt_K = cam_params[i]['K']
        tgt_R = cam_params[i]['R']
        tgt_t = cam_params[i]['t']
        
        # 计算从参考视图到目标视图的变换矩阵
        T_ref_to_tgt = torch.zeros(4, 4, device=device)
        T_ref_to_tgt[:3, :3] = torch.matmul(tgt_R, torch.inverse(ref_R))
        T_ref_to_tgt[:3, 3] = tgt_t - torch.matmul(T_ref_to_tgt[:3, :3], ref_t)
        T_ref_to_tgt[3, 3] = 1.0
        
        # 将参考视图的3D点投影到目标视图
        tgt_points3d = torch.matmul(T_ref_to_tgt, ref_points3d)  # [4, H*W]
        tgt_points2d_homo = torch.matmul(tgt_K, tgt_points3d[:3, :])  # [3, H*W]
        tgt_z = tgt_points2d_homo[2, :]  # [H*W]
        tgt_xy = tgt_points2d_homo[:2, :] / (tgt_z.unsqueeze(0) + 1e-7)  # [2, H*W]
        
        # 计算目标视图中的有效像素
        valid_x = (tgt_xy[0, :] >= 0) & (tgt_xy[0, :] < W)
        valid_y = (tgt_xy[1, :] >= 0) & (tgt_xy[1, :] < H)
        valid_z = tgt_z > 0
        valid_mask = valid_x & valid_y & valid_z  # [H*W]
        
        if torch.sum(valid_mask) < 100:  # 如果有效像素太少，则跳过
            continue
            
        # 对目标视图的深度进行插值
        tgt_depth_scaled = self.scales[i] * zoe_depths[i] + self.biases[i]  # [H, W]
        tgt_xy_valid = tgt_xy[:, valid_mask].t()  # [Valid, 2]
        tgt_depth_interp = self.bilinear_interpolate(tgt_depth_scaled, tgt_xy_valid[:, 0], tgt_xy_valid[:, 1])  # [Valid]
        
        # 计算深度一致性损失
        proj_z_valid = tgt_z[valid_mask]  # [Valid]
        depth_diff = torch.abs(tgt_depth_interp - proj_z_valid)
        mv_loss += torch.mean(depth_diff)
        valid_count += 1
    
    if valid_count > 0:
        mv_loss /= valid_count
    
    return mv_loss

def compute_smoothness_loss(self, depths, images):
    """
    计算深度平滑性损失
    
    Args:
        depths: 深度图列表 [N, H, W]
        images: 图像列表 [N, C, H, W]
        
    Returns:
        smooth_loss: 平滑性损失
    """
    N = len(depths)
    smooth_loss = 0.0
    
    for i in range(N):
        depth = self.scales[i] * depths[i] + self.biases[i]
        image = images[i]
        
        # 计算深度梯度
        dy_depth, dx_depth = self.compute_gradients(depth)
        
        # 计算图像梯度
        dy_img, dx_img = self.compute_gradients(image.mean(dim=0))
        
        # 权重计算：图像梯度大的地方，深度梯度的权重应该小
        weights_x = torch.exp(-torch.mean(torch.abs(dx_img), dim=0))
        weights_y = torch.exp(-torch.mean(torch.abs(dy_img), dim=0))
        
        # 加权平滑损失
        smooth_loss += torch.mean(weights_x * torch.abs(dx_depth)) + torch.mean(weights_y * torch.abs(dy_depth))
    
    return smooth_loss / N

def forward(self, images, cam_params):
    """
    前向传播
    
    Args:
        images: 输入图像列表 [N, C, H, W]
        cam_params: 相机参数列表
        
    Returns:
        losses: 包含各损失项的字典
        outputs: 包含优化后深度的字典
    """
    N = len(images)
    device = images[0].device
    
    # 使用ZoeDepth生成深度图
    zoe_depths = []
    with torch.no_grad():
        for img in images:
            depth = self.zoe_model.infer(img)  # [1, H, W]
            zoe_depths.append(depth.squeeze(0))
    
    # 使用MVSNet生成参考视图的深度图
    ref_idx = 0
    with torch.no_grad():
        mvs_depth, prob = self.mvs_model(images, cam_params, ref_idx)
    
    # 计算有效像素掩码
    valid_mask = (mvs_depth > 0) & (mvs_depth < self.config.max_depth)
    
    # 计算各损失项
    scale_loss = self.compute_scale_consistency_loss(zoe_depths, mvs_depth, valid_mask)
    mv_loss = self.compute_multi_view_consistency_loss(zoe_depths, cam_params, ref_idx)
    smooth_loss = self.compute_smoothness_loss(zoe_depths, images)
    
    # 总损失
    total_loss = (self.config.lambda1 * scale_loss + 
                  self.config.lambda2 * mv_loss + 
                  self.config.lambda3 * smooth_loss)
    
    # 返回损失和输出
    losses = {
        'scale_loss': scale_loss,
        'mv_loss': mv_loss,
        'smooth_loss': smooth_loss,
        'total_loss': total_loss
    }
    
    # 生成优化后的深度图
    optimized_depths = [self.scales[i] * zoe_depths[i] + self.biases[i] for i in range(N)]
    
    outputs = {
        'zoe_depths': zoe_depths,
        'mvs_depth': mvs_depth,
        'optimized_depths': optimized_depths,
        'scales': self.scales.detach(),
        'biases': self.biases.detach()
    }
    
    return losses, outputs

def train_scale_optimizer(config):
    """
    训练尺度一致性优化器
    
    Args:
        config: 配置对象
    """
    # 创建模型
    model = ScaleConsistencyOptimizer(config)
    if torch.cuda.is_available():
        model = model.cuda()
    
    # 创建优化器
    optimizer = optim.Adam([
        {'params': model.scales, 'lr': config.lr_scale},
        {'params': model.biases, 'lr': config.lr_bias}
    ])
    
    # 加载数据集
    train_loader = get_data_loader(config.dataset_path, config.batch_size, is_train=True)
    
    # 训练循环
    for epoch in range(config.num_epochs):
        model.train()
        epoch_losses = {
            'scale_loss': 0.0,
            'mv_loss': 0.0,
            'smooth_loss': 0.0,
            'total_loss': 0.0
        }
        
        for batch_idx, batch in enumerate(train_loader):
            images = [img.cuda() if torch.cuda.is_available() else img for img in batch['images']]
            cam_params = [{k: v.cuda() if torch.cuda.is_available() else v for k, v in cam.items()} 
                          for cam in batch['cam_params']]
            
            # 前向传播
            losses, outputs = model(images, cam_params)
            
            # 反向传播
            optimizer.zero_grad()
            losses['total_loss'].backward()
            optimizer.step()
            
            # 更新损失统计
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            
            # 打印批次信息
            if (batch_idx + 1) % config.print_freq == 0:
                print(f"Epoch [{epoch+1}/{config.num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], "
                      f"Scale Loss: {losses['scale_loss'].item():.4f}, "
                      f"MV Loss: {losses['mv_loss'].item():.4f}, "
                      f"Smooth Loss: {losses['smooth_loss'].item():.4f}, "
                      f"Total Loss: {losses['total_loss'].item():.4f}")
        
        # 打印每轮训练信息
        for key in epoch_losses:
            epoch_losses[key] /= len(train_loader)
        print(f"Epoch [{epoch+1}/{config.num_epochs}], "
              f"Scale Loss: {epoch_losses['scale_loss']:.4f}, "
              f"MV Loss: {epoch_losses['mv_loss']:.4f}, "
              f"Smooth Loss: {epoch_losses['smooth_loss']:.4f}, "
              f"Total Loss: {epoch_losses['total_loss']:.4f}")
        
        # 保存模型
        if (epoch + 1) % config.save_freq == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scales': model.scales.detach().cpu(),
                'biases': model.biases.detach().cpu()
            }, os.path.join(config.save_dir, f'checkpoint_epoch{epoch+1}.pth'))

def bilinear_interpolate(self, depth, x, y):
    """
    双线性插值函数
    
    Args:
        depth: 深度图 [H, W]
        x, y: 需要插值的坐标 [N]
        
    Returns:
        interpolated: 插值后的深度值 [N]
    """
    H, W = depth.shape
    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1
    
    # 裁剪到图像范围内
    x0 = torch.clamp(x0, 0, W - 1)
    x1 = torch.clamp(x1, 0, W - 1)
    y0 = torch.clamp(y0, 0, H - 1)
    y1 = torch.clamp(y1, 0, H - 1)
    
    # 计算插值权重
    wa = (x1.float() - x) * (y1.float() - y)
    wb = (x1.float() - x) * (y - y0.float())
    wc = (x - x0.float()) * (y1.float() - y)
    wd = (x - x0.float()) * (y - y0.float())
    
    # 获取四个相邻点的值
    va = depth[y0, x0]
    vb = depth[y1, x0]
    vc = depth[y0, x1]
    vd = depth[y1, x1]
    
    # 计算插值结果
    interpolated = wa * va + wb * vb + wc * vc + wd * vd
    
    return interpolated

def compute_gradients(self, image):
    """
    计算图像梯度
    
    Args:
        image: 输入图像 [H, W]
        
    Returns:
        dy, dx: y方向和x方向的梯度
    """
    padded = torch.nn.functional.pad(image, (1, 1, 1, 1), mode='replicate')
    dy = padded[2:, 1:-1] - padded[:-2, 1:-1]
    dx = padded[1:-1, 2:] - padded[1:-1, :-2]
    return dy, dx
