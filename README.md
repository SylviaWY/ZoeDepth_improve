# ZoeDepth Multi-View Fusion 深度增强框架

<img src="C:\Users\YY\Desktop\课内\cv\24210720276_王杨_hw2\output_results\evaluation_vis\10_visualization.png" alt="10_visualization" style="zoom:100%;" />

这个仓库包含了基于ZoeDepth单目深度估计模型的多视角融合增强框架的实现，旨在通过多视角几何约束和RGB图像引导的滤波技术来提高深度图质量。

## 项目概述

本项目结合了基于深度学习的单目深度估计方法和基于几何的多视角立体视觉技术，系统性地解决了单目深度估计中的三个核心挑战：

- 尺度不一致性
- 领域泛化能力不足
- 结构细节保留困难

通过构建多阶段深度图优化管线，项目融合了三个核心模块：

1. RGB图像引导的边缘增强
2. 多视角几何一致性约束
3. 自适应融合策略

## 目录结构

```
├── ZoeDepth/                    # ZoeDepth模型及扩展代码
│   ├── test_scripts/            # 测试脚本
│   ├── evaluation/              # 评估代码
│   └── ...                      # 其他ZoeDepth原始文件
├── output_results/              # 输出结果
│   ├── evaluation_vis/          # 对比图片
│   ├── _colored/                # 可视化后的深度图
│   ├── _raw/                    # 接收端深度图
│   └── metrics/                 # 评估指标结果
├── AdaptiveViewWeighting.py     # 自适应视角权重计算
├── ConfidenceBasedIntegration.py # 基于置信度的深度集成
├── edge_preserving_refinement.py # 边缘保留深度滤波
├── image_guided_optimization.py  # 图像引导深度优化
├── MultiViewVoting.py           # 多视角投票机制
├── Scale_harmonization.py       # 尺度一致性优化
└── README.md                   # 本文件
```

## 主要模块

### 1. RGB图像引导的深度增强

在`edge_preserving_refinement.py`和`image_guided_optimization.py`中实现了基于RGB图像的深度增强技术：

- **双边滤波**：保留边缘的同时平滑深度图
- **引导滤波**：使用RGB图像结构引导深度图优化
- **超像素分割辅助优化**：使用SLIC算法进行区域一致化
- **结构张量分析**：识别并保留图像中的重要结构

### 2. 多视角几何一致性

在`MultiViewVoting.py`和`Scale_harmonization.py`中实现了基于多视角几何的一致性优化：

- **尺度一致性优化**：解决单目深度估计的尺度不确定性
- **多视角投票机制**：通过投影和直方图分析提高深度的几何一致性
- **跨视角深度验证**：确保深度估计在不同视角下保持几何兼容性

### 3. 自适应融合策略

在`AdaptiveViewWeighting.py`和`ConfidenceBasedIntegration.py`中实现了深度融合策略：

- **基于置信度的深度集成**：计算深度图置信度并加权融合
- **自适应视角权重**：考虑视角角度差异、深度完整性和平均置信度
- **梯度感知置信度计算**：在边缘区域降低置信度以保留结构

## 实验结果

通过在194张多视角图像的自建数据集上进行的量化实验，本框架实现了以下改进：

- RGB图像引导的深度优化阶段降低了11.9%的平均深度误差和10.0%的深度值波动
- 多视角投票机制和基于置信度的自适应融合策略最大降低了20.2%的平均深度误差和24.1%的深度值波动
- 边缘一致性指数(ECI)从0.72提高到0.91，显著提升了深度图与RGB图像边缘的对齐精度

详细的评估指标可在`output_results/metrics/`目录下查看。

## 使用方法

### 环境配置

```bash
# 使用conda创建环境
conda env create -n zoe --file environment.yml
conda activate zoe

# 或使用pip安装依赖
pip install torch==1.13.1 torchvision==0.14.1 timm==0.6.12 opencv-python==4.6.0 wandb==0.13.9
```

### 运行测试

```bash
# 单图深度估计测试
python ZoeDepth/test_scripts/test_single_image.py --image path/to/image.jpg --output output_results/

# 多视角融合测试
python ZoeDepth/test_scripts/test_multiview_fusion.py --image_dir path/to/images/ --output output_results/
```

### 评估指标计算

```bash
python ZoeDepth/evaluation/calculate_metrics.py --pred_dir output_results/raw/ --gt_dir path/to/ground_truth/ --save_dir output_results/metrics/
```

## 引用

如果您在研究中使用了此代码，请引用：

```
@misc{wang2025zoedepth,
  author = {Wang Yang},
  title = {ZoeDepth Multi-View Fusion Framework},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/SylviaWY/ZoeDepth_improve}}
}
```

## 致谢

本项目基于以下工作：

- [ZoeDepth](https://github.com/isl-org/ZoeDepth)：零样本迁移单目深度估计框架
- [COLMAP](https://github.com/colmap/colmap)：用于SfM和MVS的开源库

