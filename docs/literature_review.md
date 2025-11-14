# 无创心梗估计文献调研笔记

## 任务定义
- **目标**: 从CMR图像直接预测心梗分割
- **输入数据**:
  - CMR图像: (T, C, H, W) - T帧动态序列
  - LGE图像: (1, H, W) - 单帧延迟增强图像
  - CMR第一帧心肌mask
  - LGE心肌和心梗mask
- **核心挑战**: LGE和CMR之间存在配准问题（移位、形变、帧数不匹配）
- **关键思路**: 利用运动场估计心肌运动来预测心梗位置

## 相关研究领域

### 1. CMR到LGE的预测与配准
- **Cine-Generated Enhancement (CGE)** - AHA Journals 2024
  - URL: https://www.ahajournals.org/doi/10.1161/CIRCIMAGING.124.016786
  - 深度生成学习方法，将无对比剂的cine CMR转换为LGE
  
- **Cine-LGE配准** - Guo et al. 2021
  - URL: https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.28596
  - 自动化cine-LGE MRI配准和心梗异质性量化
  
- **联合深度学习框架** - Upendra et al. 2021
  - URL: https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11598/115980F/Joint-deep-learning-framework-for-image-registration-and-segmentation-of/10.1117/12.2581386.full
  - STN-based RoI引导的CNN用于LGE和cine CMR配准

- **Cross-Modality分割** - Wang et al. 2022
  - URL: https://ieeexplore.ieee.org/document/9669121/
  - 使用多风格迁移网络进行无监督LGE-CMR分割

### 2. 心脏运动场估计
- **Motion Pyramid Networks** - Yu et al. 2020
  - URL: https://arxiv.org/abs/2006.15710
  - 准确高效的心脏运动估计深度学习方法
  
- **DeepMesh** - Meng et al. 2024
  - URL: https://pubmed.ncbi.nlm.nih.gov/38064325/
  - 基于网格的心脏运动追踪，从2D图像估计3D运动
  
- **光流法** - Amartur et al. 1993, Wang et al. 2019
  - URL: https://www.sciencedirect.com/science/article/abs/pii/S1361841519300581
  - 基于梯度的光流心脏运动估计方法

### 3. CMR特征追踪(Feature Tracking)
- **CMR-FT综述** - Schuster et al. 2016
  - URL: https://www.ahajournals.org/doi/10.1161/circimaging.115.004077
  - 自动追踪心脏结构，评估应变、位移、速度
  
- **心肌应变与疤痕检测** - Eitel et al. 2018
  - URL: https://www.jacc.org/doi/10.1016/j.jcmg.2017.11.034
  - CMR-FT用于心梗后预后评估

### 4. LGE分割与心梗量化
- **MS-CMR挑战赛** - Zhuang et al. 2022
  - URL: https://pubmed.ncbi.nlm.nih.gov/35834896/
  - MICCAI 2019多序列心脏MR分割挑战
  
- **深度学习心梗分割** - Zhang et al. 2022
  - URL: https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.122.060137
  - 使用深度学习进行心梗疤痕评估

## 待深入阅读的关键论文
1. Cine-Generated Enhancement (CGE) - 直接从cine预测LGE
2. Motion Pyramid Networks - 运动场估计
3. Joint registration and segmentation framework - 配准+分割联合学习
4. DeepMesh - 3D运动追踪
5. MS-CMR Challenge - 公开数据集和基准方法

## 下一步行动
- 访问关键论文获取详细技术细节
- 搜索公开数据集（MS-CMR, ACDC等）
- 设计技术方案架构


## Motion Pyramid Networks详细信息

**论文**: Motion Pyramid Networks for Accurate and Efficient Cardiac Motion Estimation (2020)
**作者**: Hanchao Yu, Xiao Chen, Humphrey Shi, Terrence Chen, Thomas S. Huang, Shanhui Sun
**机构**: United Imaging Intelligence, UIUC, University of Oregon

### 核心方法
1. **金字塔运动场预测**: 从多尺度特征表示预测并融合运动场金字塔，生成更精细的运动场
2. **循环师生训练策略**: 
   - 教师模型通过渐进式运动补偿提供更准确的运动估计作为监督
   - 学生模型从教师模型学习，在单步中估计运动同时保持准确性
   - 循环知识蒸馏进一步提升性能
3. **应用**: MRI心脏特征追踪、心肌应变评估

### 技术特点
- 端到端推理
- 多尺度特征融合
- 运动补偿机制
- 临床相关的评估指标

### 相关性
该方法可用于从CMR序列估计心肌运动场，为预测心梗位置提供运动信息。


## 公开数据集信息

### MS-CMR 2019 数据集
**官网**: https://zmiclab.github.io/zxh/0/mscmrseg19/
**数据规模**: 45例多序列CMR图像（心肌病患者）
**序列类型**:
- LGE (Late Gadolinium Enhancement) - 延迟增强，显示心梗区域
- T2-weighted CMR - 显示急性损伤和缺血区域
- bSSFP cine - 捕捉心脏运动，边界清晰

**标注内容**: 心室和心肌分割（包含测试集金标准）
**任务**: 从LGE CMR分割心室和心肌，可结合T2和bSSFP辅助
**获取方式**: 需要注册并签署协议
**相关论文**:
1. Zhuang X. Multivariate mixture model for myocardial segmentation combining multi-source images. IEEE T-PAMI 2019
2. Wu F & Zhuang X. Minimizing Estimated Risks on Unlabeled Data. IEEE T-PAMI 2023

### ACDC 数据集
**官网**: https://www.creatis.insa-lyon.fr/Challenge/acdc/
**数据来源**: 法国第戎大学医院真实临床检查数据
**数据类型**: 心脏磁共振(MR)图像 + 分割标签
**特点**: 完全匿名化，多种心脏疾病类型
**应用**: 自动化心脏诊断

### EMIDEC 数据集
**官网**: https://emidec.com/
**专注**: 延迟增强心脏MRI的心梗自动评估
**特点**: 专门用于心梗检测和量化

### CMRxMotion 数据集
**官网**: https://www.synapse.org/Synapse:syn28503327
**特点**: 包含不同程度呼吸运动的极端病例
**应用**: 心脏运动分析

### M&Ms Challenge
**官网**: https://www.ub.edu/mnms/
**目标**: 构建可跨临床中心应用的通用模型


## 深度学习配准方法

### VoxelMorph
**论文**: VoxelMorph: A Learning Framework for Deformable Medical Image Registration (2019)
**作者**: Balakrishnan et al.
**引用次数**: 2628+
**GitHub**: https://github.com/voxelmorph/voxelmorph
**核心特点**:
- 快速的基于学习的可变形医学图像配准框架
- 无监督学习方法
- 端到端训练
- 可生成变形场(deformation field)
- 适用于多模态医学图像配准

**改进版本**:
- Swin-VoxelMorph (2022): 基于Swin Transformer的对称无监督学习模型

### Spatial Transformer Network (STN)
**应用场景**:
- 医学图像配准
- 结构引导的图像配准
- 运动追踪和应变分析
- 姿态不变性学习

**相关工作**:
- Image-and-Spatial Transformer Networks (ISTNs) - Lee et al. 2019
- Co-Attention STN for motion tracking - Ahn et al. 2022

### Deep Learning Image Registration (DLIR)
**论文**: De Vos et al. 2019
**引用次数**: 963+
**特点**: 无监督仿射和可变形图像配准框架
