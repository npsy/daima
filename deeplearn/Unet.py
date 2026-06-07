# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 15:18:24 2025

@author: likaituo
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import time
import math
from typing import Optional, Tuple, List

# 启用GPU优化标志
torch.backends.cudnn.benchmark = True  # 启用cuDNN自动优化
torch.backends.cudnn.enabled = True    # 启用cuDNN

class OptimizedGaussianNoiseUNet(nn.Module):
    """
    固定高斯噪声的UNet去噪器 - GPU优化版本
    使用CBSD68数据集中已有的noisy25图像进行训练
    """
    
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, 
                 use_amp: bool = True):
        super(OptimizedGaussianNoiseUNet, self).__init__()
        self.use_amp = use_amp  # 是否使用混合精度
        
        # 编码器 - 使用深度可分离卷积优化
        self.enc1 = self._optimized_block(in_channels, base_channels)
        self.enc2 = self._optimized_block(base_channels, base_channels * 2)
        self.enc3 = self._optimized_block(base_channels * 2, base_channels * 4)
        self.enc4 = self._optimized_block(base_channels * 4, base_channels * 8)
        self.enc5 = self._optimized_block(base_channels * 8, base_channels * 16)
        
        # 解码器
        self.dec1 = self._optimized_block(base_channels * 16, base_channels * 8)
        self.dec2 = self._optimized_block(base_channels * 8, base_channels * 4)
        self.dec3 = self._optimized_block(base_channels * 4, base_channels * 2)
        self.dec4 = self._optimized_block(base_channels * 2, base_channels)
        
        # 上采样层 - 使用双线性插值而不是转置卷积，避免尺寸问题
        self.up1 = self._create_upsample_block(base_channels * 16, base_channels * 8)
        self.up2 = self._create_upsample_block(base_channels * 8, base_channels * 4)
        self.up3 = self._create_upsample_block(base_channels * 4, base_channels * 2)
        self.up4 = self._create_upsample_block(base_channels * 2, base_channels)
        
        # 输出层
        self.outc = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, 1),
            nn.Sigmoid()  # 输出范围 [0, 1]
        )
        
        self.pool = nn.MaxPool2d(2)
        
        # 缓存以加速相同尺寸的重复计算
        self._size_cache = {}
        
        # 初始化权重
        self._initialize_weights()
    
    def _optimized_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """优化卷积块"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),  # inplace=True减少内存
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def _create_upsample_block(self, in_ch: int, out_ch: int) -> nn.Module:
        """创建上采样块"""
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, out_ch, 1)  # 1x1卷积调整通道数
        )
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _get_cached_sizes(self, input_size: Tuple[int, int]) -> List[Tuple[int, int]]:
        """获取或计算中间特征图尺寸"""
        cache_key = f"{input_size[0]}_{input_size[1]}"
        
        if cache_key not in self._size_cache:
            # 计算UNet各层尺寸
            sizes = [input_size]
            
            # 编码器尺寸（每次池化减半）
            h, w = input_size
            for _ in range(4):  # 4次池化
                h = math.ceil(h / 2)
                w = math.ceil(w / 2)
                sizes.append((h, w))
            
            # 解码器尺寸（与编码器对称）
            for i in range(3, -1, -1):
                sizes.append(sizes[i])
            
            self._size_cache[cache_key] = sizes
        
        return self._size_cache[cache_key]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 使用混合精度
        with torch.cuda.amp.autocast(enabled=self.use_amp and x.is_cuda):
            return self._forward_impl(x)
    
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # 获取输入尺寸
        input_size = x.shape[2:]
        
        # 编码路径
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        
        # 解码路径 + 跳跃连接（优化版本）
        # 使用缓存避免重复计算尺寸
        intermediate_sizes = self._get_cached_sizes(input_size)
        
        # 第一层上采样
        d1 = self.up1(e5)
        # 调整e4的尺寸以匹配d1（如果需要）
        target_size = intermediate_sizes[3]  # e4的尺寸
        if e4.size()[2:] != d1.size()[2:]:
            d1 = F.interpolate(d1, size=target_size, mode='bilinear', align_corners=True)
        d1 = self.dec1(torch.cat([e4, d1], dim=1))
        
        # 第二层上采样
        d2 = self.up2(d1)
        target_size = intermediate_sizes[2]  # e3的尺寸
        if e3.size()[2:] != d2.size()[2:]:
            d2 = F.interpolate(d2, size=target_size, mode='bilinear', align_corners=True)
        d2 = self.dec2(torch.cat([e3, d2], dim=1))
        
        # 第三层上采样
        d3 = self.up3(d2)
        target_size = intermediate_sizes[1]  # e2的尺寸
        if e2.size()[2:] != d3.size()[2:]:
            d3 = F.interpolate(d3, size=target_size, mode='bilinear', align_corners=True)
        d3 = self.dec3(torch.cat([e2, d3], dim=1))
        
        # 第四层上采样
        d4 = self.up4(d3)
        target_size = intermediate_sizes[0]  # e1的尺寸
        if e1.size()[2:] != d4.size()[2:]:
            d4 = F.interpolate(d4, size=target_size, mode='bilinear', align_corners=True)
        d4 = self.dec4(torch.cat([e1, d4], dim=1))
        
        return self.outc(d4)

class GPUCachedCBSD68Dataset(Dataset):
    """使用真实噪声图像的CBSD68去噪数据集 - GPU优化版本"""
    
    def __init__(self, base_data_dir: str, patch_size: int = 128, 
                 transform: Optional[transforms.Compose] = None, 
                 train: bool = True, num_patches_per_image: int = 10,
                 cache_in_ram: bool = True, use_amp: bool = True):
        """
        参数:
            base_data_dir: CBSD68数据集根目录
            patch_size: 训练时裁剪的patch大小
            transform: 数据增强变换
            train: 是否为训练模式
            num_patches_per_image: 每张图像生成的patch数量
            cache_in_ram: 是否缓存图像到RAM
            use_amp: 是否使用混合精度
        """
        self.base_data_dir = base_data_dir
        self.patch_size = patch_size
        self.transform = transform
        self.train = train
        self.num_patches_per_image = num_patches_per_image
        self.cache_in_ram = cache_in_ram
        self.use_amp = use_amp
        
        # 加载图像路径
        self.clean_paths, self.noisy_paths = self._load_image_paths()
        
        # 缓存图像
        if cache_in_ram:
            self._cache_images()
        
        # 预计算随机裁剪位置（加速训练）
        if train:
            self._precompute_crop_positions()
        
        print(f"加载了 {len(self.clean_paths)} 张图像对")
        print(f"数据集目录: {base_data_dir}")
        print(f"Patch大小: {patch_size}x{patch_size}")
        if cache_in_ram:
            print(f"RAM缓存: 已启用")
    
    def _load_image_paths(self) -> Tuple[List[str], List[str]]:
        """加载干净和噪声图像路径"""
        clean_dir = os.path.join(self.base_data_dir, "original")
        noisy_dir = os.path.join(self.base_data_dir, "noisy25")
        
        if not os.path.exists(clean_dir):
            raise FileNotFoundError(f"找不到干净图像目录: {clean_dir}")
        if not os.path.exists(noisy_dir):
            raise FileNotFoundError(f"找不到噪声图像目录: {noisy_dir}")
        
        # 获取所有图像文件
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
        clean_paths = []
        noisy_paths = []
        
        for ext in image_extensions:
            clean_paths.extend(glob.glob(os.path.join(clean_dir, ext)))
        
        # 去重和排序
        clean_paths = sorted(list(set(clean_paths)))
        
        # 为每个干净图像找到对应的噪声图像
        matched_clean_paths = []
        matched_noisy_paths = []
        
        for clean_path in clean_paths:
            filename = os.path.basename(clean_path)
            name_without_ext = os.path.splitext(filename)[0]
            
            # 查找匹配的噪声图像
            noisy_path = None
            for ext in image_extensions:
                possible_path = os.path.join(noisy_dir, name_without_ext + ext[1:])
                if os.path.exists(possible_path):
                    noisy_path = possible_path
                    break
            
            if noisy_path is None:
                # 尝试直接匹配文件名
                possible_path = os.path.join(noisy_dir, filename)
                if os.path.exists(possible_path):
                    noisy_path = possible_path
            
            if noisy_path is not None:
                matched_clean_paths.append(clean_path)
                matched_noisy_paths.append(noisy_path)
            else:
                print(f"警告: 找不到 {filename} 对应的噪声图像")
        
        if not matched_clean_paths:
            raise FileNotFoundError("没有找到任何匹配的图像对")
        
        print(f"成功匹配 {len(matched_clean_paths)} 对图像")
        return matched_clean_paths, matched_noisy_paths
    
    def _cache_images(self):
        """缓存图像到RAM"""
        print("缓存图像到RAM...")
        self.clean_images = []
        self.noisy_images = []
        
        for clean_path, noisy_path in tqdm(zip(self.clean_paths, self.noisy_paths),
                                          total=len(self.clean_paths)):
            # 加载并转换为tensor
            clean_img = Image.open(clean_path).convert('RGB')
            noisy_img = Image.open(noisy_path).convert('RGB')
            
            # 转换为tensor并缓存
            clean_tensor = transforms.ToTensor()(clean_img)
            noisy_tensor = transforms.ToTensor()(noisy_img)
            
            self.clean_images.append(clean_tensor)
            self.noisy_images.append(noisy_tensor)
        
        print(f"已缓存 {len(self.clean_images)} 对图像")
    
    def _precompute_crop_positions(self):
        """预计算随机裁剪位置（加速训练）"""
        print("预计算随机裁剪位置...")
        self.crop_positions = []
        
        for _ in range(len(self.clean_paths) * self.num_patches_per_image):
            img_idx = _ % len(self.clean_paths)
            
            if self.cache_in_ram:
                clean_tensor = self.clean_images[img_idx]
            else:
                clean_path = self.clean_paths[img_idx]
                clean_img = Image.open(clean_path).convert('RGB')
                clean_tensor = transforms.ToTensor()(clean_img)
            
            _, h, w = clean_tensor.shape
            
            # 如果图像太小，需要先放大
            if h < self.patch_size or w < self.patch_size:
                h, w = self.patch_size, self.patch_size
            
            # 随机裁剪位置
            top = np.random.randint(0, h - self.patch_size + 1) if h > self.patch_size else 0
            left = np.random.randint(0, w - self.patch_size + 1) if w > self.patch_size else 0
            
            self.crop_positions.append((img_idx, top, left))
    
    def __len__(self):
        if self.train:
            return len(self.clean_paths) * self.num_patches_per_image
        else:
            return len(self.clean_paths)
    
    def __getitem__(self, idx):
        if self.train:
            # 训练模式：使用预计算的裁剪位置
            img_idx, top, left = self.crop_positions[idx]
            
            # 获取图像
            if self.cache_in_ram:
                clean_tensor = self.clean_images[img_idx]
                noisy_tensor = self.noisy_images[img_idx]
            else:
                clean_path = self.clean_paths[img_idx]
                noisy_path = self.noisy_paths[img_idx]
                
                clean_img = Image.open(clean_path).convert('RGB')
                noisy_img = Image.open(noisy_path).convert('RGB')
                
                clean_tensor = transforms.ToTensor()(clean_img)
                noisy_tensor = transforms.ToTensor()(noisy_img)
            
            # 数据增强
            if self.transform:
                seed = torch.randint(0, 2**32, (1,)).item()
                
                torch.manual_seed(seed)
                clean_tensor = self.transform(clean_tensor)
                
                torch.manual_seed(seed)
                noisy_tensor = self.transform(noisy_tensor)
            
            # 裁剪patch
            _, h, w = clean_tensor.shape
            
            # 如果图像太小，先放大
            if h < self.patch_size or w < self.patch_size:
                clean_tensor = F.interpolate(
                    clean_tensor.unsqueeze(0), 
                    size=(self.patch_size, self.patch_size), 
                    mode='bilinear', 
                    align_corners=True
                ).squeeze(0)
                noisy_tensor = F.interpolate(
                    noisy_tensor.unsqueeze(0), 
                    size=(self.patch_size, self.patch_size), 
                    mode='bilinear', 
                    align_corners=True
                ).squeeze(0)
                h, w = self.patch_size, self.patch_size
            
            # 使用预计算的位置裁剪
            clean_patch = clean_tensor[:, top:top+self.patch_size, left:left+self.patch_size]
            noisy_patch = noisy_tensor[:, top:top+self.patch_size, left:left+self.patch_size]
            
            return noisy_patch, clean_patch, os.path.basename(self.clean_paths[img_idx])
        
        else:
            # 测试模式：使用完整图像
            if self.cache_in_ram:
                clean_tensor = self.clean_images[idx]
                noisy_tensor = self.noisy_images[idx]
            else:
                clean_path = self.clean_paths[idx]
                noisy_path = self.noisy_paths[idx]
                
                clean_img = Image.open(clean_path).convert('RGB')
                noisy_img = Image.open(noisy_path).convert('RGB')
                
                clean_tensor = transforms.ToTensor()(clean_img)
                noisy_tensor = transforms.ToTensor()(noisy_img)
            
            return noisy_tensor, clean_tensor, os.path.basename(self.clean_paths[idx])

class GPUOptimizedPathBasedDataset(Dataset):
    """基于路径列表的数据集 - GPU优化版本"""
    
    def __init__(self, clean_paths: List[str], noisy_paths: List[str], 
                 patch_size: int, transform: Optional[transforms.Compose], 
                 train: bool, num_patches_per_image: int = 1,
                 cache_in_ram: bool = True):
        self.clean_paths = clean_paths
        self.noisy_paths = noisy_paths
        self.patch_size = patch_size
        self.transform = transform
        self.train = train
        self.num_patches_per_image = num_patches_per_image
        self.cache_in_ram = cache_in_ram
        
        # 缓存图像
        if cache_in_ram:
            self._cache_images()
        
        # 预计算裁剪位置
        if train:
            self._precompute_crop_positions()
    
    def _cache_images(self):
        """缓存图像到RAM"""
        self.clean_images = []
        self.noisy_images = []
        
        for clean_path, noisy_path in tqdm(zip(self.clean_paths, self.noisy_paths),
                                          total=len(self.clean_paths)):
            clean_img = Image.open(clean_path).convert('RGB')
            noisy_img = Image.open(noisy_path).convert('RGB')
            
            clean_tensor = transforms.ToTensor()(clean_img)
            noisy_tensor = transforms.ToTensor()(noisy_img)
            
            self.clean_images.append(clean_tensor)
            self.noisy_images.append(noisy_tensor)
    
    def _precompute_crop_positions(self):
        """预计算随机裁剪位置"""
        self.crop_positions = []
        
        for _ in range(len(self.clean_paths) * self.num_patches_per_image):
            img_idx = _ % len(self.clean_paths)
            
            if self.cache_in_ram:
                clean_tensor = self.clean_images[img_idx]
            else:
                clean_path = self.clean_paths[img_idx]
                clean_img = Image.open(clean_path).convert('RGB')
                clean_tensor = transforms.ToTensor()(clean_img)
            
            _, h, w = clean_tensor.shape
            
            if h < self.patch_size or w < self.patch_size:
                h, w = self.patch_size, self.patch_size
            
            top = np.random.randint(0, h - self.patch_size + 1) if h > self.patch_size else 0
            left = np.random.randint(0, w - self.patch_size + 1) if w > self.patch_size else 0
            
            self.crop_positions.append((img_idx, top, left))
    
    def __len__(self):
        if self.train:
            return len(self.clean_paths) * self.num_patches_per_image
        else:
            return len(self.clean_paths)
    
    def __getitem__(self, idx):
        if self.train:
            img_idx, top, left = self.crop_positions[idx]
            
            if self.cache_in_ram:
                clean_tensor = self.clean_images[img_idx]
                noisy_tensor = self.noisy_images[img_idx]
            else:
                clean_path = self.clean_paths[img_idx]
                noisy_path = self.noisy_paths[img_idx]
                
                clean_img = Image.open(clean_path).convert('RGB')
                noisy_img = Image.open(noisy_path).convert('RGB')
                
                clean_tensor = transforms.ToTensor()(clean_img)
                noisy_tensor = transforms.ToTensor()(noisy_img)
            
            # 数据增强
            if self.transform:
                seed = torch.randint(0, 2**32, (1,)).item()
                
                torch.manual_seed(seed)
                clean_tensor = self.transform(clean_tensor)
                
                torch.manual_seed(seed)
                noisy_tensor = self.transform(noisy_tensor)
            
            # 处理小图像
            _, h, w = clean_tensor.shape
            
            if h < self.patch_size or w < self.patch_size:
                clean_tensor = F.interpolate(
                    clean_tensor.unsqueeze(0), 
                    size=(self.patch_size, self.patch_size), 
                    mode='bilinear', 
                    align_corners=True
                ).squeeze(0)
                noisy_tensor = F.interpolate(
                    noisy_tensor.unsqueeze(0), 
                    size=(self.patch_size, self.patch_size), 
                    mode='bilinear', 
                    align_corners=True
                ).squeeze(0)
                h, w = self.patch_size, self.patch_size
            
            # 裁剪
            clean_patch = clean_tensor[:, top:top+self.patch_size, left:left+self.patch_size]
            noisy_patch = noisy_tensor[:, top:top+self.patch_size, left:left+self.patch_size]
            
            return noisy_patch, clean_patch, os.path.basename(self.clean_paths[img_idx])
        
        else:
            if self.cache_in_ram:
                clean_tensor = self.clean_images[idx]
                noisy_tensor = self.noisy_images[idx]
            else:
                clean_path = self.clean_paths[idx]
                noisy_path = self.noisy_paths[idx]
                
                clean_img = Image.open(clean_path).convert('RGB')
                noisy_img = Image.open(noisy_path).convert('RGB')
                
                clean_tensor = transforms.ToTensor()(clean_img)
                noisy_tensor = transforms.ToTensor()(noisy_img)
            
            return noisy_tensor, clean_tensor, os.path.basename(self.clean_paths[idx])

class GPUAcceleratedRealNoiseDenoisingTrainer:
    """使用真实噪声数据的去噪训练器 - GPU加速版本"""
    
    def __init__(self, base_data_dir: str, patch_size: int = 128, 
                 batch_size: int = 16, lr: float = 1e-4, 
                 device: Optional[torch.device] = None, train_ratio: float = 0.8,
                 use_amp: bool = True, gradient_accumulation_steps: int = 1,
                 num_workers: int = 4, prefetch_factor: int = 2):
        
        self.base_data_dir = base_data_dir
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.lr = lr
        self.train_ratio = train_ratio
        self.use_amp = use_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        
        # 设备设置
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"使用设备: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU型号: {torch.cuda.get_device_name(0)}")
            print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # 根据GPU内存自动调整批处理大小
        if torch.cuda.is_available():
            self.batch_size = self._auto_adjust_batch_size()
        
        # 初始化模型 - 使用优化版本
        self.model = OptimizedGaussianNoiseUNet(use_amp=use_amp).to(self.device)
        
        # 如果有多GPU，使用DataParallel
        if torch.cuda.device_count() > 1:
            print(f"使用 {torch.cuda.device_count()} 个GPU")
            self.model = nn.DataParallel(self.model)
        
        # 损失函数和优化器
        self.criterion = nn.MSELoss()
        
        # 使用AdamW优化器（更好的泛化性能）
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, 
                                     weight_decay=1e-5, eps=1e-8)
        
        # 使用余弦退火学习率调度
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler() if use_amp and torch.cuda.is_available() else None
        
        # 加载数据
        self._setup_data()
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.psnr_values = []
        self.ssim_values = []
        self.training_times = []
        
        # 性能监控
        self.batch_times = []
        
        # 预热GPU
        self._warmup_gpu()
    
    def _auto_adjust_batch_size(self) -> int:
        """根据GPU内存自动调整批处理大小"""
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        # 经验法则：根据GPU内存设置批次大小
        if gpu_memory_gb >= 16:  # 16GB以上GPU
            batch_size = min(32, self.batch_size)
        elif gpu_memory_gb >= 8:  # 8-16GB GPU
            batch_size = min(16, self.batch_size)
        elif gpu_memory_gb >= 4:  # 4-8GB GPU
            batch_size = min(8, self.batch_size)
        else:  # 小于4GB GPU
            batch_size = min(4, self.batch_size)
        
        print(f"自动调整批处理大小: {batch_size} (基于{self.batch_size}GB GPU内存)")
        return batch_size
    
    def _warmup_gpu(self):
        """预热GPU"""
        if torch.cuda.is_available():
            print("预热GPU...")
            # 创建一个虚拟输入进行前向传播
            dummy_input = torch.randn(self.batch_size, 3, self.patch_size, 
                                     self.patch_size).to(self.device)
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    _ = self.model(dummy_input)
            
            torch.cuda.synchronize()
            print("GPU预热完成")
    
    def _setup_data(self):
        """设置训练和验证数据 - GPU优化版本"""
        
        # 数据增强变换
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ])
        
        # 检查数据集目录
        clean_dir = os.path.join(self.base_data_dir, "original")
        noisy25_dir = os.path.join(self.base_data_dir, "noisy25")
        
        if not os.path.exists(clean_dir):
            raise FileNotFoundError(f"找不到干净图像目录: {clean_dir}")
        if not os.path.exists(noisy25_dir):
            raise FileNotFoundError(f"找不到噪声图像目录: {noisy25_dir}")
        
        # 创建完整数据集
        full_dataset = GPUCachedCBSD68Dataset(
            base_data_dir=self.base_data_dir,
            patch_size=self.patch_size,
            transform=None,
            train=False,
            cache_in_ram=True,
            use_amp=self.use_amp
        )
        
        # 获取所有图像路径
        all_clean_paths = full_dataset.clean_paths
        all_noisy_paths = full_dataset.noisy_paths
        
        # 分割数据集
        split_idx = int(self.train_ratio * len(all_clean_paths))
        train_clean_paths = all_clean_paths[:split_idx]
        train_noisy_paths = all_noisy_paths[:split_idx]
        val_clean_paths = all_clean_paths[split_idx:]
        val_noisy_paths = all_noisy_paths[split_idx:]
        
        # 创建训练数据集
        self.train_dataset = GPUOptimizedPathBasedDataset(
            clean_paths=train_clean_paths,
            noisy_paths=train_noisy_paths,
            patch_size=self.patch_size,
            transform=train_transform,
            train=True,
            num_patches_per_image=20,
            cache_in_ram=True
        )
        
        # 创建验证数据集
        self.val_dataset = GPUOptimizedPathBasedDataset(
            clean_paths=val_clean_paths,
            noisy_paths=val_noisy_paths,
            patch_size=self.patch_size,
            transform=None,
            train=False,
            num_patches_per_image=1,
            cache_in_ram=True
        )
        
        # 计算合适的num_workers
        if self.num_workers > os.cpu_count():
            self.num_workers = max(1, os.cpu_count() - 1)
            print(f"调整num_workers为: {self.num_workers}")
        
        # 创建数据加载器 - 优化设置
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,  # 固定内存，加速数据传输
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=self.prefetch_factor,
            drop_last=True,
            pin_memory_device=str(self.device) if self.device.type == 'cuda' else ''
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=min(2, self.num_workers),
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
        print(f"训练样本: {len(self.train_dataset)}")
        print(f"验证样本: {len(self.val_dataset)}")
        print(f"使用噪声水平: σ=25 (来自noisy25目录)")
        print(f"数据加载器优化:")
        print(f"  num_workers: {self.num_workers}")
        print(f"  prefetch_factor: {self.prefetch_factor}")
        print(f"  pin_memory: True")
        print(f"训练优化:")
        print(f"  混合精度训练: {self.use_amp}")
        print(f"  梯度累积步数: {self.gradient_accumulation_steps}")
    
    def compute_metrics(self, denoised: torch.Tensor, clean: torch.Tensor) -> Tuple[float, float]:
        """计算PSNR和SSIM指标 - GPU优化版本"""
        # 确保在GPU上计算
        denoised = denoised.to(self.device)
        clean = clean.to(self.device)
        
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # PSNR
            mse = F.mse_loss(denoised, clean)
            if mse == 0:
                psnr = torch.tensor(float('inf'), device=self.device)
            else:
                psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            
            # SSIM (简化版本)
            C1 = 0.01 ** 2
            C2 = 0.03 ** 2
            
            mu_x = denoised.mean()
            mu_y = clean.mean()
            sigma_x = denoised.std()
            sigma_y = clean.std()
            sigma_xy = ((denoised - mu_x) * (clean - mu_y)).mean()
            
            ssim_numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
            ssim_denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x ** 2 + sigma_y ** 2 + C2)
            ssim = ssim_numerator / ssim_denominator
        
        return psnr.item(), ssim.item()
    
    def train_epoch(self, epoch: int) -> float:
        """训练一个epoch - GPU优化版本"""
        self.model.train()
        running_loss = 0.0
        batch_time_sum = 0.0
        num_batches = len(self.train_loader)
        
        start_epoch_time = time.time()
        
        pbar = tqdm(self.train_loader, desc=f"训练 Epoch {epoch+1}")
        for batch_idx, (noisy_imgs, clean_imgs, _) in enumerate(pbar):
            batch_start_time = time.time()
            
            # 异步数据传输到GPU
            noisy_imgs = noisy_imgs.to(self.device, non_blocking=True)
            clean_imgs = clean_imgs.to(self.device, non_blocking=True)
            
            # 前向传播（使用混合精度）
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(noisy_imgs)
                loss = self.criterion(outputs, clean_imgs)
                
                # 梯度累积
                loss = loss / self.gradient_accumulation_steps
            
            # 反向传播
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 梯度累积更新
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.scaler:
                    # 梯度裁剪（防止梯度爆炸）
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                self.optimizer.zero_grad(set_to_none=True)  # 更高效地清零梯度
            
            running_loss += loss.item() * self.gradient_accumulation_steps
            
            # 计算批处理时间
            batch_time = time.time() - batch_start_time
            batch_time_sum += batch_time
            self.batch_times.append(batch_time)
            
            # 更新进度条
            avg_batch_time = batch_time_sum / (batch_idx + 1)
            remaining_time = avg_batch_time * (num_batches - batch_idx - 1)
            
            pbar.set_postfix({
                'Loss': f'{loss.item() * self.gradient_accumulation_steps:.6f}',
                'Avg Loss': f'{running_loss/(batch_idx+1):.6f}',
                'Batch Time': f'{batch_time:.3f}s',
                'ETA': f'{remaining_time/60:.1f}m'
            })
        
        # 清理可能剩余的梯度
        if (num_batches % self.gradient_accumulation_steps) != 0:
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        
        avg_loss = running_loss / num_batches
        self.train_losses.append(avg_loss)
        
        epoch_time = time.time() - start_epoch_time
        self.training_times.append(epoch_time)
        
        print(f"Epoch {epoch+1} 训练完成: 平均损失 = {avg_loss:.6f}, 时间 = {epoch_time:.2f}s")
        
        # 清理中间变量以释放内存
        torch.cuda.empty_cache()
        
        return avg_loss
    
    def validate(self) -> Tuple[float, float, float]:
        """验证 - GPU优化版本"""
        self.model.eval()
        running_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        
        val_start_time = time.time()
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="验证")
            for noisy_imgs, clean_imgs, _ in pbar:
                # 异步数据传输
                noisy_imgs = noisy_imgs.to(self.device, non_blocking=True)
                clean_imgs = clean_imgs.to(self.device, non_blocking=True)
                
                # 使用混合精度进行推理
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(noisy_imgs)
                    loss = self.criterion(outputs, clean_imgs)
                
                # 计算指标
                psnr, ssim = self.compute_metrics(outputs, clean_imgs)
                
                running_loss += loss.item()
                total_psnr += psnr
                total_ssim += ssim
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.6f}',
                    'PSNR': f'{psnr:.2f}',
                    'SSIM': f'{ssim:.4f}'
                })
        
        avg_loss = running_loss / len(self.val_loader)
        avg_psnr = total_psnr / len(self.val_loader)
        avg_ssim = total_ssim / len(self.val_loader)
        
        self.val_losses.append(avg_loss)
        self.psnr_values.append(avg_psnr)
        self.ssim_values.append(avg_ssim)
        
        val_time = time.time() - val_start_time
        print(f"验证完成: 时间 = {val_time:.2f}s, PSNR = {avg_psnr:.2f} dB, SSIM = {avg_ssim:.4f}")
        
        return avg_loss, avg_psnr, avg_ssim
    
    def train(self, epochs: int = 100, save_every: int = 10):
        """完整训练流程 - GPU优化版本"""
        print("=" * 80)
        print("开始训练固定高斯噪声UNet去噪器 - GPU加速版本")
        print("=" * 80)
        print(f"使用真实噪声数据: noisy25")
        print(f"训练周期: {epochs}")
        print(f"批次大小: {self.batch_size}")
        print(f"梯度累积步数: {self.gradient_accumulation_steps}")
        print(f"有效批次大小: {self.batch_size * self.gradient_accumulation_steps}")
        print(f"初始学习率: {self.lr}")
        
        best_val_loss = float('inf')
        best_psnr = 0.0
        
        total_start_time = time.time()
        
        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"{'='*60}")
            
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss, psnr, ssim = self.validate()
            
            # 学习率调度
            self.scheduler.step()
            
            print(f"\n训练损失: {train_loss:.6f}")
            print(f"验证损失: {val_loss:.6f}")
            print(f"验证PSNR: {psnr:.2f} dB")
            print(f"验证SSIM: {ssim:.4f}")
            print(f"当前学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_psnr = psnr
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'psnr': psnr,
                    'ssim': ssim,
                    'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
                    'batch_size': self.batch_size,
                    'lr': self.lr,
                }
                
                torch.save(checkpoint, 'best_denoising_unet_real_noise_gpu.pth')
                print(f"💾 保存最佳模型! PSNR: {psnr:.2f} dB")
            
            # 定期保存检查点
            if (epoch + 1) % save_every == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'psnr': psnr,
                    'ssim': ssim,
                    'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
                }
                
                torch.save(checkpoint, f'checkpoint_epoch_{epoch+1}_gpu.pth')
                print(f"💾 保存检查点到 checkpoint_epoch_{epoch+1}_gpu.pth")
        
        total_time = time.time() - total_start_time
        avg_epoch_time = np.mean(self.training_times) if self.training_times else 0
        
        print(f"\n{'='*80}")
        print(f"训练完成!")
        print(f"{'='*80}")
        print(f"总训练时间: {total_time/60:.2f} 分钟")
        print(f"平均每epoch时间: {avg_epoch_time:.2f} 秒")
        print(f"最佳PSNR: {best_psnr:.2f} dB")
        print(f"最佳验证损失: {best_val_loss:.6f}")
        
        # 显示性能统计
        if self.batch_times:
            avg_batch_time = np.mean(self.batch_times)
            std_batch_time = np.std(self.batch_times)
            print(f"\n性能统计:")
            print(f"平均批处理时间: {avg_batch_time:.3f} ± {std_batch_time:.3f} 秒")
            print(f"平均每秒批处理数: {1/avg_batch_time:.2f}")
            print(f"平均每秒样本数: {self.batch_size/avg_batch_time:.2f}")
            
            # 计算GPU利用率
            if torch.cuda.is_available():
                max_memory = torch.cuda.max_memory_allocated() / 1e9
                print(f"最大GPU内存使用: {max_memory:.2f} GB")
        
        # 保存最终模型
        final_checkpoint = {
            'epoch': epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'psnr': psnr,
            'ssim': ssim,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
        }
        
        torch.save(final_checkpoint, 'final_denoising_unet_real_noise_gpu.pth')
        print(f"💾 保存最终模型")
    
    def plot_training_history(self):
        """绘制训练历史"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        ax1.plot(self.train_losses, label='训练损失', linewidth=2, alpha=0.8)
        ax1.plot(self.val_losses, label='验证损失', linewidth=2, alpha=0.8)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('损失', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.set_title('训练和验证损失曲线', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # PSNR曲线
        ax2.plot(self.psnr_values, color='green', linewidth=2, marker='o', markersize=3)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('PSNR (dB)', fontsize=12)
        ax2.set_title('验证集PSNR', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # SSIM曲线
        ax3.plot(self.ssim_values, color='red', linewidth=2, marker='s', markersize=3)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('SSIM', fontsize=12)
        ax3.set_title('验证集SSIM', fontsize=14)
        ax3.grid(True, alpha=0.3)
        
        # 训练时间曲线
        if self.training_times:
            ax4.plot(self.training_times, color='purple', linewidth=2, marker='^', markersize=3)
            ax4.set_xlabel('Epoch', fontsize=12)
            ax4.set_ylabel('时间 (秒)', fontsize=12)
            ax4.set_title('每epoch训练时间', fontsize=14)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history_real_noise_gpu.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 批处理时间分布图
        if self.batch_times:
            fig2, ax = plt.subplots(figsize=(10, 6))
            ax.hist(self.batch_times, bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax.set_xlabel('批处理时间 (秒)', fontsize=12)
            ax.set_ylabel('频率', fontsize=12)
            ax.set_title('批处理时间分布', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            # 添加统计信息
            stats_text = f"平均值: {np.mean(self.batch_times):.3f}s\n标准差: {np.std(self.batch_times):.3f}s\n最小值: {np.min(self.batch_times):.3f}s\n最大值: {np.max(self.batch_times):.3f}s"
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            plt.savefig('batch_time_distribution_gpu.png', dpi=300, bbox_inches='tight')
            plt.show()


# if __name__ == "__main__":
#     # 设置CUDA设备
#     if torch.cuda.is_available():
#         # 可选：设置特定GPU
#         torch.cuda.set_device(0)  # 使用第一个GPU
        
#     # CBSD68数据集根目录
#     base_data_dir = "C:/Users/likaituo/Desktop/神经网络/CBSD68-dataset-master/CBSD68"
    
#     # 检查数据集是否存在
#     if not os.path.exists(base_data_dir):
#         raise FileNotFoundError(f"找不到CBSD68数据集目录: {base_data_dir}")
    
#     if not os.path.exists(os.path.join(base_data_dir, "original")):
#         raise FileNotFoundError(f"在 {base_data_dir} 中找不到original目录")
    
#     if not os.path.exists(os.path.join(base_data_dir, "noisy25")):
#         raise FileNotFoundError(f"在 {base_data_dir} 中找不到noisy25目录")
    
#     # 检查GPU是否可用
#     if torch.cuda.is_available():
#         print(f"✅ 检测到GPU: {torch.cuda.get_device_name(0)}")
#         print(f"✅ CUDA版本: {torch.version.cuda}")
#         print(f"✅ GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
#         # 设置GPU内存优化
#         torch.cuda.empty_cache()  # 清空缓存
#         use_amp = True  # 启用混合精度
#     else:
#         print("⚠️  警告: 未检测到GPU，将使用CPU训练（速度会慢很多）")
#         use_amp = False  # CPU上禁用混合精度
    
#     # 创建训练器 - 使用GPU加速版本
#     try:
#         trainer = GPUAcceleratedRealNoiseDenoisingTrainer(
#             base_data_dir=base_data_dir,
#             patch_size=128,
#             batch_size=16,  # 初始批次大小，会自动调整
#             lr=1e-4,
#             use_amp=use_amp,  # 自动根据GPU可用性设置
#             gradient_accumulation_steps=2,  # 梯度累积
#             num_workers=4,  # 数据加载workers数量
#             prefetch_factor=2  # 预取因子
#         )
        
#         # 训练模型
#         trainer.train(epochs=50, save_every=10)
        
#         # 绘制训练历史
#         trainer.plot_training_history()
        
#     except RuntimeError as e:
#         print(f"❌ 运行时错误: {e}")
#         if "CUDA out of memory" in str(e):
#             print("💡 提示: GPU内存不足，尝试减小批次大小或使用梯度累积")
#     except Exception as e:
#         print(f"❌ 错误: {e}")
#     finally:
#         # 清理GPU缓存
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#             print("✅ GPU缓存已清理")
# 使用示例
if __name__ == "__main__":
    # 设置CUDA设备
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    
    # CBSD68数据集根目录
    base_data_dir = "C:/Users/likaituo/Desktop/神经网络/CBSD68-dataset-master/CBSD68"
    
    # 检查数据集是否存在
    if not os.path.exists(base_data_dir):
        raise FileNotFoundError(f"找不到CBSD68数据集目录: {base_data_dir}")
    
    # 检查GPU是否可用
    if torch.cuda.is_available():
        print(f"✅ 检测到GPU: {torch.cuda.get_device_name(0)}")
        use_amp = True
    else:
        print("⚠️  警告: 未检测到GPU，将使用CPU训练")
        use_amp = False
    
    # 创建训练器
    try:
        trainer = GPUAcceleratedRealNoiseDenoisingTrainer(
            base_data_dir=base_data_dir,
            patch_size=128,
            batch_size=16,
            lr=1e-4,
            use_amp=use_amp,
            gradient_accumulation_steps=2,
            num_workers=4,
            prefetch_factor=2
        )
        
        # 训练模型
        trainer.train(epochs=50, save_every=10)
        
        # ============ 添加以下代码 ============
        print("\n" + "="*60)
        print("📊 绘制损失演变曲线...")
        print("="*60)
        
        # 方法1：调用原有的plot_training_history（可能有字体问题）
        try:
            trainer.plot_training_history()
        except Exception as e:
            print(f"⚠️  原有绘图方法出错: {e}")
        
        # 方法2：调用修复后的绘图函数
        plot_loss_evolution_fixed(trainer)
        # ============ 添加结束 ============
        
    except RuntimeError as e:
        print(f"❌ 运行时错误: {e}")
        if "CUDA out of memory" in str(e):
            print("💡 提示: GPU内存不足，尝试减小批次大小或使用梯度累积")
    except Exception as e:
        print(f"❌ 错误: {e}")
    finally:
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("✅ GPU缓存已清理")