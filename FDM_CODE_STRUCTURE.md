# FDM 核心代码模块说明

## 🎯 核心模块概览

```
FBD/
├── util/
│   ├── freq_domain_mod.py         ⭐⭐⭐ FDM核心算法
│   └── unet_fdm_wrapper.py        ⭐⭐⭐ UNet集成包装器
├── configs/
│   ├── efficientUnet_FDM_*.yaml   ⭐⭐  FDM配置文件
│   └── efficientUnet_FDM_zero_workers_*.yaml  ← 当前使用
└── main.py                        ⭐    训练主程序
```

---

## 1️⃣ FDM核心算法模块 ⭐⭐⭐

**文件**: `util/freq_domain_mod.py` (约300行)

### 核心类

#### 1.1 `FreqDomainModLayer` - 单层频域调制
```python
class FreqDomainModLayer(nn.Module):
    """
    单个特征层的频域调制

    核心功能：
    1. FFT变换到频域
    2. 分离幅度谱和相位谱
    3. 频带划分（径向同心环）
    4. 计算PAC/SBP/SRI统计（可选）
    5. 自适应幅度调制
    6. IFFT回到空间域
    """

    def __init__(self, in_channels, num_bands=8, ...):
        # 可学习的频带权重
        self.freq_weights = nn.Parameter(torch.ones(num_bands) / num_bands)

    def forward(self, x):
        # 1. FFT到频域
        X = torch.fft.fft2(x)
        A = torch.abs(X)      # 幅度谱
        phase = torch.angle(X) # 相位谱

        # 2. 频带调制
        A_mod = 调制幅度(A, self.freq_weights)

        # 3. IFFT回空间域
        x_mod = torch.fft.ifft2(重构(A_mod, phase))
        return x_mod
```

**关键方法**：
- `_build_band_masks()`: 构建径向频带掩码（同心环）
- `_estimate_band_stats()`: 计算PAC/SBP/SRI统计
- `forward()`: 前向传播（FFT→调制→IFFT）

#### 1.2 `FreqDomainModulation` - 多层包装器
```python
class FreqDomainModulation(nn.Module):
    """
    管理多个FreqDomainModLayer
    通常用于encoder的多个阶段（如stage 2和3）
    """

    def __init__(self, channels_list=[24, 48], ...):
        self.fdm_layers = nn.ModuleList([
            FreqDomainModLayer(c, ...) for c in channels_list
        ])
```

---

## 2️⃣ UNet集成包装器 ⭐⭐⭐

**文件**: `util/unet_fdm_wrapper.py` (约200行)

### 核心类

#### `UNetWithFDM` - FDM增强的UNet
```python
class UNetWithFDM(nn.Module):
    """
    将FDM模块插入到segmentation_models_pytorch.Unet中

    架构：
    Input → Encoder → [FDM on stage 2,3] → Decoder → Output
    """

    def __init__(self, fdm_enabled=True, fdm_stages=[2,3], ...):
        # 基础UNet
        self.unet = smp.Unet(encoder_name='efficientnet-b2', ...)

        # FDM模块（插入到指定stage）
        if fdm_enabled:
            self.fdm = FreqDomainModulation(
                channels_list=[self.encoder_channels[s] for s in fdm_stages]
            )

    def forward(self, x):
        # 编码
        features = self.unet.encoder(x)  # [f0, f1, f2, f3, f4, f5]

        # FDM调制（stage 2和3）
        if self.fdm_enabled:
            features[2] = self.fdm.fdm_layers[0](features[2])
            features[3] = self.fdm.fdm_layers[1](features[3])

        # 解码
        output = self.unet.decoder(features)
        return self.unet.segmentation_head(output)
```

**关键功能**：
- ✅ SMP版本自动检测（0.2.x vs 0.3+）
- ✅ 动态enable/disable FDM
- ✅ 提供optimizer参数分组
- ✅ 统计信息获取

---

## 3️⃣ 配置文件 ⭐⭐

**当前使用**: `configs/efficientUnet_FDM_zero_workers_CHAOS_to_SABSCT.yaml`

```yaml
model:
  target: util.unet_fdm_wrapper.UNetWithFDM  # ← 使用FDM包装器
  params:
    encoder_name: efficientnet-b2
    in_channels: 1
    classes: 5

    # FDM参数 ⭐
    fdm_enabled: true              # 启用FDM
    fdm_stages: [2, 3]             # 在stage 2,3插入
    fdm_num_bands: 4               # 频带数量
    fdm_alpha: 0.5                 # PAC-SBP权重
    fdm_init_from_stats: false     # 禁用统计（加速）

data:
  params:
    num_workers: 0  # ← 当前优化：单进程加载
```

---

## 4️⃣ 数据流程

### 完整训练流程

```
main.py:
├── 加载配置 (yaml)
├── 实例化模型 (UNetWithFDM)
│   ├── 创建UNet encoder/decoder
│   └── 创建FDM模块（2层）
├── 创建DataLoader
│   └── AbdominalDataset
│       └── Location Scale Augmentation ⚠️ (慢！)
└── train_one_epoch_SBF()
    └── 每个batch:
        ├── data = dataloader.next()  ← 18秒！
        ├── output = model(data)      ← 2秒
        │   ├── encoder(data)
        │   ├── FDM调制 ⭐            ← 0.5秒
        │   └── decoder
        ├── loss计算                  ← 0.3秒
        └── backward + optim          ← 0.5秒
```

---

## 5️⃣ FDM核心算法伪代码

### 前向传播
```python
def fdm_forward(x):
    # x: [N, C, H, W] 空间域特征

    # 1. 频域变换
    X = FFT2D(x)                    # [N, C, H, W] 复数
    A = |X|                         # 幅度谱
    φ = angle(X)                    # 相位谱

    # 2. 频带划分（径向）
    for b in range(num_bands):
        Ω_b = radial_band(b)        # 同心环掩码

    # 3. 幅度调制
    A' = Σ_b (w_b × mask_b × A)    # w_b是可学习权重

    # 4. 重构
    X' = A' × exp(j×φ)              # 保持相位不变

    # 5. 逆变换
    x' = IFFT2D(X')                 # 回到空间域

    return real(x')
```

### PAC/SBP统计（可选，当前禁用）
```python
def compute_PAC_SBP(A, φ, band_masks):
    for each band b:
        # 扰动幅度
        A_pert = A × (1 + δ × mask_b)

        # 传播扰动
        φ_pert = angle(FFT(IFFT(A_pert × exp(jφ))))

        # 计算PAC
        PAC_b = mean(|φ_pert - φ|) in Ω_b

        # 计算SBP
        SBP_b = exp(-KL(P(φ), P(φ_pert)))

    # 融合得到SRI
    SRI = α×PAC + (1-α)×SBP

    # 用SRI初始化权重
    w_init = SRI / sum(SRI)
```

---

## 6️⃣ 当前性能瓶颈 ⚠️

### 时间分解（20秒/iter）
```
总时间: 20.38秒
├── DataLoader: 18.22秒 (89%) ⚠️ 主要瓶颈
│   └── Location Scale Aug: ~15秒 (nTimes=10000)
├── FDM模块: ~0.5秒 (2%)
├── UNet其他: ~1.5秒 (8%)
└── 损失+优化: ~0.16秒 (1%)
```

### 仍然很慢的原因

**Location Scale Augmentation 仍然是瓶颈！**

即使改为 `nTimes=10000`，在 `OMP_NUM_THREADS=1` 下：
- Bernstein多项式计算：纯串行
- `np.interp`：单线程
- `np.percentile`：单线程

**解决方案**：
1. 进一步减少到 `nTimes=1000`
2. 或完全禁用LSA（`location_scale: false`）

---

## 7️⃣ FDM相关文件清单

### 核心代码（必须）
```
util/freq_domain_mod.py          - FDM算法实现
util/unet_fdm_wrapper.py         - UNet集成
```

### 配置文件（选择一个）
```
configs/efficientUnet_FDM_zero_workers_*.yaml        - 当前使用（num_workers=0）
configs/efficientUnet_FDM_optimized_*.yaml          - 优化版（num_workers=2）
configs/efficientUnet_FDM_disabled_*.yaml           - 禁用FDM
```

### 文档
```
FDM_INTEGRATION_SUMMARY.md       - FDM集成总结
FDM_CODE_STRUCTURE.md            - 本文档
PROBLEM_DESCRIPTION_FOR_AI.md    - 问题完整描述
```

### 优化相关
```
SPEED_OPTIMIZATION.md            - 速度优化指南
THREADING_FIX.md                 - 线程问题修复
PATCH_LSA_SPEED.md              - LSA加速patch
```

---

## 8️⃣ 关键参数说明

### FDM参数
```python
fdm_enabled: bool               # 启用/禁用FDM
fdm_stages: [2, 3]             # 应用FDM的encoder阶段
fdm_num_bands: 4               # 频带数量（越少越快）
fdm_alpha: 0.5                 # PAC和SBP的融合权重
fdm_delta: 0.2                 # 扰动强度（用于统计计算）
fdm_init_from_stats: false     # 是否计算PAC/SBP（禁用更快）
fdm_update_stats_interval: 999999  # 统计更新频率
```

### 性能权衡
| 参数 | 速度 | 效果 |
|------|------|------|
| fdm_num_bands=8 | 慢 | 精细调制 |
| fdm_num_bands=4 | ✅ 快 | 足够好 |
| fdm_init_from_stats=true | 极慢 | 理论最优 |
| fdm_init_from_stats=false | ✅ 快 | 权重可学习 |

---

## 9️⃣ 下一步优化建议

### 立即尝试（高优先级）
```bash
# 1. 进一步减少LSA复杂度
sed -i 's/nTimes=10000/nTimes=1000/g' dataloaders/location_scale_augmentation.py

# 2. 或完全禁用LSA测试基准
# 创建配置：location_scale: false
```

### 中期优化
1. 使用Numba加速LSA
2. 预计算并缓存LSA结果
3. 考虑GPU数据增强（Kornia）

### 长期优化
1. 混合精度训练（AMP）
2. 梯度累积（虚拟更大batch）
3. 分布式训练

---

## 🎯 总结

**FDM核心就是2个文件**：
1. ⭐⭐⭐ `util/freq_domain_mod.py` - 算法实现
2. ⭐⭐⭐ `util/unet_fdm_wrapper.py` - UNet集成

**当前瓶颈不在FDM（只占2%），而在数据加载（占89%）**

**根本原因**：Location Scale Augmentation太复杂 + OMP_NUM_THREADS=1限制
