# 频域调制模块 (FDM) 集成总结

## 问题诊断与修复

### 问题描述
训练启动后遇到错误：
```
IndexError: tuple index out of range
```

### 根本原因
**SMP版本兼容性问题**：

1. 原始SLAug使用 `segmentation-models-pytorch==0.2.1`
2. FDM wrapper初始实现针对 SMP 0.5.0 (新版API)
3. 两个版本的decoder接口不同：
   - **SMP 0.2.1 (旧版)**: `decoder.forward(self, *features)` - 使用 `*args` 解包
   - **SMP 0.3+ (新版)**: `decoder.forward(self, features: List)` - 接收list参数

### 修复方案

在 `util/unet_fdm_wrapper.py` 的 `__init__` 方法中自动检测decoder API版本：

```python
# 检测 SMP decoder API 版本
import inspect
decoder_sig = inspect.signature(self.unet.decoder.forward)
has_varargs = any(
    p.kind == inspect.Parameter.VAR_POSITIONAL
    for p in decoder_sig.parameters.values()
)
self._use_new_decoder_api = not has_varargs
```

在 `forward` 方法中根据检测结果选择正确的调用方式：

```python
if self._use_new_decoder_api:
    # 新版 API: 传递 list
    if isinstance(features, tuple):
        features = list(features)
    decoder_output = self.unet.decoder(features)
else:
    # 旧版 API: 解包参数
    decoder_output = self.unet.decoder(*features)
```

### 技术细节

**关键区别**：
- 旧版API使用 `*features` (VAR_POSITIONAL)，参数被解包为多个独立tensor
- 新版API使用 `features: List`，接收单个list对象

**检测方法**：
- 使用 `inspect.signature()` 获取方法签名
- 检查参数的 `kind` 属性是否为 `VAR_POSITIONAL`
- 在 `__init__` 时检测一次，避免每次forward都执行inspect

## 已实现文件

### 1. 核心模块
- **`util/freq_domain_mod.py`**: FDM核心实现
  - `FreqDomainModLayer`: 单层频域调制
    - 径向频带分解 (B个同心环带)
    - PAC计算：幅度扰动对相位的影响
    - SBP计算：相位分布的KL散度稳定性
    - SRI计算：α·PAC + (1-α)·SBP
    - 可学习频带权重（从SRI初始化）
  - `FreqDomainModulation`: 多层FDM包装器

- **`util/unet_fdm_wrapper.py`**: UNet+FDM集成
  - 自动检测SMP版本（0.2.x vs 0.3+）
  - 在编码器stages 2和3插入FDM（4×和8×下采样）
  - 支持enable/disable切换
  - 支持freeze/unfreeze FDM权重

### 2. 配置文件
- **`configs/efficientUnet_FDM_CHAOS_to_SABSCT.yaml`**: FDM启用
- **`configs/efficientUnet_FDM_disabled_CHAOS_to_SABSCT.yaml`**: FDM禁用

### 3. 测试脚本
- **`test_fdm.py`**: 6个综合测试（全部通过✓）
- **`test_config_instantiation.py`**: 配置实例化测试

## 使用方法

### 训练启用FDM
```bash
python main.py -b configs/efficientUnet_FDM_CHAOS_to_SABSCT.yaml
```

### 训练禁用FDM（基线）
```bash
python main.py -b configs/efficientUnet_FDM_disabled_CHAOS_to_SABSCT.yaml
```

### FDM参数配置
```yaml
model:
  target: util.unet_fdm_wrapper.UNetWithFDM
  params:
    encoder_name: efficientnet-b2
    encoder_weights: null
    in_channels: 1
    classes: 5
    activation: null
    # FDM参数
    fdm_enabled: true              # 启用/禁用开关
    fdm_stages: [2, 3]             # 应用FDM的编码器阶段（4×, 8×下采样）
    fdm_num_bands: 8               # 频带数量
    fdm_alpha: 0.5                 # PAC-SBP融合权重
    fdm_delta: 0.2                 # 扰动强度
    fdm_eps: 1.0e-6                # 数值稳定性常数
    fdm_init_from_stats: true      # 从SRI统计初始化权重
    fdm_update_stats_interval: 100 # 统计更新间隔（迭代数）
```

## 性能优化建议

### 如果训练速度慢
1. **增大统计更新间隔**：
   ```yaml
   fdm_update_stats_interval: 500  # 默认100，可增至500-1000
   ```
   - PAC/SBP计算包含IFFT+FFT+直方图，较耗时
   - 更新间隔越大，开销越小

2. **减少频带数量**：
   ```yaml
   fdm_num_bands: 4  # 默认8，可减至4-6
   ```
   - 减少频带可降低计算量

3. **减小batch size**：
   - 如果GPU内存不足，减小batch_size

### 如果遇到数值问题
1. **检查扰动强度**：
   ```yaml
   fdm_delta: 0.1  # 默认0.2，可减小
   ```

2. **增大数值稳定性常数**：
   ```yaml
   fdm_eps: 1.0e-5  # 默认1e-6
   ```

## 兼容性

### 支持的SMP版本
- ✅ segmentation-models-pytorch 0.2.1 (SLAug原版)
- ✅ segmentation-models-pytorch 0.3.x
- ✅ segmentation-models-pytorch 0.5.0

### 自动检测机制
模块会在初始化时自动检测SMP版本并选择正确的decoder调用方式，无需手动配置。

## 测试验证

所有6项测试通过：
1. ✓ FreqDomainModLayer基本功能
2. ✓ FreqDomainModulation多层模块
3. ✓ UNetWithFDM包装器模型
4. ✓ FDM启用/禁用切换
5. ✓ FDM权重冻结/解冻
6. ✓ 梯度流动

## 关键特性

- **自适应频域调制**：基于PAC/SBP/SRI指标
- **即插即用**：通过YAML配置启用/禁用
- **版本兼容**：自动适配SMP 0.2.x和0.3+
- **可学习权重**：频带权重可训练
- **统计驱动**：权重从结构相关性初始化
- **高效缓存**：频带掩码只构建一次

## 数学原理

### 相位-幅度耦合度 (PAC)
对频带 b 施加幅度扰动：
```
A'_{l,b}(u,v) = A_l(u,v) * (1 + δ), (u,v) ∈ Ω_b
```

通过IFFT+FFT传播扰动，计算相位偏移：
```
Δφ_{l,b} = Mean_{(u,v)∈Ω_b} |φ'_{l,b}(u,v) - φ_l(u,v)|
```

归一化得到PAC：
```
PAC_{l,b} = Δφ_{l,b} / (max_b Δφ_{l,b} + ε)
```

### 结构频带持续性 (SBP)
估计扰动前后相位分布的KL散度：
```
SBP_{l,b} = exp(- D_KL(P(φ_{l,b}) || P(φ'_{l,b})))
```

### 结构相关性指标 (SRI)
融合PAC和SBP：
```
SRI_{l,b} = α * PAC_{l,b} + (1 - α) * SBP_{l,b}
```

### 自适应幅度调制
可学习权重初始化：
```
w_{l,b}^{init} = SRI_{l,b} / Σ_{b'} SRI_{l,b'}
```

前向调制：
```
A'_l(u,v) = A_l(u,v) * w_{l,b}, for (u,v) ∈ Ω_b
```

## 文件路径映射

```
FBD/
├── util/
│   ├── freq_domain_mod.py      # FDM核心模块
│   └── unet_fdm_wrapper.py     # UNet集成包装器
├── configs/
│   ├── efficientUnet_FDM_CHAOS_to_SABSCT.yaml          # FDM启用配置
│   └── efficientUnet_FDM_disabled_CHAOS_to_SABSCT.yaml # FDM禁用配置
├── test_fdm.py                 # 综合测试
├── test_config_instantiation.py # 配置测试
└── FDM_INTEGRATION_SUMMARY.md  # 本文档
```

## 下一步

1. ✅ 核心FDM模块实现
2. ✅ UNet集成
3. ✅ 配置文件
4. ✅ SMP版本兼容性修复
5. ✅ 测试验证
6. 🔄 训练验证和性能评估

现在可以开始完整训练了！
