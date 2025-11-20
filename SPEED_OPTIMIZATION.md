# FDM训练速度优化指南

## 问题分析

根据你的训练log：
```
time: 158.8870  data: 138.0698
```

**关键发现**：
- 总训练时间：158.9秒/iter
- 数据加载时间：138.1秒（**占87%**！）
- 模型计算时间：~20.8秒（占13%）

**结论：主要瓶颈是DataLoader，而不是FDM模块！**

---

## 已修复的问题

### 1. 复数转实数警告
**问题**：
```
Warning: Casting complex values to real discards the imaginary part
```

**原因**：
在 `freq_domain_mod.py` 中使用 `.real` 属性会触发警告

**修复**：
```python
# 修复前
x_mod = x_mod.real  # 触发警告

# 修复后
x_mod = torch.real(x_mod)  # 使用torch.real()函数
```

---

## 速度优化方案

### 方案1：快速配置（推荐）

使用 `configs/efficientUnet_FDM_fast_CHAOS_to_SABSCT.yaml`

**关键优化**：
```yaml
fdm_num_bands: 4              # 从8减到4（减少50%计算量）
fdm_update_stats_interval: 1000  # 从100增到1000（减少10倍统计开销）
num_workers: 4                # 从8减到4（避免死锁）
```

**预期效果**：
- FDM开销从 ~5秒 降至 ~2秒
- 统计更新频率降低10倍

**训练命令**：
```bash
python main.py -b configs/efficientUnet_FDM_fast_CHAOS_to_SABSCT.yaml
```

---

### 方案2：极速配置（最快）

使用 `configs/efficientUnet_FDM_ultra_fast_CHAOS_to_SABSCT.yaml`

**关键优化**：
```yaml
fdm_num_bands: 4              # 使用4个频带
fdm_init_from_stats: false    # 完全禁用PAC/SBP统计计算
fdm_update_stats_interval: 999999  # 永不更新
num_workers: 4
```

**说明**：
- **不计算PAC/SBP/SRI**，直接用均匀权重初始化
- FDM权重仍然可学习，通过梯度下降自动优化
- 几乎零统计开销

**权衡**：
- ✅ 最快速度
- ❌ 失去基于结构相关性的初始化
- ⚖️ 可能略微影响收敛速度，但最终性能可能相近

**训练命令**：
```bash
python main.py -b configs/efficientUnet_FDM_ultra_fast_CHAOS_to_SABSCT.yaml
```

---

### 方案3：优化DataLoader（最重要！）

**DataLoader占87%时间，是真正的瓶颈！**

#### 3.1 调整num_workers

当前配置 `num_workers=8` 可能过多，尝试：

```yaml
data:
  params:
    num_workers: 2  # 或 4
```

**原因**：
- 过多worker导致进程切换开销
- 可能触发GIL锁竞争
- 数据预处理如果有Python代码，多进程效率低

#### 3.2 检查数据增强

Location Scale Augmentation 可能很慢，检查：

```python
# 在 dataloaders/location_scale_augmentation.py 中
# 看是否有复杂的CPU计算
```

**临时测试**：禁用LSA看速度提升
```yaml
train:
  params:
    location_scale: false  # 临时禁用测试
```

#### 3.3 使用persistent_workers

在 `main.py` 中已经使用：
```python
persistent_workers=True  # ✓ 已启用
```

#### 3.4 预加载到GPU

如果数据集不大，考虑预加载：
```python
# 修改train_loader创建
train_loader = DataLoader(..., pin_memory=True)  # ✓ 已启用
```

---

## 性能测试工具

运行诊断脚本：
```bash
python profile_training_speed.py
```

**输出**：
- FDM不同配置的实际开销
- 前向/反向传播时间分解
- DataLoader加载速度测试

---

## 配置对比

| 配置 | fdm_num_bands | fdm_update_interval | fdm_init_from_stats | 预期FDM开销 |
|------|---------------|---------------------|---------------------|------------|
| 原始 | 8 | 100 | true | ~5-8秒 |
| 快速 | 4 | 1000 | true | ~2-3秒 |
| 极速 | 4 | 999999 | false | ~1-2秒 |
| 禁用 | - | - | - | 0秒 |

**注意**：即使是原始配置，FDM只占总时间的 ~5/159 = 3%，主要瓶颈仍是DataLoader！

---

## 推荐优化顺序

1. **首先**：优化DataLoader（占87%！）
   ```yaml
   num_workers: 2  # 减少worker数量
   ```

2. **其次**：使用快速FDM配置
   ```bash
   python main.py -b configs/efficientUnet_FDM_fast_CHAOS_to_SABSCT.yaml
   ```

3. **如果还慢**：临时禁用Location Scale Augmentation测试
   ```yaml
   location_scale: false
   ```

4. **极端情况**：使用极速配置
   ```bash
   python main.py -b configs/efficientUnet_FDM_ultra_fast_CHAOS_to_SABSCT.yaml
   ```

---

## 预期加速效果

### 假设优化前：
```
总时间: 158.9秒
├── DataLoader: 138.1秒 (87%)
└── 模型计算: 20.8秒 (13%)
    ├── FDM: ~5秒
    └── 其他: ~15秒
```

### 优化后（快速配置 + num_workers=2）：
```
总时间: ~50秒  (↓ 109秒)
├── DataLoader: ~30秒 (↓ 108秒, 假设减少4倍)
└── 模型计算: ~20秒
    ├── FDM: ~2秒 (↓ 3秒)
    └── 其他: ~15秒
```

**预期加速比：3倍+**

### 极速配置（禁用PAC/SBP统计）：
```
总时间: ~48秒
├── DataLoader: ~30秒
└── 模型计算: ~18秒
    ├── FDM: ~1秒 (↓ 4秒)
    └── 其他: ~15秒
```

**预期加速比：3.3倍**

---

## 快速测试命令

```bash
# 1. 原始配置（慢）
python main.py -b configs/efficientUnet_FDM_CHAOS_to_SABSCT.yaml

# 2. 快速配置（推荐）
python main.py -b configs/efficientUnet_FDM_fast_CHAOS_to_SABSCT.yaml

# 3. 极速配置（最快）
python main.py -b configs/efficientUnet_FDM_ultra_fast_CHAOS_to_SABSCT.yaml

# 4. 完全禁用FDM（基线）
python main.py -b configs/efficientUnet_FDM_disabled_CHAOS_to_SABSCT.yaml

# 5. 运行性能诊断
python profile_training_speed.py
```

---

## 进一步调试

如果优化后仍然慢，检查：

1. **GPU利用率**：
   ```bash
   watch -n 1 nvidia-smi
   ```
   应该看到GPU利用率 >90%

2. **CPU利用率**：
   ```bash
   htop
   ```
   检查是否有进程占用过多CPU

3. **内存**：
   检查是否OOM导致swap

4. **数据增强**：
   在 `dataloaders/` 中添加计时，找出慢的增强操作

---

## 总结

**关键点**：
1. ⚠️ **DataLoader是主要瓶颈（87%）**，不是FDM（~3%）
2. ✅ 复数警告已修复
3. ✅ 提供3种优化配置
4. ✅ 预期加速比：3倍+

**立即行动**：
```bash
# 使用快速配置重新训练
python main.py -b configs/efficientUnet_FDM_fast_CHAOS_to_SABSCT.yaml
```

观察新的训练log中的 `time` 和 `data` 指标！
