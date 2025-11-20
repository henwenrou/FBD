# FDM训练快速开始

## 🚀 立即开始（3步）

```bash
# 1. 进入目录并pull最新代码
cd /root/FBD
git pull origin

# 2. 开始训练（使用优化配置）
python main.py -b configs/efficientUnet_FDM_optimized_CHAOS_to_SABSCT.yaml

# 3. 观察第一个iter的log
# 期望：time: ~25秒, data: ~8秒（之前是91秒和70秒）
```

---

## ✅ 已修复的问题

1. **复数警告** - 使用`torch.real()`替代`.real`
2. **DataLoader慢** - worker_init_fn中恢复线程数
3. **SMP版本兼容** - 自动检测0.2.x vs 0.3+

---

## 📊 预期性能

| 指标 | 修复前 | 修复后 | 改善 |
|------|--------|--------|------|
| 总时间 | 91秒 | ~25秒 | **3.6×** |
| Data加载 | 70秒 | ~8秒 | **8.8×** |
| 模型计算 | ~21秒 | ~17秒 | 1.2× |

---

## 🎛️ 配置选项

### 推荐（平衡）
```bash
configs/efficientUnet_FDM_optimized_CHAOS_to_SABSCT.yaml
# - fdm_num_bands: 4
# - fdm_init_from_stats: false
# - num_workers: 2
```

### 原始配置
```bash
configs/efficientUnet_FDM_CHAOS_to_SABSCT.yaml
# - fdm_num_bands: 8
# - fdm_init_from_stats: true
# - num_workers: 8
```

### 快速模式
```bash
configs/efficientUnet_FDM_fast_CHAOS_to_SABSCT.yaml
# - fdm_num_bands: 4
# - fdm_update_interval: 1000
# - num_workers: 4
```

### 禁用FDM（基线）
```bash
configs/efficientUnet_FDM_disabled_CHAOS_to_SABSCT.yaml
```

---

## 🔍 验证修复

训练开始后，检查第一个iter的log：

```
Epoch: [0]  [ 0/12]  time: XX.XXX  data: YY.YYY
```

**成功标准**：
- ✅ 没有复数警告（`[W Copy.cpp:244]`）
- ✅ `time` < 30秒
- ✅ `data` < 15秒
- ✅ `data/time` 比例 < 50%（之前是78%）

---

## 🐛 如果还慢

### 选项1：单进程加载
```bash
python main.py -b configs/efficientUnet_FDM_fix_threading_CHAOS_to_SABSCT.yaml
# 使用 num_workers: 0
```

### 选项2：简化FDM
```bash
python main.py -b configs/efficientUnet_FDM_ultra_fast_CHAOS_to_SABSCT.yaml
# 完全禁用PAC/SBP统计
```

### 选项3：性能诊断
```bash
python profile_training_speed.py
```

---

## 📚 详细文档

- `FINAL_FIX_SUMMARY.md` - 修复总结（必读）
- `THREADING_FIX.md` - 线程问题详解
- `SPEED_OPTIMIZATION.md` - 速度优化指南
- `FDM_INTEGRATION_SUMMARY.md` - FDM集成说明

---

## 💡 关键修复点

### 1. 复数警告修复
**文件**：`util/freq_domain_mod.py:258`
```python
x_mod = torch.real(x_mod)  # ✅ 正确
# x_mod = x_mod.real       # ❌ 会警告
```

### 2. DataLoader线程修复
**文件**：`main.py:20-29, 233`
```python
def worker_init_fn(worker_id):
    os.environ["OMP_NUM_THREADS"] = "4"  # Worker用4线程
    ...

train_loader = DataLoader(..., worker_init_fn=worker_init_fn)  # ✅ 必须加
```

---

## 🎯 一句话总结

**pull代码 → 用optimized配置训练 → 速度提升3倍**

```bash
git pull origin && \
python main.py -b configs/efficientUnet_FDM_optimized_CHAOS_to_SABSCT.yaml
```

🚀 开始训练！
