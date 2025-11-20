# Location Scale Augmentation 速度优化 Patch

## 发现的问题

**根本原因找到了！**

`dataloaders/location_scale_augmentation.py` 第6行：
```python
def __init__(self, nTimes=100000):  # ← 10万个点！
```

每次数据增强都要：
1. 计算10万个点的Bernstein多项式
2. 在`OMP_NUM_THREADS=1`下，这是纯串行计算
3. **每个样本耗时可能数秒！**

## 解决方案

### 方案1：减少nTimes（推荐）

**修改文件**：`dataloaders/location_scale_augmentation.py` 第6行

```python
# 修改前
def __init__(self, vrange=(0.,1.), background_threshold=0.01, nPoints=4, nTimes=100000):

# 修改后（10倍加速）
def __init__(self, vrange=(0.,1.), background_threshold=0.01, nPoints=4, nTimes=10000):

# 或者更激进（100倍加速）
def __init__(self, vrange=(0.,1.), background_threshold=0.01, nPoints=4, nTimes=1000):
```

**影响**：
- Bezier曲线稍微不那么平滑
- 但对增强效果影响很小（人眼难以分辨）
- **速度提升10-100倍**

### 方案2：使用快速版本

我已经创建了优化版本：`dataloaders/location_scale_augmentation_fast.py`

**修改 AbdominalDataset.py**：
```python
# 修改前
from dataloaders.location_scale_augmentation import LocationScaleAugmentation

# 修改后
from dataloaders.location_scale_augmentation_fast import LocationScaleAugmentationFast as LocationScaleAugmentation
# 或极速版本
from dataloaders.location_scale_augmentation_fast import LocationScaleAugmentationUltraFast as LocationScaleAugmentation
```

## 立即实施

### 快速修复（推荐）

```bash
cd /root/FBD

# 方法1：直接修改原文件
sed -i 's/nTimes=100000/nTimes=10000/g' dataloaders/location_scale_augmentation.py

# 验证修改
grep nTimes dataloaders/location_scale_augmentation.py

# 重新训练
python main.py -b configs/efficientUnet_FDM_zero_workers_CHAOS_to_SABSCT.yaml
```

### 完整修复（需要pull代码）

```bash
cd /root/FBD
git pull origin  # 获取location_scale_augmentation_fast.py

# 然后修改 dataloaders/AbdominalDataset.py
# 将 LocationScaleAugmentation 的导入改为 Fast 版本
```

## 预期效果

### 修改前
```
time: 16.7秒/iter, data: 13.2秒
├── LSA: ~10秒（推测）
└── 其他: ~3秒
```

### 修改后（nTimes=10000）
```
time: ~8秒/iter, data: ~4秒
├── LSA: ~1秒
└── 其他: ~3秒

加速比: 2倍
```

### 修改后（nTimes=1000）
```
time: ~6秒/iter, data: ~3秒
├── LSA: ~0.1秒
└── 其他: ~3秒

加速比: 2.8倍
```

## 验证

训练开始后，观察：
```
Epoch: [0]  [ 0/12]
time: XX.XXXX  data: YY.YYYY

期望：
- time < 10秒
- data < 5秒
```

## 风险评估

### nTimes=10000（安全）
- ✅ 曲线仍然很平滑
- ✅ 增强效果几乎不变
- ✅ 10倍加速

### nTimes=1000（激进）
- ⚠️ 曲线稍显粗糙
- ⚠️ 可能略微影响增强质量
- ✅ 100倍加速

**建议**：先试 `nTimes=10000`，如果还慢再试 `nTimes=1000`

## 其他优化（可选）

### 1. 缓存增强结果
如果数据集不大，可以预计算所有增强：
```python
# 训练前一次性生成增强数据并保存
```

### 2. 使用 Numba 加速
```python
from numba import jit

@jit(nopython=True)
def bernstein_poly(i, n, t):
    ...
```

### 3. 多线程增强（需要移除OMP限制）
```python
# 在worker中临时允许多线程
def worker_init_fn(worker_id):
    os.environ["OMP_NUM_THREADS"] = "4"
```

## 总结

**关键发现**：Location Scale Augmentation 的 `nTimes=100000` 是速度瓶颈！

**立即行动**：
```bash
# 在服务器上执行
sed -i 's/nTimes=100000/nTimes=10000/g' dataloaders/location_scale_augmentation.py
python main.py -b configs/efficientUnet_FDM_zero_workers_CHAOS_to_SABSCT.yaml
```

**预期**：速度提升2倍以上（从16.7秒降至8秒）
