# DataLoader线程冲突问题修复

## 问题诊断

你的训练慢主要是因为 **DataLoader多线程与OpenMP线程限制冲突**：

```
OMP_NUM_THREADS=1 (main.py设置)
+
num_workers=8 (config设置)
+
persistent_workers=True
=
每个worker进程只能用1个OpenMP线程 → 极慢！
```

**现象**：
- `data: 70.7958秒` - DataLoader占77%时间
- 之前设置`OMP_NUM_THREADS=1`解决了其他卡顿，但引入了新问题

---

## 根本原因

1. **main.py设置了线程限制**：
   ```python
   os.environ.setdefault("OMP_NUM_THREADS", "1")
   os.environ.setdefault("MKL_NUM_THREADS", "1")
   ```

2. **DataLoader的worker进程继承了这些环境变量**

3. **结果**：每个worker的numpy/scipy操作只能用1个线程 → 数据预处理极慢

---

## 解决方案

### 方案1：使用num_workers=0（最简单）

**配置文件**：`configs/efficientUnet_FDM_fix_threading_CHAOS_to_SABSCT.yaml`

```yaml
data:
  params:
    num_workers: 0  # 单进程加载，避免线程冲突
```

**优点**：
- 完全避免多进程开销
- 每个操作可以用全部CPU
- 简单可靠

**缺点**：
- 没有并行加载
- 如果数据预处理很重，可能成为瓶颈

**训练命令**：
```bash
python main.py -b configs/efficientUnet_FDM_fix_threading_CHAOS_to_SABSCT.yaml
```

---

### 方案2：移除线程限制（推荐）

如果你之前设置`OMP_NUM_THREADS=1`是为了避免卡顿，现在可以尝试更智能的设置：

**编辑main.py**，在导入之前：
```python
import os
# 智能设置：允许适度并行，但不过度
cpu_count = os.cpu_count() or 4
os.environ.setdefault("OMP_NUM_THREADS", str(max(1, cpu_count // 4)))
os.environ.setdefault("MKL_NUM_THREADS", str(max(1, cpu_count // 4)))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(max(1, cpu_count // 4)))
```

然后使用：
```yaml
num_workers: 2  # 适度并行
```

**原理**：
- 如果32核CPU，每个库用8线程
- 2个worker × 8线程 = 16线程并行
- 避免过度线程竞争

---

### 方案3：在worker_init_fn中重新设置线程数

**修改main.py的worker_init_fn**：
```python
def worker_init_fn(worker_id):
    import os
    # 在worker进程中恢复合理的线程数
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"

    import numpy as np
    np.random.seed(np.random.get_state()[1][0] + worker_id)
```

然后在DataLoader中使用：
```python
train_loader = DataLoader(..., num_workers=2, worker_init_fn=worker_init_fn)
```

**优点**：
- 主进程保持线程限制（避免卡顿）
- Worker进程可以并行（加速数据加载）

---

## 快速测试

### 测试1：num_workers=0
```bash
python main.py -b configs/efficientUnet_FDM_fix_threading_CHAOS_to_SABSCT.yaml
```

观察 `data:` 时间是否降低。

### 测试2：不同num_workers对比

创建测试脚本：
```python
import os
os.environ["OMP_NUM_THREADS"] = "1"

import time
from torch.utils.data import DataLoader
from dataloaders.AbdominalDataset import get_training

dataset = get_training(location_scale=True, modality=['CHAOST2'], tile_z_dim=1)

for nw in [0, 1, 2, 4]:
    loader = DataLoader(dataset, batch_size=32, num_workers=nw, drop_last=True)

    start = time.time()
    for i, batch in enumerate(loader):
        if i >= 5:
            break
    elapsed = time.time() - start

    print(f"num_workers={nw}: {elapsed/5:.2f}秒/batch")
```

---

## 复数警告修复

确保pull最新代码：
```bash
git pull origin
```

检查 `util/freq_domain_mod.py` 第258行应该是：
```python
x_mod = torch.real(x_mod)  # 不是 x_mod.real
```

---

## 推荐配置组合

### 配置A：简单快速（推荐首选）
```yaml
# configs/efficientUnet_FDM_fix_threading_CHAOS_to_SABSCT.yaml
num_workers: 0
fdm_num_bands: 4
fdm_init_from_stats: false
```

**预期**：
- DataLoader时间：~5-10秒（从70秒降低）
- 总时间：~25秒/iter

### 配置B：平衡模式
修改main.py的worker_init_fn（见方案3），然后：
```yaml
num_workers: 2
fdm_num_bands: 4
fdm_init_from_stats: false
```

**预期**：
- DataLoader时间：~10-15秒
- 总时间：~30秒/iter

### 配置C：最大性能（需要修改main.py线程设置）
```python
# main.py中改为：
os.environ.setdefault("OMP_NUM_THREADS", "4")
```
```yaml
num_workers: 2
fdm_num_bands: 4
```

---

## 实施步骤

**立即执行**：

1. **Pull最新代码**（修复复数警告）：
   ```bash
   cd /root/FBD
   git pull origin
   ```

2. **使用修复配置**：
   ```bash
   python main.py -b configs/efficientUnet_FDM_fix_threading_CHAOS_to_SABSCT.yaml
   ```

3. **观察第一个epoch的log**：
   ```
   time: XX  data: YY
   ```
   期望：
   - `time` < 30秒
   - `data` < 15秒

4. **如果仍然慢**，尝试在main.py中修改worker_init_fn（方案3）

---

## 理解问题本质

```
原始问题：过多线程 → 设置OMP_NUM_THREADS=1
副作用：DataLoader worker继承限制 → 数据加载极慢
解决方案：
  选项1：num_workers=0（避免继承）
  选项2：worker_init_fn中恢复线程数（分离主/worker设置）
  选项3：移除过度限制，用适度线程数
```

---

## 调试checklist

- [ ] Pull了最新代码（修复复数警告）
- [ ] 使用了fix_threading配置
- [ ] 观察了新的data时间
- [ ] 如果仍慢，尝试了worker_init_fn修改
- [ ] 检查GPU利用率（nvidia-smi）
- [ ] 确认没有其他进程占用CPU

---

## 预期结果

**修复前**：
```
time: 91.1037  data: 70.7958  (data占78%)
```

**修复后（num_workers=0）**：
```
time: ~25秒  data: ~5秒  (data占20%)
```

**加速比：3.6倍**
