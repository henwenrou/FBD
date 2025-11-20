# 紧急修复：persistent_workers 错误

## 错误信息
```
ValueError: persistent_workers option needs num_workers > 0
```

## 原因
`main.py` 第223行硬编码了 `persistent_workers=True`，但当 `num_workers=0` 时这是不允许的。

## 修复

**文件**：`main.py` 第222-226行

**修改前**：
```python
train_loader=DataLoader(data.datasets["train"], batch_size=data.batch_size,
                      num_workers=data.num_workers, shuffle=True,
                      persistent_workers=True, drop_last=True, pin_memory=True)
```

**修改后**：
```python
# persistent_workers requires num_workers > 0
use_persistent = data.num_workers > 0
train_loader=DataLoader(data.datasets["train"], batch_size=data.batch_size,
                      num_workers=data.num_workers, shuffle=True,
                      persistent_workers=use_persistent, drop_last=True, pin_memory=True)
```

## 立即执行

```bash
cd /root/FBD

# Pull修复后的代码
git pull origin

# 重新训练
python main.py -b configs/efficientUnet_FDM_zero_workers_CHAOS_to_SABSCT.yaml
```

## 预期输出

训练应该正常启动，看到：
```
385
Epoch: [0]  [ 0/12]  eta: 0:XX:XX
time: XX.XXXX  data: XX.XXXX
```

**不应该再有 ValueError**

## 验证修复

```bash
# 检查main.py是否包含修复
grep -A2 "persistent_workers requires" main.py

# 应该看到：
# persistent_workers requires num_workers > 0
# use_persistent = data.num_workers > 0
```

---

现在立即在服务器上执行：

```bash
git pull origin && \
python main.py -b configs/efficientUnet_FDM_zero_workers_CHAOS_to_SABSCT.yaml
```
