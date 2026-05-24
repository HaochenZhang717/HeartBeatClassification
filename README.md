# ECG_Datasets

一个用于把 PhysioNet 上多个 ECG 数据库统一处理成 NPZ 格式、并构建 AAMI 五分类心拍数据集的工作目录。

## 目录结构

项目根目录分为两个顶级子文件夹：`data_source/`（数据源，只读）和 `workspace/`（代码 + 处理后的数据）。

```
ECG_Datasets/
├── data_source/                  # PhysioNet 原始 WFDB 文件（只读、不动）
│   ├── mitdb/   1.0.0/ ...
│   ├── svdb/    1.0.0/ ...
│   ├── nsrdb/   1.0.0/ ...
│   ├── edb/     1.0.0/ ...
│   ├── stdb/    1.0.0/ ...
│   ├── afdb/    1.0.0/ ...
│   ├── nstdb/   1.0.0/ ...        （噪声库，process_ecg.py 暂未处理）
│   └── vfdb/    1.0.0/ ...        （室颤库，process_ecg.py 暂未处理）
└── workspace/                    # 代码 + 中间产物
    ├── README.md
    ├── process_ecg.py            # 原始 WFDB → 统一 NPZ 的处理脚本
    ├── process_log.txt           # 上一次 process_ecg.py 的运行日志
    ├── processed_data/           # 处理后的逐记录 NPZ（重采样到 250 Hz，前两路导联）
    │   ├── mitdb/   44 个记录
    │   ├── svdb/    78 个记录
    │   ├── nsrdb/   18 个记录
    │   ├── edb/     90 个记录
    │   ├── stdb/    18 个记录
    │   └── afdb/    23 个记录
    ├── heartbeat_dataset/        # 基于 processed_data 的心拍分类数据集库
    │   ├── __init__.py
    │   ├── build_index.py        # 扫描 processed_data，生成全局 beats_index.npz
    │   ├── beats_index.npz       # 每个心拍一行的扁平索引（~19 MB）
    │   ├── dataset.py            # HeartbeatDataset（PyTorch 风格的 Dataset）
    │   └── splits.py             # MITDB DS1 / DS2（de Chazal 2004 病人间划分）
    └── heartbeat_classifier/     # 1D-CNN 心拍分类模型
        ├── __init__.py
        ├── model.py              # HeartbeatCNN（4 ConvBlock + GAP + FC）
        ├── data.py               # 数据划分 / WeightedRandomSampler
        ├── utils.py              # FocalLoss / 指标 / 种子 / CSVLogger
        ├── train.py              # 训练入口（AdamW + Cosine + 早停）
        └── evaluate.py           # DS2 评估 + 混淆矩阵
```

> 所有 Python 命令都假设 **工作目录为 `workspace/`**。`process_ecg.py` 内部使用绝对路径指向 `../data_source/`，跑的位置无所谓；其他脚本（`heartbeat_dataset.build_index`、`heartbeat_classifier.train` 等）走相对路径，必须 `cd workspace/` 后再运行。

## 处理流水线

### 1. `process_ecg.py`：原始 → 统一 NPZ

把 `data_source/<dataset>/1.0.0/<record>` 下的 WFDB 文件转成 `processed_data/<dataset>/<record>.npz`。脚本内部使用绝对路径常量 `RAW_ROOT` / `OUT_ROOT`，分别指向 `data_source/` 和 `workspace/processed_data/`。

- **重采样**：所有信号统一到 `TARGET_FS = 250 Hz`，使用 `scipy.signal.resample_poly`。
- **导联**：固定取前两路。少于两路的记录会被跳过（stdb 里有 10 条因此被跳过）。
- **R 峰标注**：
  - mitdb / svdb / nsrdb / edb / stdb：从 `.atr` 中取 WFDB beat symbol，按 AAMI EC57 映射到 5 类 `N/S/V/F/Q`。
  - afdb：使用 `.qrsc`（优先）或 `.qrs`；它没有 AAMI 心拍标注，所以全部默认标 `'N'`（**建议训练时排除 afdb**）。
- **节律标注**：解析 `.atr` 里 `'+'` 标记和 `aux_note`（如 `(AFIB`、`(VT`），输出 `rhythm_start / rhythm_end / rhythm_label` 三个数组。
- **AAMI 排除记录**：mitdb 中的 `102, 104, 107, 217`（起搏）按标准排除；afdb 中 `00735, 03665` 因缺 `.dat` 排除。

每个 `.npz` 含字段：

| 字段 | 类型 / 形状 | 含义 |
| --- | --- | --- |
| `signal` | float32, (n_samples, 2) | 重采样后双导联信号 |
| `fs` | int32 | 采样率，恒为 250 |
| `beat_sample_idx` | int64, (n_beats,) | R 峰在 `signal` 中的样本下标 |
| `beat_aami_label` | <U1 | AAMI 5 类标签 `N/S/V/F/Q` |
| `beat_wfdb_symbol` | <U2 | 原始 WFDB 标注符号 |
| `rhythm_start` / `rhythm_end` | int64 | 节律段起止样本下标 |
| `rhythm_label` | object | 节律字符串（`N`、`AFIB`、`VT`、…） |
| `meta` | dict | `dataset / record_name / original_fs / duration_s / leads / beat_ann_source` |

### 2. `heartbeat_dataset/build_index.py`：构建全局心拍索引

扫描 `processed_data/` 中各数据库的所有 NPZ，把每个心拍展平成一行，输出 `heartbeat_dataset/beats_index.npz`。字段：`record_path, dataset, record_name, sample_idx, label_str, label_int`。

默认包含 `mitdb, svdb, edb, stdb, nsrdb`，**默认排除 afdb**（标签都被强制成 `N`，会严重失衡）。

### 3. `heartbeat_dataset.HeartbeatDataset`：训练用 Dataset

读取 `beats_index.npz` 后按 `dataset / records / classes / indices` 过滤，按 R 峰中心切窗。要点：

- **窗口**：`[r - pre_samples, r + post_samples)`，默认 `pre=100, post=150` → 窗长 250（约 1 秒）。
- **边界**：窗口越界时用边缘值 padding。
- **归一化**：`none / zscore-window / zscore-record` 三种。
- **导联**：`'both' / 0 / 1`。
- **懒加载**：内部用一个 LRU 缓存（默认 8 条记录的信号），适合按记录分组取样。
- **return_torch=True**：直接返回 `torch.Tensor`，否则返回 `numpy.ndarray`。
- 还提供 `class_counts()` 和 `class_weights(method='inv_freq')` 供处理类别不均衡。

### 4. `heartbeat_dataset/splits.py`：病人间划分

- `MITDB_DS1` / `MITDB_DS2`：de Chazal et al. 2004 标准病人间划分（各 22 个记录，互不重叠）。
- `split_by_record(index, val_frac, seed, dataset=None)`：按 `(dataset, record_name)` 整体划分到 train / val，避免病人级泄漏。
- `filter_index_by_records(index, records, dataset=None)`：按记录名筛选 index 行。

## 数据库概览（处理后）

| 数据库 | 处理成功 | 跳过 | 备注 |
| --- | --- | --- | --- |
| mitdb | 44 | 4 | 102/104/107/217 按 AAMI 排除 |
| svdb | 78 | 0 | |
| nsrdb | 18 | 0 | 正常窦律，无节律事件 |
| edb | 90 | 0 | 欧洲 ST-T |
| stdb | 18 | 10 | 10 条记录因少于 2 路导联被跳过 |
| afdb | 23 | 2 | 00735/03665 缺 .dat；心拍标签全为 `N` |

> `nstdb`（噪声压力测试库）和 `vfdb`（室颤库）目前只存在于 `data_source/`，`process_ecg.py` 中没有为它们配置处理规则。

## 典型用法

> 所有命令都在 `workspace/` 下运行（`cd workspace`）。

```bash
# 1) 处理原始数据（输出 → workspace/processed_data/）
python process_ecg.py

# 2) 构建心拍索引（输出 → heartbeat_dataset/beats_index.npz）
python -m heartbeat_dataset.build_index \
    --processed-root processed_data \
    --out heartbeat_dataset/beats_index.npz \
    --datasets mitdb svdb edb stdb nsrdb

# 3) 训练 1D-CNN
python -m heartbeat_classifier.train \
    --index-path heartbeat_dataset/beats_index.npz \
    --out-dir runs/baseline

# 4) 在 mitdb DS2 上评估
python -m heartbeat_classifier.evaluate \
    --ckpt runs/baseline/best.pt
```

```python
from heartbeat_dataset import HeartbeatDataset, MITDB_DS1, MITDB_DS2

train_ds = HeartbeatDataset(
    index_path="heartbeat_dataset/beats_index.npz",
    records=MITDB_DS1, dataset="mitdb",
    pre_samples=100, post_samples=150,
    normalize="zscore-window",
    return_torch=True,
)
val_ds = HeartbeatDataset(
    index_path="heartbeat_dataset/beats_index.npz",
    records=MITDB_DS2, dataset="mitdb",
    pre_samples=100, post_samples=150,
    normalize="zscore-window",
    return_torch=True,
)

wave, label = train_ds[0]   # wave: (250, 2) float32; label: int in [0..4]
print(train_ds.class_counts())
print(train_ds.class_weights())
```

## 依赖

- `numpy`, `scipy`（`resample_poly`）
- `wfdb`（读 PhysioNet 原始文件）
- `torch`、`scikit-learn`、`matplotlib`（分类训练 / 评估）
