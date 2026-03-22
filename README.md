# UAV Collaborative Detection

一个基于 VisDrone 数据集的无人机目标检测实验仓库，包含：

- `YOLO11n` 小模型训练与评估
- `RT-DETR-L` 大模型训练与推理
- `VisDrone -> YOLO` 标注格式转换
- 一个简单的协同路由推理脚本

## 1. 数据下载

VisDrone 数据集从官方仓库下载：

`https://github.com/VisDrone/VisDrone-Dataset`

本项目默认使用目标检测数据集 `VisDrone2019-DET`，至少需要下载：

- `VisDrone2019-DET-train`
- `VisDrone2019-DET-val`

下载并解压后，建议目录结构如下：

```text
uav_collab_det/
├── data/
│   ├── VisDrone2019-DET-train/
│   │   ├── images/
│   │   └── annotations/
│   ├── VisDrone2019-DET-val/
│   │   ├── images/
│   │   └── annotations/
│   ├── visdrone.yaml
│   └── visdrone_1of3.yaml
├── scripts/
├── models/
└── README.md
```

## 2. UV环境安装

建议使用 Python 3.10+。

```bash
uv venv --python 3.12
source .venv/bin/activate
pip install ultralytics opencv-python tqdm pyyaml numpy
```

如果你使用 GPU，请确保本机 PyTorch / CUDA 环境可用，或先按 PyTorch 官方方式安装对应版本。

## 3. 数据预处理

VisDrone 原始标注不是 YOLO 格式，需要先转换。转换脚本位于仓库根目录，默认会读取 `data/`：

```bash
python convert_visdrone_to_yolo.py
```

如果你的数据不在默认位置，也可以手动指定：

```bash
python convert_visdrone_to_yolo.py --root data
```

运行后会在以下目录自动生成 `labels/`：

- `data/VisDrone2019-DET-train/labels`
- `data/VisDrone2019-DET-val/labels`

类别映射已经在脚本中固定为 10 类：

- pedestrian
- people
- bicycle
- car
- van
- truck
- tricycle
- awning-tricycle
- bus
- motor

默认数据配置文件为 [data/visdrone.yaml](/home/ellen/Projects/uav_collab_det/data/visdrone.yaml)。

## 4. 训练

### 4.1 训练小模型 YOLO11n

使用默认参数训练：

```bash
python scripts/train_small.py
```

或使用配置文件：

```bash
python scripts/train_small.py --config configs/train_small.yaml
```

默认会读取：

- 模型权重：`yolo11n.pt`
- 数据配置：`data/visdrone.yaml`

训练结果默认保存在：

`runs/yolo11n_visdrone/`

### 4.2 训练大模型 RT-DETR-L

```bash
python scripts/train_large.py
```

或使用配置文件：

```bash
python scripts/train_large.py --config configs/train_large.yaml
```

如果你想跑多卡分布式训练，直接把 `device` 设成逗号分隔的 GPU 列表即可，例如：

```bash
python scripts/train_large.py \
  --config configs/train_large.yaml \
  --device 0,1,2,3 \
  --batch 16 \
  --workers 16 \
  --name rtdetr_l_visdrone_ddp
```

常见建议：

- 多卡时把 `batch` 适当调大，按总显存余量试
- `workers` 通常可以按 GPU 数量一起增加
- 如果内存足够，可以把 `cache` 设成 `ram` 或 `disk` 加速数据读取
- 断点续训时传入 checkpoint 路径并加 `--resume`

默认会读取：

- 模型权重：`rtdetr-l.pt`
- 数据配置：`data/visdrone.yaml`

训练结果默认保存在：

`runs/rtdetr_l_visdrone/`

## 5. 评估与推理

### 5.1 评估小模型

```bash
python scripts/eval_small.py --config configs/eval_small.yaml
```

默认会使用训练得到的小模型权重：

`runs/detect/runs/yolo11n_visdrone/weights/best.pt`

如果想手动指定某个 checkpoint，例如 `last.pt`，可以这样跑：

```bash
python scripts/eval_small.py \
  --model runs/detect/runs/yolo11n_visdrone/weights/last.pt
```

注意：

- 评估脚本要求模型类别与数据集类别严格一致
- 如果直接拿官方 `yolo11n.pt` 做 `val`，类别不一致时脚本会报错，这是预期行为
- 正常做法是使用已经在 VisDrone 上微调后的权重进行评估

### 5.2 大模型推理

```bash
python scripts/infer_large.py --config configs/infer_large.yaml
```

推理结果默认保存在：

`runs/predict/rtdetr_l_pretrained_visdrone/`

## 6. 协同推理

协同推理脚本会先用小模型做预测，再通过简单路由器决定是否升级到大模型：

```bash
python scripts/infer_collab.py
```

脚本默认读取以下权重：

- `runs/yolo11n_visdrone/weights/best.pt`
- `runs/rtdetr_l_visdrone/weights/best.pt`

输出文件默认写入：

`runs/collab_predictions.json`

## 7. 关于 `*-third` 子集

仓库里的 [data/visdrone_1of3.yaml](/home/ellen/Projects/uav_collab_det/data/visdrone_1of3.yaml) 和 [configs/infer_large.yaml](/home/ellen/Projects/uav_collab_det/configs/infer_large.yaml) 默认指向：

- `data/VisDrone2019-DET-train-third/`
- `data/VisDrone2019-DET-val-third/`

这两个目录不是转换脚本自动生成的，通常表示你手动准备的 1/3 子集。如果本地没有这两个目录，可以：

- 直接改配置文件，换成完整数据集目录
- 或者自行从 train/val 中抽取子集后按同样目录命名

## 8. 主要文件说明

- [convert_visdrone_to_yolo.py](/home/ellen/Projects/uav_collab_det/convert_visdrone_to_yolo.py)：把 VisDrone 检测标注转为 YOLO 格式
- [scripts/train_small.py](/home/ellen/Projects/uav_collab_det/scripts/train_small.py)：训练 YOLO11n
- [scripts/train_large.py](/home/ellen/Projects/uav_collab_det/scripts/train_large.py)：训练 RT-DETR-L
- [scripts/eval_small.py](/home/ellen/Projects/uav_collab_det/scripts/eval_small.py)：评估小模型
- [scripts/infer_large.py](/home/ellen/Projects/uav_collab_det/scripts/infer_large.py)：大模型推理
- [scripts/infer_collab.py](/home/ellen/Projects/uav_collab_det/scripts/infer_collab.py)：协同推理
- [models/router.py](/home/ellen/Projects/uav_collab_det/models/router.py)：简单路由策略

## 9. 快速开始

如果你只想先把流程跑通，可以按下面顺序执行：

```bash
python convert_visdrone_to_yolo.py
python scripts/train_small.py --config configs/train_small.yaml
python scripts/train_large.py --config configs/train_large.yaml
python scripts/infer_collab.py
```
