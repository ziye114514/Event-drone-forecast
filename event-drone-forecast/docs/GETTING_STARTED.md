# 安装与运行

## 1. 合成演示：普通Python即可

在仓库根目录运行：

```text
python -m venv .venv
.venv\Scripts\python -m pip install -e .
.venv\Scripts\python forecast.py doctor --base-only
.venv\Scripts\python forecast.py demo --output outputs/demo_01
.venv\Scripts\python -m unittest discover -s tests -v
```

Linux/macOS将Python路径改为 `.venv/bin/python`。合成演示不要求SDK、检测器或GPU。
完整RAW只在下述Windows环境做过本机接线，未声称跨平台RAW运行通过。

## 2. 真实RAW的两个环境

| 环境 | 作用 | 必需依赖 |
|---|---|---|
| SDK Python | RAW解码、事件处理、五圆几何与状态预测 | 基础依赖＋兼容的Metavision SDK/Open/Advanced模块 |
| GPU Python | 单类ONNX整机检测 | CUDA版PyTorch/Torchvision、ONNX Runtime GPU、Ultralytics |

两个环境可以相同，也可以分开。SDK按 [Prophesee官方安装说明](https://docs.prophesee.ai/stable/installation/index.html)
安装；本项目使用 `FrequencyAlgorithm` 等事件算法，仅有OpenEB基础IO不一定足够。
SDK及CUDA软件包没有打进源码。

请在已激活SDK环境的终端运行；仅填写 `sdk_python` 不会替你安装或激活SDK。
按官方安装方式保留其模块搜索路径（某些安装需要 `PYTHONPATH`）和动态库搜索路径，
然后用 `doctor` 检查。若提示找不到 `metavision_sdk_base`，先修复SDK环境，不要从其他项目复制DLL。

先按 [PyTorch官方安装选择器](https://pytorch.org/get-started/locally/) 安装匹配驱动的PyTorch/Torchvision，
再在GPU环境安装 `requirements-gpu.txt`。`docs/ENVIRONMENT.md`记录本机版本，不能保证所有机器照抄同一个CUDA轮子都可用。

## 3. 配置检测模型

权重不随包提供。先阅读 [模型说明](MODEL_CARD.md)，使用满足契约的单类YOLO ONNX，
或按文档用自己的合法数据训练导出。

复制 `configs/runtime.example.json` 为 `configs/local.json`。
`sdk_python`、`gpu_python`可为Python可执行文件路径；留空表示当前Python。
其他相对路径以配置文件所在目录为基准。填写模型实际SHA256，防止静默换模型后仍沿用旧结果。

```text
python forecast.py doctor --config configs/local.json
python forecast.py predict --config configs/local.json --raw data/event.raw --output outputs/run_01 --start 26 --duration 2 --dry-run
```

dry-run只检查路径/模型哈希和生成命令，不读RAW内容、不创建输出目录、不启动GPU。
去掉 `--dry-run` 才真正执行。每次指定新的输出目录，避免覆盖实验。

## 4. 运行和输出

```text
python forecast.py predict --config configs/local.json --raw data/event.raw --output outputs/run_01 --start 26 --duration 2 --preview
```

默认目标30Hz源更新、10Hz整机神经检测，其间用当前ON/OFF图像位移更新整机ROI；
几何使用当前20ms事件，频率保留所选像素的原始事件时间。默认不运行传统并行加速度预测。
`--kinematic-control`显式打开研究控制；它增加计算量。

* `predictions.jsonl`：每个因果时刻的整机、结构、周期、预测和资格。
* `protocol.json` / `source_snapshot/`：输入约定、源码哈希、模型哈希。
* `summary.json`：处理时间、有效率等，不能把平均计算能力等同稳定低延迟更新率。
* `results.mp4`：指定preview时才生成；播放速度不用于测速。

当前只接受与原型匹配的1280×720原生事件。相机标定、其他分辨率、任意ONNX布局、多目标、
实际相机驱动、未知相机运动都需要适配，不能只改文件名后默认支持。

## 5. 获取训练图像和自有模型

在SDK环境导出原生ON/OFF图，再做真实人工框标注：

```text
python tools/export_event_frames.py --raw data/event.raw --output outputs/frames_01 --duration 10 --fps 10
```

工具不会生成假框或把未标图当负样本。按YOLO单类格式整理 `images/train`、`images/val`、
`labels/train`、`labels/val`，参考 `configs/dataset.example.yaml`。按录像拆分训练、验证与测试，避免邻帧泄漏。

在GPU环境运行：

```text
python tools/train_detector.py --data configs/dataset.local.yaml --base yolo11n.pt --epochs 50 --output outputs/training_01
```

该命令可能按Ultralytics机制下载基模型；也可用有权使用的本地模型。它不是本项目已执行的新训练结果。
输出ONNX及 `model_metadata.json`；将实际路径/哈希写入配置。
新训练模型不等于历史参考模型，不能直接复现README中的像素分数。
