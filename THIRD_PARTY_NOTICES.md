# 第三方与许可证

本集成源码发布版采用AGPL-3.0-only，完整文本见根目录 `LICENSE`。
原项目自写代码的MIT版权声明保留在 `LICENSES/MIT-original.txt`；这里未删除原许可，
也没有将第三方文件重新标成自己的专有作品。工作目录原有许可未被本次导出覆盖。

## 随包包含的第三方源码

仅包括固定FilterPy的MIT源码片段/完整作者文件与原许可：

| 目录 | 来源 | 许可 |
|---|---|---|
| `src/event_drone_forecast/_vendor/filterpy_linear_upstream` | [rlabbe/filterpy](https://github.com/rlabbe/filterpy)，提交 `3b51149ebcff0401ff1e10bf08ffca7b6bbc4a33` | MIT，目录内LICENSE与快照哈希保留 |
| `src/event_drone_forecast/_vendor/filterpy_process_upstream` | 同一FilterPy提交 | MIT，原源码/许可保留 |

位置核心执行固定作者predict/update及过程噪声函数，不能称这些函数是本项目原创。
`SOURCE_PROVENANCE.json`另记录自有历史代码的原路径、哈希、提取定义和导出适配；
这些仅是本项目重组，不构成对上游著作权的重新授权。

## 外部安装，不随包分发

* **Ultralytics**：实际用于检测预处理/NMS和可选训练导出；遵守其
  [AGPL-3.0/Enterprise许可](https://www.ultralytics.com/license)。本版不是“纯MIT检测系统”。
  商业或闭源使用不能只看本仓库许可；模型转ONNX不自动豁免上游模型条款。
* **Metavision SDK**：遵守Prophesee条款。OpenEB基础模块与Advanced/Pro模块可用条件不同；
  SDK、密钥、安装器、二进制均未打包。本仓库不授予SDK部署/分发权利。
* **NumPy、SciPy、OpenCV、PyTorch、Torchvision、ONNX、ONNX Runtime、Pillow、Matplotlib**：
  分别由其各自许可证覆盖，依赖安装时应保留相应通知；本ZIP不附带这些库或CUDA运行时。
* **Numba、llvmlite、pybind11、MSVC**：可选JIT/本地SDK适配构建依赖，不随源码包分发。
  自有AVX2包装仅引用用户已经安装的SDK头文件，不包含或修改厂商头文件；本地生成二进制不得当作本包已授权的通用分发资产。

根AGPL文本从本机已安装Ultralytics发行包的原始GNU许可文本逐字复制，来源与SHA写入构建记录；
该许可文本的版权归Free Software Foundation，保留其完整内容。

## 排除的研究仓库、权重和数据

HelixTrack作者仓库、C++头文件、DLL、本地ABI桥接和未明确可重分发的适配目录不在本包。
历史上调用过的EEPPR、DroneStalker、FRED数据及其他整仓库均不在本包。
本版空间相位FFT为自写NumPy/SciPy实验，不复制或链接HelixTrack实现。
ViSP仅为轮廓法线跟踪方法参照，未复制、链接或打包其源码。

检测权重未分发。第三方论文PDF、论文截图、原作者数据/模型均没有打进ZIP。
示例媒体权利范围见 `examples/README.md`。
