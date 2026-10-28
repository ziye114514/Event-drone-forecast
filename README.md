# Event Drone Forecast

基于原生ON/OFF事件的四旋翼五圆建模与二维短时运动预测研究原型。

源码位于 [event-drone-forecast](event-drone-forecast/) 子目录，当前版本为 **v0.4.0-alpha.2**。
运行命令前请先进入该子目录。

## 项目思路

以“一个机体参考中心＋四个桨盘”构建共享图像模型，根据当前可见证据切换处理：

- **四桨可见**：由真实弧线拟合椭圆、计算桨心，建立或更新共享模型。
- **三桨可见**：结合近期模型条件补全第四点；推断点不当作真实观测。
- **两点可见**：只更新可观测部分，不强行恢复完整三维平面。
- **近侧视成线**：更新当前整机轴线，保留不确定性并允许重捕获。

桨叶周期代理与当前图像法线构成相对输入，结合未知合项反馈预测未来 **0.1 / 0.2 / 0.4秒** 的图像位置。
二力有效模型为 `a_image = k · u + d`，未知项吸收重力、风、阻力及未建模影响。
当前不等于测出真实RPM、牛顿推力或完整三维姿态。

流程为：RAW → 整机ROI → ON/OFF抑制 → 残弧/椭圆 → 五圆可见性更新 → 周期输入 → 二力反馈 → 短时位置预测。

## 安装与合成演示

```bash
cd event-drone-forecast
python -m pip install -e .
python forecast.py doctor --base-only
python forecast.py demo --output outputs/demo_01
python -m unittest discover -s tests -v
```

合成演示不需要SDK、检测权重或GPU。真实RAW需要兼容的Metavision SDK、GPU环境和单类ONNX检测器。
**参考权重没有公开下载地址**，因此目前不能开箱复现历史精度数字。

## 当前开发结果与限制

| 自采开发片段 | 0.4秒平均误差 | P95 |
|---|---:|---:|
| 侧视，26个目标 | 43.27px | 80.60px |
| 另一录像，15个目标 | 62.65px | 124.00px |

固定侧视片段处理能力35.41FPS，清晰片段32.94FPS；初期仍有积压。
这些是已暴露开发片段的稀疏人工框中心评分，不是独立测试、mAP或全程低延迟保证。
未证明所有场景50px内、稳定30FPS或物理预测优于传统运动学。

![原生事件与五圆模型演示](event-drone-forecast/examples/clear.gif)

## 文档

- [完整检测与预测流程](event-drone-forecast/docs/CURRENT_PIPELINE.md)
- [安装与真实RAW运行](event-drone-forecast/docs/GETTING_STARTED.md)
- [检测模型、数据与复现限制](event-drone-forecast/docs/MODEL_CARD.md)
- [结果与失败记录](event-drone-forecast/docs/RESULTS.md)
- [相关论文及实际代码复用](event-drone-forecast/docs/RELATED_WORK.md)
- [本版本更新说明](event-drone-forecast/RELEASE_NOTES.md)

## 许可与署名

集成发布版采用 **AGPL-3.0-only**；原MIT和FilterPy等第三方通知保留。
请阅读 [LICENSE](event-drone-forecast/LICENSE)、[NOTICE](event-drone-forecast/NOTICE.md)、
[第三方说明](event-drone-forecast/THIRD_PARTY_NOTICES.md)。
SDK、权重、数据和示例媒体分别受其自身条款约束；源码包不含RAW、权重、SDK或编译二进制。

本项目不是引用论文的官方实现，不宣称五圆图形、椭圆拟合、Kalman、相位测速或扰动观察器本身首创。
开发使用AI编码工具辅助实现、检索和文档整理，由维护者负责核验与公开声明。
引用见 [CITATION.cff](event-drone-forecast/CITATION.cff)，并请引用实际使用的上游工作。
