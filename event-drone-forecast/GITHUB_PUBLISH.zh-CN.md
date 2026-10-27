# GitHub发布：可以直接复制的文案

## 仓库设置

仓库名建议：`event-drone-forecast`

About / Description：

> 基于原生ON/OFF事件的四旋翼五圆建模与二维短时预测研究原型：部分可见/侧视退化处理、条件周期与推力代理、未知合力反馈，附合成演示和RAW接口。

Topics：

```text
event-camera event-based-vision quadrotor motion-prediction rotor-tracking python research-prototype
```

Release标签：`v0.4.0-alpha.1`

Release标题：`Five-circle event-based drone forecasting — source preview`

Release正文：直接复制根目录 `RELEASE_NOTES.md`，勾选 **This is a pre-release**。

## 发布时需要保留的说明

1. **项目身份**：独立研究原型/工程整合，不是引用论文官方代码，不宣称SOTA或方法首创。
2. **许可**：保留根AGPL-3.0、原MIT声明和第三方LICENSE/NOTICE；Ultralytics、模型、SDK各有条件。
3. **参考工作**：保留 `docs/RELATED_WORK.md`，引用实际使用/借鉴的上游，不把别人的代码/结果写成自己的。
4. **实验范围**：2D像素预测，历史已暴露的小样本、无独立RPM/姿态/力真值；平均FPS不等于稳定实时。
5. **可复现性**：源码和合成演示可用；当前参考权重、自采RAW、私有标注不附，真实分数无法直接完整复现。
6. **开发方式**：AI工具辅助实现/检索/文档，维护者承担审查与结果责任。该项是本项目的透明说明，不是GitHub通用强制规定。

以上内容已经写入README、NOTICE、第三方说明、模型说明和结果说明，不必自行重写。
`CITATION.cff`暂用项目贡献者署名，无虚构姓名/DOI；发布后可补自己的公开署名与仓库URL。

## 上传方式

ZIP不会自动变成可浏览的GitHub源码树。解压后，把**内层项目目录的内容**上传到仓库根目录，
确保README、LICENSE、forecast.py位于仓库根目录，而不是只上传一个ZIP或多套一层目录。
ZIP本身可另附在Release资产中。

GitHub网页一次最多上传100个文件、单文件25MiB；文件较多时用GitHub Desktop或分批上传。
官方说明：[上传文件](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository)、
[创建Release](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)。

本包未包含RAW、权重、SDK、DLL、虚拟环境、私有配置、聊天交接日志或私人路径。
不要把整个工作目录的output、dataset、下载的第三方整仓库补传进去。

## 简短项目介绍

> 我构建了一个面向事件相机四旋翼的五圆几何建模与二维短时预测原型。项目按照桨盘可见证据进行部分信息更新与退化处理，将当前残弧、推断桨心和短时持有状态分离，并探索以周期/方向代理与未知合力反馈改进预测。实现包含可复现合成验证、因果日志和同起点对照；目前仍未证明真实推力恢复、稳定实时或优于运动学基线。
