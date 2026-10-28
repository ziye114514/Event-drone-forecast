# 相关工作与本项目关系

| 工作 | 作用 | 本包包含其代码吗 |
|---|---|---|
| [Thakur et al., Relative State Estimation using Event-Based Propeller Sensing, arXiv:2604.18289](https://arxiv.org/abs/2604.18289) | 桨盘几何、周期和动力学状态估计的主要参照 | 不包含；不声称完整复现 |
| [Spetlik et al., HelixTrack, arXiv:2603.09235](https://arxiv.org/abs/2603.09235) | 几何与空间相位、RPM估计的参照 | 不包含作者核心或DLL；本地原型实验的结果不代表论文成绩 |
| [Spetlik et al., CoHelix, arXiv:2610.31384](https://arxiv.org/abs/2610.31384) | 多桨共享投影几何和慢椭圆观察器的参照 | 不包含；本版主要用弱透视仿射状态，不等于其完整共享单应滤波器 |
| [Chen et al., Count Every Rotation and Every Rotation Counts: Exploring Drone Dynamics via Propeller Sensing, arXiv:2511.13100](https://arxiv.org/abs/2511.13100) | EventPro周期与无人机动力学线索的参照 | 不包含 |
| [effqpdet](https://github.com/radimspetlik/effqpdet) | 事件桨盘检测与属性提取思路 | 不包含 |
| [Real-Time Drone Detection in Event Cameras via Per-Pixel Frequency Analysis, arXiv:2603.08386](https://arxiv.org/abs/2603.08386) | 每像素周期/谐波筛选与检测前端参照 | 不包含论文作者代码；当前固定像素周期实现不声称逐项复现该文 |
| [Comport et al., Real-time markerless tracking for augmented reality: the virtual visual servoing framework](https://pubmed.ncbi.nlm.nih.gov/16805268/) / [ViSP](https://github.com/lagadic/visp) | 轮廓法线距离和鲁棒模型跟踪参照 | 不复制或调用ViSP代码 |
| [FilterPy](https://github.com/rlabbe/filterpy) | 实际执行的固定线性滤波/过程噪声原函数 | 包含MIT快照，见第三方通知 |
| [Ultralytics](https://github.com/ultralytics/ultralytics) | 实际检测预处理/NMS、可选训练导出依赖 | 由用户独立安装，遵循其AGPL/商业许可 |
| [DroneStalker](https://github.com/takara-ai/DroneStalker) | 历史参考检测器谱系与工程对照 | 代码/权重均不包含 |

不把通用几何、Kalman、扰动观察器、时序切换、角矩/FFT或相位测速本身作为首创。
合理定位是**面向事件四旋翼部分可见与退化观测的研究原型、工程整合和具体机制探索**。
如写简历或项目简介，可描述自己构建和评估的框架；不要把引用论文的指标写成本项目指标。
