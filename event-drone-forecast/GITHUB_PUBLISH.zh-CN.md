# GitHub更新说明

## 更新方式

先下载/备份自己已经发布的版本。本次是v0.4.0-alpha.2预览版，旧alpha.1留作历史版本。

**推荐仓库根目录直接包含README.md、LICENSE、forecast.py、src、docs等内容。**
解压后上传内层event-drone-forecast目录的“内容”，不是再次上传外层目录形成多级嵌套。
用Git更新已有仓库时，先核对远程地址和差异，再提交、推送；本源码整理过程没有替你执行远程发布。
不要只换README而保留旧源码，否则文档与运行结果不对应。

如果暂时继续沿用“根目录README＋event-drone-forecast子目录”的现有布局：
更新该子目录内容，并把本次交付附带的ROOT_README_FOR_NESTED.md全文复制到根README。
根目录另放与子目录一致的LICENSE、NOTICE.md、THIRD_PARTY_NOTICES.md、CITATION.cff，
让仓库访问者直接看见授权与署名。README链接不能代替实际许可证文件。
稍后迁移成平铺结构时恢复内层完整README，并移除过时重复子目录。

ZIP可作为Release附件；仅上传ZIP不会展开源码树供浏览。
本版超过100个文件，网页上传请分批进行（如先根文件，再分别上传src、docs、tools、examples、native等目录），每批不超过100个文件；或使用GitHub Desktop/Git提交完整改动。
GitHub网页每次最多100文件、单文件25MiB，见 [GitHub官方说明](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository)。不要把文件数超限误认为源码损坏。
新建Release时使用标签v0.4.0-alpha.2，勾选Pre-release。
发布前核对ZIP的SHA256和SOURCE_MANIFEST.json。

## 可复制仓库简介

基于原生ON/OFF事件的四旋翼五圆建模与二维短时预测研究原型：按四桨、三桨、两点与侧视轴线分工更新，结合周期代理与未知合力反馈。

## 可复制Release文案

v0.4.0-alpha.2更新五圆退化可见性预测主线，新增当前KLT运动复用、检测框重定位纠偏、
同窗未知合项估计、切向反馈和计算共享，保留旧配置与合成演示。

自采侧视开发片段0.4秒平均误差43.27px（26目标，P9580.60px），另一录像62.65px（15目标）；
固定侧视处理能力35.41FPS、清晰32.94FPS，仍有初期积压，未证明全场景稳定实时或公开基准优势。

原始数据、SDK与检测权重未打包，参考权重没有公开下载地址。完整RAW需要外部环境；
普通Python可运行合成演示。请先读README、MODEL_CARD与RESULTS。

集成版AGPL-3.0-only，保留原MIT/FilterPy通知；上游模型/SDK/数据各有其条款。
非论文官方代码，不宣称真实RPM、牛顿推力或通用方法首创。开发使用AI编码工具辅助。

## 发布必须保留

- LICENSE、LICENSES、NOTICE.md、THIRD_PARTY_NOTICES.md及引用文件。
- 参考论文和实际代码复用来源，不能把上游函数改名后称原创。
- 私有模型不可获取、SDK要求、数据范围、计时口径和失败/限制说明。
- AI辅助开发说明、示例媒体授权边界。
- 自己的署名可改成希望公开的姓名或GitHub账号，不用公开私人路径和邮箱。

不要上传RAW、他人权重、SDK头文件/DLL、编译缓存、密钥或个人交接日志。
没有独立真值和统一基准时，不写“真实推力”“首次”“SOTA”“全程30FPS”等未经支持的结论。
