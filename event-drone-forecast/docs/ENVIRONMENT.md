# 本机验证环境

基础/SDK侧：Windows，Python3.11，NumPy2.4.6，SciPy1.17.1，OpenCV headless5.0.0.93，Pillow12.3.0。GPU侧：Python3.13，Torch2.13.0+cu130，Torchvision0.28.0+cu130，Ultralytics8.4.110，ONNX Runtime GPU1.30.0，ONNX1.23.1。

这是已用本机环境记录，不是干净机器或所有CUDA/SDK组合的安装保证。合成CI不需要GPU/SDK；GitHub Actions配置不等于已经在GitHub运行通过。真实模型/数据分发边界见MODEL_CARD.md。
