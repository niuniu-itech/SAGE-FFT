<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/logo.svg">
    <img src="assets/logo.png" alt="SAGE-FFT — FFT butterfly logo" width="640">
  </picture>
</p>

[English](README.md)

版本 **0.3.0**，作者与维护者：**even**。

SAGE-FFT 将已注册的 CUDA FFT pipeline 导入 FFT contract 和 stage IR，
由编译器检查分组、尾部算子融合和核映射，再生成 Ascend C 源码与执行计划。
LLM 参与离线搜索，部署后的 FFT 执行不调用模型。

## 安装与测试

需要 Python 3.10 或以上版本。在当前目录执行：

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
sage validate-cpu --shape 8,8,8
sage emit-cuda --shape 8,8,8 --output runs/reference.cu
sage migrate runs/reference.cu --variant grouped --blocks 8 --output runs/migrated
```

CPU 测试使用生成输入和合成响应，不需要服务器或 API Key。
CUDA 导入器支持已注册的 cuFFT 源码结构，不是通用 CUDA 编译器。

## 真机与模型配置

参见[复现指南](docs/REPRODUCIBILITY.md)、[原生基准](docs/NATIVE_BENCHMARKS.md)
和[模型 API 配置](docs/LLM_CONFIGURATION.md)。按 `.env.example` 导出自己的环境变量，
安装 CANN、CUDA 或相应模型运行时后执行实验。配置文件不会自动加载。
SiliconFlow API Key 由每个使用者自行提供，不写入源码。

纯净版保留源码、测试、输入生成器和复现说明，不包含历史实验数据、模型对话、
论文迭代记录、个人路径或机器配置。实验会在新工作目录生成自己的结果；
大模型决策和硬件计时存在波动，应保留真实结果与失败记录。

源码目录结构见英文 README。许可状态见 [LICENSING.md](LICENSING.md)。
