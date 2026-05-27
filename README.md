# 轻量化中文语音识别系统

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![EMNLP](https://img.shields.io/badge/Conference-EMNLP%202025-purple.svg)]()

基于低秩压缩与INT8量化的轻量化中文语音识别系统——LiteASR论文复现、噪声鲁棒性验证与Web部署

---

## 📋 项目简介

本课程设计针对Whisper语音识别模型在边缘设备部署时面临的**体积大、速度慢、精度与效率矛盾**三大挑战，从两个正交的技术路线出发：

| 技术路线     | 方法                   | 效果                                       |
| ------------ | ---------------------- | ------------------------------------------ |
| **结构压缩** | PCA低秩分解（LiteASR） | 编码器输出 384维 → 140维，压缩率 **63.5%** |
| **精度压缩** | INT8动态量化           | 模型大小 72MB → 18MB，压缩率 **75%**       |

**核心价值**：不改变模型结构，不重新训练，仅通过训练后压缩技术，在保持识别精度的前提下实现模型压缩和推理加速。

---

## 📊 实验结果

### LiteASR 论文复现

| 指标           | 结果       |
| -------------- | ---------- |
| 编码器原始维度 | 384        |
| 压缩后维度     | **140**    |
| 压缩率         | **63.5%**  |
| 信息保留率     | **95.0%**  |
| 验证规模       | 7485条音频 |

### INT8 量化压缩

| 指标     | FP32   | INT8       | 提升        |
| -------- | ------ | ---------- | ----------- |
| 模型大小 | 72 MB  | **18 MB**  | ↓ **75%**   |
| 推理时间 | 2.12秒 | **1.16秒** | ↑ **1.83x** |
| 精度损失 | 基准   | **0%**     | ✅ 无损      |

### 噪声鲁棒性验证 (THCHS-30)

| 噪声类型     | 压缩率 | 信息保留率 |
| ------------ | ------ | ---------- |
| ☕ 咖啡馆噪声 | 63.3%  | 95.0%      |
| 🚗 汽车内噪声 | 63.8%  | 95.1%      |
| ⚪ 白噪声     | 63.5%  | 95.0%      |

### 与论文对比

| 对比维度 | LiteASR论文       | 本实验             | 结论         |
| -------- | ----------------- | ------------------ | ------------ |
| 压缩率   | 50%+              | **63.5%**          | ✅ 优于论文   |
| 验证集   | LibriSpeech(英文) | **THCHS-30(中文)** | ✅ 跨语言扩展 |
| 噪声测试 | 无                | **3种真实噪声**    | ✅ 新增验证   |
| 验证规模 | ~1000条           | **7485条**         | ✅ 更大规模   |
| Web部署  | 无                | **✅ 有**           | ✅ 新增贡献   |

---

## 🚀 快速开始

### 环境配置

```bash
# 创建虚拟环境
conda create -n asr python=3.10 -y
conda activate asr

# 安装依赖
pip install torch torchaudio openai-whisper flask scikit-learn
conda install -c conda-forge ffmpeg -y
运行实验

# 1. LiteASR 复现实验
cd 4_data
python batch_test_thchs30.py

# 2. INT8 量化实验
cd ../2_quantization_code
python quantize.py

# 3. 量化方法对比
python compare_quantization.py

# 4. 启动 Web 演示系统
cd ../3_web_demo
python app_final.py
浏览器打开 http://127.0.0.1:5000
```

## 📁 项目结构


LiteASR_Project/
├── 1_liteasr_code/           # LiteASR 复现代码
│   ├── batch_test_noise.py   # 批量测试脚本
│   └── liteasr_compress.py   # 低秩压缩实现
├── 2_quantization_code/      # INT8 量化实验
│   ├── quantize.py           # 量化实验主程序
│   ├── compare_quantization.py # 动态vs静态对比
│   └── calculate_wer.py      # WER/CER 计算
├── 3_web_demo/               # Web 演示系统
│   └── app_final.py          # Flask 主程序
├── 4_data/                   # 数据集（需自行下载）
│   └── thchs30_data/         # THCHS-30 数据集
├── 5_results/                # 实验结果
│   └── quantization_results.png
└── plot_all_results.py       # 图表生成脚本

## 📦 数据集

使用 THCHS-30 噪声测试集：下载地址：http://www.openslr.org/resources/18/test-noise.tgz

| 噪声名称 | 噪声类型   | 数量    |
| -------- | ---------- | ------- |
| cafe     | 咖啡馆噪声 | ~2500条 |
| car      | 汽车内噪声 | ~2500条 |
| white    | 白噪声     | ~2500条 |
| 总计     |            | 7485条  |

## 🎯 创新点

跨语言验证：将论文从英文 LibriSpeech 扩展至中文 THCHS-30

噪声鲁棒性验证：首次在三种真实噪声场景下验证方法有效性

规模化验证：验证规模从 ~1000条 扩展至 7485条

系统集成：将量化方法封装为可交互的 Web 演示系统

