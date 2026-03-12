
---

# SVP-Framework: A Multimodal Deep Learning Approach for Sentiment-Value Prediction

## 📖 简介 (Introduction)

本项目是基于多模态数据融合的**情感-价值预测系统（Sentiment-Value Prediction, SVP Framework）**的核心代码库。本项目旨在探究非结构化金融舆情（散户情感与机构新闻）与 A 股市场收益率之间的非线性动态映射关系。

针对金融时间序列“低信噪比”与“非平稳性”的物理特性，本框架提出了一种混合深度学习架构 **1D-CNN-LSTM-Attention**。系统通过空间降维过滤高频情感噪音，利用 LSTM 建模长程序列依赖，并引入全局注意力机制动态聚焦关键时间步，最终在严苛的**扩展窗口滚动回测（Walk-Forward Backtesting）**中实现了显著的最大回撤控制与超额收益。

## 🏗️ 核心架构 (Core Architecture)

SVP 框架的核心预测模块由以下四个级联组件构成：

1. **Multimodal Input**: 融合量价滞后特征、股吧情感得分（高频散户）与证券时报情感得分（低频机构）。
2. **1D-CNN Layer**: 作为低通滤波器（Low-pass Filter），提取局部空间特征并抑制情感噪音。
3. **LSTM Layer**: 建立跨时间步的隐状态转移方程，捕捉市场记忆效应。
4. **Global Attention**: 自适应评估历史状态对当前决策的贡献度，输出最终的做多/空仓概率。

## 📁 目录结构 (Repository Structure)

为贯彻**关注点分离（Separation of Concerns, SoC）**的软件工程原则，本仓库采用高度模块化设计，覆盖从数据抓取、NLP 情感特征提取到深度学习预测与量化回测的全生命周期：

```text
SVP-Framework/
│
├── data_pipeline/              # 数据流水线 (Data Pipeline)
│   └── (爬虫、清洗与多模态数据对齐脚本)
│
├── nlp_engine/                 # 自然语言处理引擎 (NLP Engine)
│   └── (股吧散户与证券时报机构文本的情感极性提取)
│
├── data/                       # 数据流转中心
│   ├── raw/                    # 原始非结构化文本与量价数据 (Ignored by Git)
│   └── processed/              # 融合后的多模态特征矩阵 (Multimodal CSV)
│
├── models/                     # 深度学习模型库 (Model Architectures)
│   └── (CNN-LSTM-Attention 等网络结构的 PyTorch 定义)
│
├── econometrics/               # 计量经济学检验模块 (Econometrics)
│   └── var_pipeline.py         # VAR 模型因果检验与平稳性分析
│
├── backtest/                   # 核心量化回测引擎 (Backtest Engine)
│   └── engine.py               # 扩展窗口滚动回测框架 (Walk-Forward)
│
├── visualization/              # 学术级图表渲染引擎 (Publication-Ready)
│   ├── generate_all_thesis_figures.py     # 生成英文顶刊标准图表
│   └── generate_all_thesis_figures_cn.py  # 生成中文毕业论文标准图表
│
├── results/                    # 自动生成的英文高清图表集 (Fig 1 - Fig 8)
├── results_cn/                 # 自动生成的中文高清图表集 (Fig 1 - Fig 8)
│
├── requirements.txt            # 环境与第三方库依赖清单
└── README.md                   # 项目说明文档
```
## 📊 核心实证图表 (Key Visualizations)

执行可视化脚本后，系统将自动生成 8 张经过严格样式对齐的学术图表（涵盖计量、训练、回测与消融实验）：

* **Fig 1**: `VAR_IRF` - 情绪对收益率的正交化脉冲响应函数（揭示市场吸收半衰期）。
* **Fig 2**: `Training_Loss` - 模型初始扩展窗口的 BCE Loss 收敛景观。
* **Fig 3**: `Fold_Accuracy` - 滚动回测时间切片的样本外方向准确率（附统计框）。
* **Fig 4**: `Cumulative_PnL` - 叠加最大回撤阴影（Drawdown Zone）的量价实盘净值曲线。
* **Fig 5**: `EDA_Overlay` - 市场价格演变与情绪宏观趋势叠加分析。
* **Fig 6**: `Architecture` - 混合神经网络张量流转与架构示意图。
* **Fig 7**: `Ablation_Study` - 各异质组件对预测准确率的递进贡献分析。
* **Fig 8**: `Sentiment_Source` - 散户舆情与机构新闻的独立/联合预测能力对比。

## 🚀 快速开始 (Quick Start)

### 1. 环境配置 (Environment Setup)

本项目推荐使用 Python 3.10+，核心网络基于 `PyTorch` 构建，并针对 Apple Silicon (M1/M2/M3) 的 MPS 加速进行了底层适配与随机种子锁死，确保实验 100% 可复现。

```bash
# 克隆仓库
git clone https://github.com/jia-sama/SVP_Framework.git
cd SVP_Framework

# 安装依赖
pip install -r requirements.txt

```

### 2. 生成学术图表 (Reproduce Thesis Figures)

确保预处理数据 `LSTM_Multimodal_Dataset_2025.csv` 已放置于 `data/processed/` 目录下。

**生成英文顶刊标准图表：**

```bash
python visualization/generate_all_thesis_figures.py

```

**生成中文毕业论文标准图表：**

```bash
python visualization/generate_all_thesis_figures_cn.py

```

运行结束后，所有高清图表将自动输出至 `results/` 与 `results_cn/` 文件夹。

## 🛠️ 技术栈 (Tech Stack)

* **深度学习**: `PyTorch`, `torch.nn`
* **机器学习/回测基准**: `scikit-learn` (Logistic Regression, StandardScaler)
* **计量经济学**: `statsmodels` (Vector Autoregression, IRF)
* **数据处理**: `pandas`, `numpy`
* **高质量可视化**: `matplotlib` (Customized Academic Palettes)

## 📝 许可证 (License)

This project is licensed under the MIT License - see the LICENSE file for details.

---