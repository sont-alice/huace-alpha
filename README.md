---
title: Huace Alpha Research
sdk: docker
app_port: 8501
pinned: false
license: mit
---

# A 股波段预测推荐软件

本项目是一个本地 Web 投研工具，用于 A 股 2-6 周波段推荐研究。系统先拉取真实行情，构建技术面、资金面、行业和基本面特征，再训练模型、回测验证，并输出按综合评估从高到低排列的候选清单。

当前版本只在产品界面展示真实数据结果，不提供演示模式。公开接口不可用时，系统会提示错误或使用合格的最近真实缓存，不会生成占位推荐。

## 数据源

- AKShare：默认数据源，拉取前复权日线、上市信息、行业字段和财务指标。
- Tushare Pro：填写 token 并勾选优先使用后启用，拉取日线、日行情基础指标和财务指标。
- 本地缓存：真实数据会缓存到 `data/cache/`，同一天相同参数优先读取缓存。

## 功能

- 推荐列表：最多扫描 3000 只真实股票，按模型综合评分从高到低输出候选股。
- 个股评估：输入股票代码后，强制加入该股票，展示评级、胜率评分、模型分位、技术面、基本面、风险标签和近一年走势。
- 板块筛选：支持上证主板、深证主板、创业板、科创板。
- 回测证据：展示样本外资金曲线、胜率、回撤和收益回撤比。
- 数据源状态：展示 AKShare/Tushare 可用性、数据行数和训练/测试切分。

## 安装

```powershell
python -m pip install -e .
```

## 启动

```powershell
streamlit run app.py
```

## 测试

```powershell
pytest
```

## 公网部署

公网版使用 Hugging Face Docker Space 运行，只读取已验证的每日快照，不会因为访客点击而重新训练 3000 股模型。

生成本地快照：

```powershell
python scripts/build_snapshot.py --output data/snapshot --max-symbols 3000
```

使用本地快照测试公网模式：

```powershell
$env:PUBLIC_SNAPSHOT_MODE = "true"
$env:SNAPSHOT_DIR = "data/snapshot"
streamlit run app.py
```

Docker 启动：

```powershell
docker build -t huace-alpha .
docker run --rm -p 8501:8501 `
  -e HF_SNAPSHOT_REPO_ID="<account>/<dataset>" `
  huace-alpha
```

Hugging Face Space 需要设置变量 `HF_SNAPSHOT_REPO_ID`。GitHub Actions 需要设置同名 Repository Variable，并将可写入 Dataset 仓库的 token 保存为 `HF_TOKEN` Secret。`TUSHARE_TOKEN` 是可选 Secret。

## 重要说明

软件仅用于量化研究和辅助决策，不构成投资建议，不承诺收益或固定胜率。公开数据接口可能受网络、频次、权限和字段变动影响；界面会显示当前数据源状态。
