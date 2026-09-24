# IN6227 Assignment 1 — Variant 1

本项目比较 **CART 决策树** 和 **L2 正则化逻辑回归**，另外提供多数类预测作为基线。代码按作业要求组织完整分类流程，重点是能够解释每一步，而不是追求最高准确率。

## 运行

建议 Python 3.12。在本文件所在目录打开终端：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py --data-dir "D:\dataset"
```

本机已经准备好父目录的环境，也可在 `D:\msis\data mining` 直接运行：

```powershell
.\.venv\Scripts\python.exe assignment_variant1\train.py --data-dir "D:\dataset"
```

默认结果写入代码旁的 `outputs` 文件夹。`--output-dir` 可设置其他结果目录，`--jobs` 可设置交叉验证并行数，默认 1，便于普通电脑稳定运行。数据无需复制进项目，也不必上传到 GitHub。

## 分类流程及理由

1. **探索数据**：记录行列数、类别比例、缺失值、重复行、类别基数、数值分布和 IQR 异常值数量。训练数据有 7 个数值特征、8 个类别特征，目标为 `label`，yes 映射为 1。
2. **清洗**：去除类别字符串首尾空白，空字符串及非有限数值视为缺失。标签缺失的样本不填补、不参加监督学习或评估；训练集完全重复的行在交叉验证前去重。测试集保留原始有效标签行，不因预测结果而清洗。
3. **特征处理**：数值缺失用训练折的中位数填补；类别缺失用训练折的众数填补。类别用 one-hot 编码，不臆造顺序；未知类别使用全零编码。逻辑回归的数值特征做标准化，决策树不需要标准化。所有变换都放入 Pipeline，在每个训练折内部拟合。
4. **异常值与特征选择**：IQR 仅用来标记和观察，缺少数据字典时不能断定极端值是错误，因此保留。保留全部 15 个原始特征，不根据字段名称猜测重要性，也不根据测试集做筛选。逻辑回归用 L2 正则化约束系数；决策树通过分裂选择特征。这不等同于执行了独立的特征选择实验。
5. **训练与调参**：固定随机种子 42，训练集上使用相同的 5 折分层交叉验证，以 yes 类 F1 选择超参数及推荐模型。逻辑回归搜索 `C=[0.01,0.1,1,10]`；决策树搜索 `max_depth=[3,5,8,12,None]`、`min_samples_leaf=[5,20,50]`。两者都比较 `class_weight=None` 与 `balanced`。权重按训练折计算，不预先对完整数据过采样。
6. **停止条件**：逻辑回归使用 lbfgs，`tol=1e-4`、`max_iter=3000`，收敛警告会使运行失败而不是被隐藏。树使用 Gini 分裂，并通过深度和叶节点最少样本限制生长；纯节点或无有效分裂也会停止。调参后在全部有效训练样本上重新拟合。
7. **评估**：调参结束后才加载测试集并进行最终比较。报告 accuracy、balanced accuracy、precision、recall、F1、ROC-AUC、average precision、混淆矩阵和 ROC/PR 曲线。yes 为正类；使用分类器默认决策规则，不在测试集上调阈值。average precision 是阶梯加权精确率，不称其为梯形 PR-AUC。

## 为什么选择这两个模型

- 决策树对应 Lecture 4 的 Gini、贪心分裂、过拟合及预剪枝；能捕捉非线性和交互，树结构可解释，但复杂树容易过拟合。
- 逻辑回归对应 Lecture 2 的模型导向预处理及线性模型讨论；提供可解释的线性基准，适合检验类别编码和标准化后的可分性。它并不假定原始特征与标签呈线性关系，而是对正类 log-odds 建立线性关系。
- 数据有类别不平衡，因此以正类 F1 调参，并同时报告 precision/recall，说明取舍；多数类基线用于展示单看 accuracy 的局限。

课件逐项核对及适用边界见 `COURSE_AUDIT.md`。逻辑回归只在 Lecture 2 中被提及，Lecture 5 详细介绍的是 KNN、规则、Bayes 和集成模型。训练后可运行 `python check_coursework.py --data-dir "D:\dataset"`，补充类别频数、相关矩阵、按标签箱线图和散点图，并独立核对保存的指标与模型预测。该脚本不重新训练或选择模型。

## 输出文件

| 文件 | 用途 |
|---|---|
| `model_comparison.csv` | 两个模型及多数类基线的最终指标 |
| `data_audit.json` | 缺失、重复、异常值、类别比例、训练/测试特征重合检查 |
| `tuning.json` | 最佳参数、交叉验证均值与标准差、推荐模型、实际迭代次数/树大小 |
| `*_cv_results.csv` | 每组参数的交叉验证结果及训练分数 |
| `*_classification_report.json` | 两个类别各自的 precision/recall/F1 |
| `test_predictions.csv` | 原 CSV 行号、真实标签、预测标签及 yes 概率 |
| `*.joblib` | 两个完整模型，包含预处理 |
| `*_features.csv` | 编码后特征的回归系数/树不纯度重要性 |
| `data_exploration.png` | 训练数据分布 |
| `confusion_matrices.png`、`roc_pr_curves.png` | 模型评估图 |
| `tree_preview.png` | 树的前三层，并非完整树 |
| `reproducibility.json` | 软件版本、数据文件 SHA-256 |

系数和树重要性反映模型关联，不代表因果关系；one-hot 特征也会分散同一个原始字段的重要性。交叉验证最优分数存在调参选择偏差，最终测试结果用于独立比较；折间标准差不是置信区间。一次固定划分不能证明模型在其他数据上仍然领先。字段来源未知，无法确认是否存在时间/实体相关性或语义层面的目标泄漏。

## 提交说明

你这次要求的是代码，因此交付包括源代码、已训练模型及实验结果。作业最终另外要求不超过两页的 PDF 报告、姓名、学号、`IN6227-Assignment-1`、`Variant-1`，以及源码 GitHub 链接。此项目尚未上传 GitHub，也未代填个人信息。可以根据 `RESULTS.md` 和实际输出编写报告，理解并核对后再提交。
