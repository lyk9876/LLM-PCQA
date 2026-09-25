# QNet：使用原 Test 替代 Validation 的泄漏实验

## 实验设置

- Train：原 720 个训练 patch，不变。
- Selection Validation：原 7-shape Test，共 168 个 patch，用于学习率调度、early stopping 和 best checkpoint 选择。
- Holdout Test：原 6-shape Validation，共 144 个 patch，训练和 checkpoint 选择均不访问。
- Best epoch：61。
- 本实验有意使用 Test 选择模型，属于数据泄漏消融，相关数值不能作为无偏 Test 性能报告。

## Test-as-Validation checkpoint 的结果

| 数据集合 | 样本数 | PLCC ↑ | SRCC ↑ | KRCC ↑ | MSE ↓ | MAE ↓ |
|---|---:|---:|---:|---:|---:|---:|
| 原 7-shape Test（已用于选模型） | 168 | **0.9098** | **0.8361** | **0.6860** | **0.0741** | **0.2012** |
| 其中 3-shape seen overlap | 72 | **0.9110** | **0.8773** | **0.7365** | **0.0743** | **0.1937** |
| 其中 4-shape unseen | 96 | **0.9089** | **0.8067** | **0.6532** | **0.0740** | **0.2067** |
| 原 6-shape Validation（untouched holdout） | 144 | **0.7386** | **0.7497** | **0.6071** | **0.1890** | **0.3098** |

## 与正常 Validation 选模 QNet 对照

| 数据集合 | 模型 | PLCC ↑ | SRCC ↑ | KRCC ↑ | MSE ↓ |
|---|---|---:|---:|---:|---:|
| 7-shape Test 混合 | 正常 Val 选模 | 0.8882 | **0.8465** | **0.6993** | 0.0911 |
| 7-shape Test 混合 | Test-as-Val | **0.9098** | 0.8361 | 0.6860 | **0.0741** |
| 3-shape overlap | 正常 Val 选模 | 0.8481 | 0.8398 | 0.6921 | 0.1239 |
| 3-shape overlap | Test-as-Val | **0.9110** | **0.8773** | **0.7365** | **0.0743** |
| 4-shape unseen | 正常 Val 选模 | **0.9214** | **0.8546** | **0.7118** | **0.0665** |
| 4-shape unseen | Test-as-Val | 0.9089 | 0.8067 | 0.6532 | 0.0740 |
| 原 6-shape Validation | 正常 Val 选模 | **0.7625** | **0.7709** | **0.6236** | **0.1645** |
| 原 6-shape Validation | Test-as-Val | 0.7386 | 0.7497 | 0.6071 | 0.1890 |

## 结论

使用原 Test 选择 checkpoint，确实显著提高了与 Train 重复的 3-shape overlap 指标：PLCC 从 0.8481 提升到 0.9110。但它没有使指标接近 1，并且：

- 7-shape 混合 Test 的 PLCC 和 MSE改善，但 SRCC、KRCC下降；
- 4 个未见 shape 的 PLCC、SRCC、KRCC、MSE全部退化；
- untouched 的原 Validation holdout 也全面退化。

因此，这个 checkpoint 更偏向被用于选模的 Test 分布，不能替代正常 Validation 选模的 QNet，也不应作为正式 PF 联合训练的默认教师。
