------

## 🔹 1. 背景

在二分类或多分类问题里，常见情况是 **正负样本比例严重不平衡**（比如欺诈检测、医学诊断，正样本比例可能 <1%）。

- 如果直接训练模型，容易导致模型偏向多数类。
- 这时候就需要**重采样方法**来平衡数据：
  - **欠采样（Under-sampling）**：减少多数类样本。
  - **过采样（Over-sampling）**：增加少数类样本。

------

## 🔹 2. SMOTE 的核心思想

**普通过采样**（如复制少数类样本）容易导致过拟合。
 **SMOTE** 的改进：不是简单复制，而是**合成新的少数类样本**。

具体做法：

1. 对于每一个少数类样本 xx，在其 **k 个近邻（少数类）**中随机选择一个邻居 x_{nn}

2. 在这两个点之间生成一个新的合成样本：
   $$
   x_{new}=x+λ×(x_{nn}−x),λ∈[0,1]
   $$
   

   也就是在样本和邻居之间的连线上随机取一点。

3. 重复上述过程，直到少数类样本数量达到预期比例。

------

## 🔹 3. 优点

- 解决了普通过采样重复样本导致过拟合的问题。
- 让少数类的**分布更“扩展”**，不再局限在原始点上。
- 效果通常优于简单复制。

------

## 🔹 4. 缺点

- 可能在类边界附近生成“错误的”样本，导致噪声增加。
- 对高维数据可能不太稳定。
- 没有考虑多数类的分布，容易在重叠区域引入混淆。

------

## 🔹 5. Python 实现（使用 imbalanced-learn）

```python
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 构造不平衡数据集
X, y = make_classification(n_classes=2, class_sep=2,
                           weights=[0.9, 0.1], n_informative=3, n_redundant=1,
                           flip_y=0, n_features=20, n_clusters_per_class=1,
                           n_samples=1000, random_state=42)

print("采样前:", Counter(y))

# SMOTE 过采样
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

print("采样后:", Counter(y_res))
```

输出：

```
采样前: Counter({0: 900, 1: 100})
采样后: Counter({0: 900, 1: 900})
```

------

## 🔹 6. 变种方法

- **Borderline-SMOTE**：只在边界附近生成样本。
- **SMOTEENN / SMOTETomek**：结合欠采样和清理噪声。
- **ADASYN**：改进 SMOTE，在难以学习的区域生成更多样本。

------

简单来说：**SMOTE 是在少数类样本之间插值生成新样本的过采样方法**，效果通常比简单复制要好。

