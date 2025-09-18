# 通用疾病预测项目 — 答辩PPT大纲

1. 封面
   - 题目：通用疾病预测与落地应用
   - 团队与时间：驰星教育 | 2025-09-04
   - 关键词：CatBoost、SMOTE、Flask API、零漏诊

2. 项目概览与目标
   - 背景：医疗场景早筛与预警，对“零漏诊”要求高
   - 目标：提高健康/疾病识别准确度，尤其降低健康人群漏诊率；AUC≥0.99
   - 落地：Flask API + 简易前端（心脏病表单页）+ 模型自动加载

3. 数据与问题定义
   - 数据源：`data_set/thyroidDF.csv`（甲状腺）、`data_set/heart.csv`（心脏病）等
   - 任务：二分类（健康 vs. 需关注/疾病）
   - 难点：类别极度不平衡（报告示例：约12:1）

4. 特征工程与清洗
   - 数值特征：缺失填充/类型转换（如 TSH/T3/T4、Age 等）
   - 类别特征：二值化与 One-Hot（如 referral source、ChestPainType/RestingECG/ST_Slope）
   - 训练/推理特征对齐：严格按 expected_features 顺序输入

5. 技术路线与模型选择
   - 基线与对比：CatBoost/LightGBM/XGBoost 尝试；网格/Optuna 调参
   - 关键：SMOTE 过采样 + CatBoost 管道；必要时阈值策略（普通模型 0.75）
   - 产物管理：自动选择 `balanced_models` 最新模型，标准化持久化（joblib/pickle/cbm）

6. 训练与验证流程
   - 数据分层拆分 + 交叉验证
   - 指标：Accuracy、Precision、Recall、F1、AUC，关注“健康人群零漏诊”
   - 泄漏防护：仅在训练折内拟合编码与采样；推理阶段复用同一流水线

7. 结果与性能（以最新报告为例）
   - 文件：`balanced_models/performance_report_20250904_100033.txt`
   - 指标：Acc≈0.984、少数类Recall=1.00、Precision≈0.812、F1≈0.897、AUC≈0.998
   - 提升：Recall +19.2%、F1 +12.6%，关键成就：健康人群零漏诊

8. 系统架构与实现
   - 服务：`app.py`（甲状腺预测，端点：/、/predict、/model_info、/reload_model）
   - 服务：`app_heart.py`（心脏病预测，端点：/、/predict、/model_info；端口5001）
   - 数据库：心脏病预测结果写入 MySQL（表 heart_predictions）
   - 前端：`app_heart.html` 简易表单页；启用 CORS

9. 演示方案（现场）
   - 启动服务（两个 Flask 实例，5000/5001）
   - 打开主页与表单页，提交一组样例查看预测/概率/风险分级
   - 查看 `/model_info` 返回模型文件与时间戳；心脏病结果入库演示

1. 项目亮点

- 类不平衡解决：SMOTE + 阈值策略；健康人群零漏诊
- 工程化：自动加载最新模型、接口清晰、前后端即用
- 可扩展：多数据集通用化、可对接 Explain（SHAP）与更多模型融合

1. 风险与不足

- 数据代表性与泛化风险；依赖与环境一致性
- 数据安全与隐私（数据库凭据、日志）
- 改进：更丰富数据、特征选择/重要性、模型集成与在线监控

1. 规划与展望

- 引入解释性与可视化大屏；上线 A/B 与阈值自动优化
- 部署容器化/CI/CD；隐私合规与审计

1. 总结与 Q&A

- 复盘目标、方法、结果与价值；开放问答
