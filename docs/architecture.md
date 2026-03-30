这个项目秉承着结构清晰的可复现的简单结构流程
1.下载金融数据来在kaggele经典文本式数据集
2. 添加了轻量的词汇水印文本（有监督学习）
3. 提取了文本特征，转化为机器所读的样本
4. 逻辑回归训练特征，区分水印
5. 产出报告生成可视化图表

## 模型采用工业界常规思维（高内聚，低耦合）

- `src/financial_watermark_detector/data.py`
 数据处理（数据后面可以用更多真实、多元的金融文本）
- `src/financial_watermark_detector/watermarking.py`
嵌入水印，对比结构（水平有限后面升级，词汇级替换为llm生成式水印，研究自适应水印强度）
- `src/financial_watermark_detector/detector.py`
 定义特征, 逻辑回归训练, 特征图（水平有限后面可升级，检测器的鲁棒性和不可感知性方面，研究金融文本安全的领域自适应水印方案，类似于套一个金融的壳子）
- `src/financial_watermark_detector/pipeline.py`
  统筹串联各个模块

## Output Artifacts

- `models/watermark_detector.joblib`
- `data/watermark_detection_dataset.csv`
- `reports/corpus_summary.json`
- `reports/watermark_metrics.json`
- `reports/watermark_feature_importance.png`
- `reports/watermark_samples.json`
就做了一个比较完整的文本水印检测流程，虽然黑箱。首先对金融文本进行词汇级水印嵌入，构建带标签的数据集，然后通过特征工程提取统计特征，并使用逻辑回归模型进行分类检测。整个系统采用流水线结构，支持实验复现和结果分析。但目前模型比较简单（自己太菜了），水印策略也较基础，后面可升级。
