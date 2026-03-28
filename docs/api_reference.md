# API 参考文档

## 核心类

### WatermarkDetector

主检测器类，整合所有功能。

#### 构造函数

```python
WatermarkDetector(
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[str] = None,
)
```

**参数：**
- `config`: 配置字典
- `config_path`: 配置文件路径（YAML格式）

**示例：**
```python
from src.detector import WatermarkDetector

# 使用默认配置
detector = WatermarkDetector()

# 使用配置文件
detector = WatermarkDetector(config_path='configs/default_config.yaml')

# 使用配置字典
config = {
    'model': {'type': 'xgboost'},
    'features': {'window_size': 60}
}
detector = WatermarkDetector(config=config)
```

#### 方法

##### train

```python
train(
    data_path: Union[str, Path, pd.DataFrame],
    validation_split: float = 0.2,
) -> 'WatermarkDetector'
```

训练模型。

**参数：**
- `data_path`: 数据路径或DataFrame
- `validation_split`: 验证集比例

**返回：**
- self（支持链式调用）

**示例：**
```python
# 从文件训练
detector.train('data/train.csv')

# 从DataFrame训练
detector.train(df, validation_split=0.3)
```

##### detect

```python
detect(
    data_path: Union[str, Path, pd.DataFrame],
) -> DetectionResult
```

执行水印检测。

**参数：**
- `data_path`: 数据路径或DataFrame

**返回：**
- `DetectionResult`: 检测结果对象

**示例：**
```python
result = detector.detect('data/test.csv')
print(result.summary())
```

##### evaluate

```python
evaluate(
    data_path: Union[str, Path, pd.DataFrame],
) -> Dict[str, Any]
```

评估模型性能。

**参数：**
- `data_path`: 数据路径或DataFrame

**返回：**
- 评估指标字典

**示例：**
```python
metrics = detector.evaluate('data/test.csv')
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

##### load_model

```python
load_model(model_path: Union[str, Path]) -> 'WatermarkDetector'
```

加载预训练模型。

**参数：**
- `model_path`: 模型文件路径

**示例：**
```python
detector.load_model('models/xgboost.pkl')
```

---

### DetectionResult

检测结果类。

#### 属性

- `predictions`: 预测标签数组
- `probabilities`: 预测概率数组
- `feature_importance`: 特征重要性字典
- `metadata`: 元数据字典

#### 方法

##### summary

```python
summary() -> str
```

生成结果摘要。

**返回：**
- 摘要字符串

**示例：**
```python
print(result.summary())
```

##### to_dataframe

```python
to_dataframe() -> pd.DataFrame
```

转换为DataFrame格式。

**返回：**
- 包含预测结果的DataFrame

##### save

```python
save(output_dir: Union[str, Path]) -> None
```

保存结果到目录。

**参数：**
- `output_dir`: 输出目录路径

---

### DataLoader

数据加载器类。

#### 构造函数

```python
DataLoader(
    time_column: str = "timestamp",
    value_columns: Optional[List[str]] = None,
    freq: Optional[str] = None,
)
```

#### 方法

##### load

```python
load(path: Union[str, Path], **kwargs) -> pd.DataFrame
```

加载单个数据文件。

**支持格式：**
- CSV (.csv)
- Excel (.xlsx, .xls)
- Parquet (.parquet)
- Pickle (.pkl)

##### load_directory

```python
load_directory(
    directory: Union[str, Path],
    pattern: str = "*.csv",
    **kwargs
) -> pd.DataFrame
```

加载目录中的所有匹配文件。

---

### FeatureExtractor

特征提取器类。

#### 构造函数

```python
FeatureExtractor(
    window_size: int = 60,
    step_size: int = 10,
    feature_groups: Optional[List[str]] = None,
)
```

**参数：**
- `window_size`: 滑动窗口大小
- `step_size`: 滑动步长
- `feature_groups`: 启用的特征组列表

**可用特征组：**
- `'statistical'`: 统计特征
- `'time_domain'`: 时域特征
- `'frequency_domain'`: 频域特征
- `'complexity'`: 复杂度特征
- `'watermark_specific'`: 水印特定特征

#### 方法

##### extract

```python
extract(
    df: pd.DataFrame,
    target_column: str = 'close',
) -> pd.DataFrame
```

提取特征。

##### fit_transform

```python
fit_transform(
    df: pd.DataFrame,
    target_column: str = 'close',
) -> np.ndarray
```

拟合并转换数据（包含标准化）。

##### transform

```python
transform(
    df: pd.DataFrame,
    target_column: str = 'close',
) -> np.ndarray
```

转换数据（使用已拟合的scaler）。

---

### ModelFactory

模型工厂类。

#### 构造函数

```python
ModelFactory(
    model_type: str = 'xgboost',
    model_params: Optional[Dict[str, Any]] = None,
)
```

**支持的模型类型：**
- `'xgboost'`: XGBoost分类器
- `'lightgbm'`: LightGBM分类器
- `'random_forest'`: 随机森林
- `'gradient_boosting'`: 梯度提升
- `'logistic_regression'`: 逻辑回归
- `'svm'`: 支持向量机

#### 方法

##### fit

```python
fit(X: np.ndarray, y: np.ndarray, **fit_params) -> 'ModelFactory'
```

训练模型。

##### predict

```python
predict(X: np.ndarray) -> np.ndarray
```

预测类别。

##### predict_proba

```python
predict_proba(X: np.ndarray) -> np.ndarray
```

预测概率。

##### get_feature_importance

```python
get_feature_importance(
    feature_names: Optional[List[str]] = None
) -> Dict[str, float]
```

获取特征重要性。

##### save / load

```python
save(path: Union[str, Path]) -> None
load(path: Union[str, Path]) -> 'ModelFactory'
```

保存和加载模型。

---

### Evaluator

评估器类。

#### 构造函数

```python
Evaluator(output_dir: Optional[str] = None)
```

#### 方法

##### evaluate

```python
evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    model_name: str = "model",
) -> Dict[str, Any]
```

评估模型性能。

##### plot_confusion_matrix

```python
plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
) -> None
```

绘制混淆矩阵。

##### plot_roc_curve

```python
plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Optional[str] = None,
) -> None
```

绘制ROC曲线。

##### plot_feature_importance

```python
plot_feature_importance(
    importance_dict: Dict[str, float],
    top_n: int = 20,
    save_path: Optional[str] = None,
) -> None
```

绘制特征重要性图。

---

## 工具函数

### 可视化工具

#### plot_time_series

```python
plot_time_series(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    title: str = "Time Series",
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None,
) -> None
```

绘制时序图。

#### plot_watermark_detection_result

```python
plot_watermark_detection_result(
    df: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    value_column: str = 'close',
    figsize: tuple = (15, 8),
    save_path: Optional[str] = None,
) -> None
```

绘制水印检测结果。

### 辅助函数

#### setup_logger

```python
setup_logger(
    name: Optional[str] = None,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    verbose: bool = False,
) -> logging.Logger
```

设置日志记录器。

#### ensure_dir

```python
ensure_dir(path: Union[str, Path]) -> Path
```

确保目录存在。

---

## 配置参考

### 完整配置示例

```yaml
# 数据配置
data:
  format: csv
  time_column: timestamp
  value_column: close
  label_column: watermark_label
  freq: null

# 特征配置
features:
  window_size: 60
  step_size: 10
  feature_groups:
    - statistical
    - time_domain
    - frequency_domain
    - complexity
    - watermark_specific

# 模型配置
model:
  type: xgboost
  params:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1

# 检测配置
detection:
  threshold: 0.5
  batch_size: 1000

# 输出配置
output:
  save_features: true
  save_model: true
```

---

## 异常处理

所有主要方法都可能抛出以下异常：

- `FileNotFoundError`: 文件不存在
- `ValueError`: 参数值错误
- `RuntimeError`: 运行时错误（如模型未训练）
- `KeyError`: 列名不存在

建议在使用时添加适当的异常处理：

```python
try:
    result = detector.detect('data.csv')
except FileNotFoundError:
    print("数据文件不存在")
except RuntimeError as e:
    print(f"运行时错误: {e}")
```
