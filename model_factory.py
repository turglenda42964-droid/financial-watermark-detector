import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest, StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin
import xgboost as xgb
import lightgbm as lgb
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Bidirectional
    from tensorflow.keras.optimizers import Adam
except ImportError:
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, LSTM, GRU, Bidirectional
    from keras.optimizers import Adam

class KerasClassifier(BaseEstimator, ClassifierMixin):
    """
    Keras模型的scikit-learn包装器
    """
    
    def __init__(self, model):
        self.model = model
    
    def fit(self, X, y, **kwargs):
        # 重塑输入数据以适应LSTM/GRU模型
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        self.model.fit(X, y, **kwargs)
        return self
    
    def predict(self, X):
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        return (self.model.predict(X) > 0.5).astype(int).flatten()
    
    def predict_proba(self, X):
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        return np.hstack([1 - self.model.predict(X), self.model.predict(X)])

class ModelFactory:
    """模型工厂"""
    
    def __init__(self, config=None):
        """
        初始化模型工厂
        
        参数:
            config: 配置字典
        """
        self.config = config or {}
        self.models = {}
        self.results = {}
    
    def create_model(self, model_type):
        """
        创建模型
        
        参数:
            model_type: 模型类型
            
        返回:
            模型实例
        """
        model_params = self.config.get('model_parameters', {})
        random_state = self.config.get('random_state', 42)
        
        if model_type == 'svm':
            params = model_params.get('svm', {'C': 1.0, 'kernel': 'rbf', 'probability': True})
            return SVC(**params, random_state=random_state)
        elif model_type == 'rf':
            params = model_params.get('rf', {'n_estimators': 100, 'max_depth': 10})
            return RandomForestClassifier(**params, random_state=random_state)
        elif model_type == 'xgb':
            params = model_params.get('xgb', {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1})
            return xgb.XGBClassifier(**params, random_state=random_state)
        elif model_type == 'lgb':
            params = model_params.get('lgb', {'n_estimators': 100, 'max_depth': 7, 'learning_rate': 0.05})
            return lgb.LGBMClassifier(**params, random_state=random_state)
        elif model_type == 'mlp':
            params = model_params.get('mlp', {'hidden_layer_sizes': (100, 50), 'alpha': 0.01, 'max_iter': 500})
            return MLPClassifier(**params, random_state=random_state)
        elif model_type == 'iforest':
            params = model_params.get('iforest', {})
            return IsolationForest(**params, random_state=random_state)
        elif model_type == 'lstm':
            return self._create_lstm_model()
        elif model_type == 'gru':
            return self._create_gru_model()
        elif model_type == 'stacking':
            return self._create_stacking_model()
        else:
            raise ValueError(f"未知的模型类型: {model_type}")
    
    def _create_lstm_model(self):
        """
        创建LSTM模型
        
        返回:
            LSTM模型实例
        """
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=(None, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(32))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return KerasClassifier(model)
    
    def _create_gru_model(self):
        """
        创建GRU模型
        
        返回:
            GRU模型实例
        """
        model = Sequential()
        model.add(GRU(64, return_sequences=True, input_shape=(None, 1)))
        model.add(Dropout(0.2))
        model.add(GRU(32))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return KerasClassifier(model)
    
    def _create_stacking_model(self):
        """
        创建Stacking集成模型
        
        返回:
            Stacking模型实例
        """
        # 基础模型
        base_estimators = [
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=self.config.get('random_state', 42))),
            ('xgb', xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=self.config.get('random_state', 42))),
            ('lgb', lgb.LGBMClassifier(n_estimators=100, max_depth=7, learning_rate=0.05, random_state=self.config.get('random_state', 42)))
        ]
        # 元模型
        meta_estimator = SVC(probability=True, random_state=self.config.get('random_state', 42))
        # 创建Stacking分类器
        stacking_model = StackingClassifier(estimators=base_estimators, final_estimator=meta_estimator, cv=5)
        return stacking_model
    
    def train_models(self, X, y, methods=None):
        """
        训练模型
        
        参数:
            X: 特征数据
            y: 标签数据
            methods: 模型类型列表
            
        返回:
            训练好的模型字典
        """
        if methods is None:
            methods = self.config.get('detection_methods', ['svm', 'rf', 'xgb', 'lgb', 'mlp'])
        
        # 无标签时使用无监督检测
        if y is None:
            methods = ['iforest']
            print("警告: 未提供标签列，改用无监督 IsolationForest 检测水印...")
        else:
            print(f"开始训练 {len(methods)} 个模型...")
        
        # 划分训练集和测试集
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.get('test_size', 0.3), 
                random_state=self.config.get('random_state', 42), 
                stratify=y
            )
        else:
            X_train, X_test = train_test_split(
                X, test_size=self.config.get('test_size', 0.3),
                random_state=self.config.get('random_state', 42)
            )
            y_train, y_test = None, None
        
        trained_models = {}
        
        for method in methods:
            print(f"\n训练 {method.upper()} 模型...")
            
            try:
                # 创建模型
                model = self.create_model(method)
                
                if method == 'iforest':
                    # 无监督学习
                    model.fit(X_train)
                    trained_models[method] = model
                    
                    # 预测
                    y_pred = (model.predict(X_test) == -1).astype(int)  # -1: anomaly
                    scores = model.score_samples(X_test)  # 越大越“正常”
                    anomaly_score = -scores
                    y_pred_proba = self._minmax_scale_to_0_1(anomaly_score)
                    
                    # 无监督场景不计算准确率/召回率
                    self.results[method] = {
                        'model': model,
                        'accuracy': 0.0,
                        'precision': 0.0,
                        'recall': 0.0,
                        'f1': 0.0,
                        'auc': 1.0,
                        'y_pred': y_pred,
                        'y_pred_proba': y_pred_proba
                    }
                elif method in ['lstm', 'gru']:
                    # 深度学习模型
                    if y is not None:
                        # 训练深度学习模型
                        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
                        trained_models[method] = model
                        
                        # 预测
                        y_pred = model.predict(X_test)
                        y_pred_proba = model.predict_proba(X_test)[:, 1]
                        
                        # 计算评估指标
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, zero_division=0)
                        recall = recall_score(y_test, y_pred, zero_division=0)
                        f1 = f1_score(y_test, y_pred, zero_division=0)
                        
                        if len(np.unique(y_test)) > 1:
                            auc_score = roc_auc_score(y_test, y_pred_proba)
                        else:
                            auc_score = 0.5
                        
                        print(f"{method.upper()} 模型训练完成")
                        print(f"  准确率: {accuracy:.4f}")
                        print(f"  精确率: {precision:.4f}")
                        print(f"  召回率: {recall:.4f}")
                        print(f"  F1分数: {f1:.4f}")
                        print(f"  AUC分数: {auc_score:.4f}")
                        
                        self.results[method] = {
                            'model': model,
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'auc': auc_score,
                            'y_pred': y_pred,
                            'y_pred_proba': y_pred_proba
                        }
                else:
                    # 其他监督学习模型
                    if y is not None:
                        model.fit(X_train, y_train)
                        trained_models[method] = model
                        
                        # 预测
                        y_pred = model.predict(X_test)
                        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
                        
                        # 计算评估指标
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, zero_division=0)
                        recall = recall_score(y_test, y_pred, zero_division=0)
                        f1 = f1_score(y_test, y_pred, zero_division=0)
                        
                        if len(np.unique(y_test)) > 1:
                            auc_score = roc_auc_score(y_test, y_pred_proba)
                        else:
                            auc_score = 0.5
                        
                        print(f"{method.upper()} 模型训练完成")
                        print(f"  准确率: {accuracy:.4f}")
                        print(f"  精确率: {precision:.4f}")
                        print(f"  召回率: {recall:.4f}")
                        print(f"  F1分数: {f1:.4f}")
                        print(f"  AUC分数: {auc_score:.4f}")
                        
                        self.results[method] = {
                            'model': model,
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'auc': auc_score,
                            'y_pred': y_pred,
                            'y_pred_proba': y_pred_proba
                        }
            except Exception as e:
                print(f"训练 {method} 模型时出错: {e}")
                continue
        
        self.models = trained_models
        print(f"\n模型训练完成，共训练了 {len(trained_models)} 个模型")
        
        return trained_models
    
    def hyperparameter_tuning(self, X, y, model_type, param_grid):
        """
        超参数调优
        
        参数:
            X: 特征数据
            y: 标签数据
            model_type: 模型类型
            param_grid: 参数网格
            
        返回:
            调优后的模型
        """
        print(f"对 {model_type} 模型进行超参数调优...")
        
        model = self.create_model(model_type)
        
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X, y)
        
        print(f"最佳参数: {grid_search.best_params_}")
        print(f"最佳F1分数: {grid_search.best_score_:.4f}")
        
        best_model = grid_search.best_estimator_
        self.models[model_type] = best_model
        
        return best_model
    
    def ensemble_detection(self, X, weights=None):
        """
        集成检测
        
        参数:
            X: 输入数据
            weights: 模型权重
            
        返回:
            集成预测结果和概率
        """
        if not self.models:
            print("警告: 没有训练好的模型")
            return None, None
        
        print("进行集成检测...")
        
        all_predictions = []
        all_probabilities = []
        
        if weights is None:
            total_auc = sum(self.results[m]['auc'] for m in self.models)
            weights = {m: self.results[m]['auc']/total_auc for m in self.models}
        
        for method, model in self.models.items():
            weight = weights.get(method, 1.0/len(self.models))
            
            if hasattr(model, "score_samples") and not hasattr(model, "predict_proba"):
                # IsolationForest 等无监督模型
                scores = model.score_samples(X)
                anomaly_score = -scores
                proba = self._minmax_scale_to_0_1(anomaly_score)
                pred = (proba >= 0.5).astype(int)
            elif hasattr(model, "predict_proba"):
                # 处理深度学习模型的输入形状
                if method in ['lstm', 'gru']:
                    # 重塑输入数据以适应LSTM/GRU模型
                    if len(X.shape) == 2:
                        X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
                        proba = model.predict_proba(X_reshaped)[:, 1]
                        pred = model.predict(X_reshaped)
                    else:
                        proba = model.predict_proba(X)[:, 1]
                        pred = model.predict(X)
                else:
                    proba = model.predict_proba(X)[:, 1]
                    pred = model.predict(X)
            else:
                pred = model.predict(X)
                proba = pred.astype(float)
            
            all_predictions.append(pred * weight)
            all_probabilities.append(proba * weight)
        
        ensemble_pred = (np.sum(all_predictions, axis=0) >= 0.5).astype(int)
        ensemble_proba = np.sum(all_probabilities, axis=0)
        
        return ensemble_pred, ensemble_proba
    
    def _minmax_scale_to_0_1(self, arr):
        """
        将分数缩放到 [0, 1]
        
        参数:
            arr: 输入数组
            
        返回:
            缩放后的数组
        """
        arr = np.asarray(arr, dtype=float).reshape(-1)
        if arr.size == 0:
            return arr
        vmin = np.nanmin(arr)
        vmax = np.nanmax(arr)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == vmin:
            return np.zeros_like(arr)
        return (arr - vmin) / (vmax - vmin)
    
    def get_model(self, model_type):
        """
        获取模型
        
        参数:
            model_type: 模型类型
            
        返回:
            模型实例
        """
        return self.models.get(model_type)
    
    def get_results(self):
        """
        获取模型结果
        
        返回:
            模型结果字典
        """
        return self.results
