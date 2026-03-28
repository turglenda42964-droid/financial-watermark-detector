import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

class Preprocessor:
    """数据预处理器"""
    
    def __init__(self, config=None):
        """
        初始化数据预处理器
        
        参数:
            config: 配置字典
        """
        self.config = config or {}
        self.scaler = StandardScaler()
        self.pca = None
        self.feature_selector = None
        self.scaler_fitted = False
        self.selected_features = None
        self.feature_columns = None
    
    def preprocess(self, features, y=None):
        """
        预处理数据
        
        参数:
            features: 特征数据
            y: 标签数据（用于特征选择）
            
        返回:
            X_processed: 处理后的数据
        """
        print("数据预处理中...")
        
        # 处理缺失值
        features = self._handle_missing_values(features)
        
        # 特征选择
        if y is not None:
            features = self._select_features(features, y)
        
        # 特征缩放
        X_scaled = self._scale_features(features)
        
        # 降维
        X_processed = self._reduce_dimension(X_scaled)
        
        print(f"预处理完成，特征维度: {X_processed.shape[1]}")
        
        return X_processed
    
    def _handle_missing_values(self, features):
        """
        处理缺失值
        
        参数:
            features: 特征数据
            
        返回:
            处理后的数据
        """
        # 使用均值填充缺失值
        features = features.fillna(features.mean())
        return features
    
    def _scale_features(self, features):
        """
        特征缩放
        
        参数:
            features: 特征数据
            
        返回:
            缩放后的数据
        """
        if hasattr(self, 'scaler_fitted') and self.scaler_fitted:
            X_scaled = self.scaler.transform(features)
        else:
            X_scaled = self.scaler.fit_transform(features)
            self.scaler_fitted = True
        return X_scaled
    
    def _reduce_dimension(self, X):
        """
        降维
        
        参数:
            X: 输入数据
            
        返回:
            降维后的数据
        """
        n_components = self.config.get('feature_parameters', {}).get('n_components', 0)
        
        if n_components > 0:
            n_components = min(n_components, X.shape[1])
            if n_components < X.shape[1]:
                if self.pca is None:
                    self.pca = PCA(n_components=n_components, 
                                  random_state=self.config.get('random_state', 42))
                    X_processed = self.pca.fit_transform(X)
                else:
                    X_processed = self.pca.transform(X)
                
                explained_variance = np.sum(self.pca.explained_variance_ratio_)
                print(f"PCA降维: {X.shape[1]} -> {n_components} 维")
                print(f"解释方差: {explained_variance:.2%}")
            else:
                X_processed = X
        else:
            X_processed = X
        
        return X_processed
    
    def _select_features(self, features, y):
        """
        特征选择
        
        参数:
            features: 特征数据
            y: 标签数据
            
        返回:
            选择后的特征
        """
        feature_selection_method = self.config.get('feature_selection', 'none')
        k = self.config.get('feature_parameters', {}).get('k_features', 50)
        
        # 首先进行特征交叉组合
        features_with_cross = self._create_feature_crosses(features)
        
        # 保存特征列名
        self.feature_columns = features_with_cross.columns.tolist()
        
        if feature_selection_method == 'kbest':
            print(f"使用SelectKBest选择前 {k} 个特征...")
            self.feature_selector = SelectKBest(f_classif, k=k)
            X_selected = self.feature_selector.fit_transform(features_with_cross, y)
            self.selected_features = features_with_cross.columns[self.feature_selector.get_support()].tolist()
            print(f"选择的特征: {self.selected_features}")
            return pd.DataFrame(X_selected, columns=self.selected_features)
        elif feature_selection_method == 'mutual_info':
            print(f"使用互信息选择前 {k} 个特征...")
            self.feature_selector = SelectKBest(mutual_info_classif, k=k)
            X_selected = self.feature_selector.fit_transform(features_with_cross, y)
            self.selected_features = features_with_cross.columns[self.feature_selector.get_support()].tolist()
            print(f"选择的特征: {self.selected_features}")
            return pd.DataFrame(X_selected, columns=self.selected_features)
        else:
            self.selected_features = features_with_cross.columns.tolist()
            return features_with_cross
    
    def _create_feature_crosses(self, features):
        """
        创建特征交叉组合
        
        参数:
            features: 特征数据
            
        返回:
            包含交叉特征的特征数据
        """
        print("创建特征交叉组合...")
        
        # 选择重要特征进行交叉组合
        # 这里选择前20个特征进行交叉
        n_features = min(20, features.shape[1])
        selected_features = features.columns[:n_features]
        
        # 创建交叉特征
        cross_features = []
        cross_feature_names = []
        
        # 创建特征对的交叉组合
        for i in range(len(selected_features)):
            for j in range(i+1, len(selected_features)):
                feature1 = selected_features[i]
                feature2 = selected_features[j]
                
                # 乘法交叉
                cross_name = f"{feature1}_x_{feature2}"
                cross_features.append(features[feature1] * features[feature2])
                cross_feature_names.append(cross_name)
                
                # 除法交叉（避免除零）
                cross_name_div = f"{feature1}_div_{feature2}"
                cross_features.append(features[feature1] / (features[feature2] + 1e-8))
                cross_feature_names.append(cross_name_div)
        
        # 使用pd.concat避免DataFrame碎片化
        if cross_features:
            cross_df = pd.DataFrame(np.column_stack(cross_features), columns=cross_feature_names, index=features.index)
            features_with_cross = pd.concat([features, cross_df], axis=1)
        else:
            features_with_cross = features.copy()
        
        print(f"特征交叉组合完成，新增 {features_with_cross.shape[1] - features.shape[1]} 个交叉特征")
        return features_with_cross
    
    def transform(self, features):
        """
        对新数据进行预处理
        
        参数:
            features: 新的特征数据
            
        返回:
            处理后的数据
        """
        # 处理缺失值
        features = self._handle_missing_values(features)
        
        # 创建特征交叉组合
        features_with_cross = self._create_feature_crosses(features)
        
        # 确保特征列名与训练时一致
        if self.feature_columns:
            # 添加缺失的特征列并填充为0
            for col in self.feature_columns:
                if col not in features_with_cross.columns:
                    features_with_cross[col] = 0
            # 按照训练时的顺序排列特征
            features_with_cross = features_with_cross[self.feature_columns]
        
        # 特征选择
        if self.feature_selector is not None and self.selected_features:
            features_with_cross = features_with_cross[self.selected_features]
        
        # 特征缩放
        X_scaled = self.scaler.transform(features_with_cross)
        
        # 降维
        if self.pca is not None:
            X_processed = self.pca.transform(X_scaled)
        else:
            X_processed = X_scaled
        
        return X_processed
