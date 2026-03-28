import numpy as np
import pandas as pd
from scipy import signal, fft
from scipy.stats import kurtosis, skew, entropy
from tqdm import tqdm
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    nltk = None
    word_tokenize = None
    stopwords = None
try:
    from gensim.models import Word2Vec, LdaModel
    from gensim.corpora import Dictionary
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    Word2Vec = None
    LdaModel = None
    Dictionary = None

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

# 下载必要的NLTK数据
if NLTK_AVAILABLE:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# 加载spaCy模型
nlp = None
if SPACY_AVAILABLE:
    try:
        nlp = spacy.load('en_core_web_sm')
    except:
        # 如果没有安装，使用简化版本
        nlp = None

class FeatureExtractor:
    """特征提取器"""
    
    def __init__(self, config=None):
        """
        初始化特征提取器
        
        参数:
            config: 配置字典
        """
        self.config = config or {}
        self.feature_names = []
        
    def extract_features(self, X, y=None):
        """
        提取特征
        
        参数:
            X: 输入数据
            y: 标签数据
            
        返回:
            features: 提取的特征
            watermark_features: 水印相关特征
        """
        print("正在提取特征...")
        
        features_list = []
        feature_names = []
        
        # 遍历每个序列提取特征
        for i in tqdm(range(len(X)), desc="提取特征"):
            seq_features = []
            seq_names = []
            
            # 获取序列数据
            if isinstance(X, pd.DataFrame):
                sequence = X.iloc[i].values
            else:
                sequence = X[i]
            
            # 确保序列是数值型
            sequence = self._ensure_numeric_sequence(sequence)
            
            # 1. 统计特征
            stat_features = self._extract_statistical_features(sequence)
            seq_features.extend(stat_features)
            seq_names.extend([
                f'stat_mean', f'stat_std', f'stat_skew', 
                f'stat_kurt', f'stat_min', f'stat_max',
                f'stat_q25', f'stat_q50', f'stat_q75',
                f'stat_iqr', f'stat_mad', f'stat_entropy'
            ])
            
            # 2. 频域特征
            freq_features = self._extract_frequency_features(sequence)
            seq_features.extend(freq_features)
            frequency_bands = self.config.get('frequency_bands', [(0.01, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3)])
            for j, band in enumerate(frequency_bands):
                seq_names.extend([
                    f'freq_power_{band[0]:.2f}-{band[1]:.2f}',
                    f'freq_ratio_{band[0]:.2f}-{band[1]:.2f}',
                    f'freq_peak_{band[0]:.2f}-{band[1]:.2f}'
                ])
            
            # 3. 时域特征
            time_features = self._extract_time_domain_features(sequence)
            seq_features.extend(time_features)
            seq_names.extend([
                f'time_acf1', f'time_acf5', f'time_pacf1',
                f'time_zero_cross', f'time_turning', f'time_volatility',
                f'time_hurst', f'time_lyapunov', f'time_complexity'
            ])
            
            # 4. 相关性特征
            corr_features = self._extract_correlation_features(sequence)
            seq_features.extend(corr_features)
            seq_names.extend([
                f'corr_auto', f'corr_partial', f'corr_detrended',
                f'corr_multifractal', f'corr_nonlinear'
            ])
            
            # 5. 水印特定特征
            watermark_features = self._extract_watermark_specific_features(sequence)
            seq_features.extend(watermark_features)
            seq_names.extend([
                f'wm_amplitude', f'wm_phase', f'wm_frequency',
                f'wm_pattern', f'wm_energy', f'wm_snr'
            ])
            
            # 6. 文本特征（如果需要）
            text_features = self._extract_text_features(sequence)
            seq_features.extend(text_features)
            # 基础文本特征
            seq_names.extend([
                f'text_length', f'text_complexity', f'text_vocab_size'
            ])
            # 词向量特征
            seq_names.extend([f'text_word2vec_{i}' for i in range(10)])
            # 主题模型特征
            seq_names.extend([f'text_lda_topic_{i}' for i in range(3)])
            # 情感分析特征
            seq_names.extend([f'text_sentiment_score', f'text_positive_ratio', f'text_negative_ratio'])
            # 句法分析特征
            seq_names.extend([f'text_noun_count', f'text_verb_count', f'text_adjective_count', f'text_adverb_count', f'text_sentence_count'])
            
            # 处理缺失值
            seq_features = np.nan_to_num(
                np.asarray(seq_features, dtype=float),
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            ).tolist()
            features_list.append(seq_features)
            if i == 0:  # 只记录一次特征名
                self.feature_names = [f'{name}_{i}' for i, name in enumerate(seq_names)]
        
        # 转换为DataFrame
        features_df = pd.DataFrame(features_list, columns=self.feature_names)
        
        # 识别水印相关特征
        watermark_features = self._identify_watermark_features(features_df, y)
        
        print(f"特征提取完成，共提取 {len(self.feature_names)} 个特征")
        print(f"特征矩阵形状: {features_df.shape}")
        
        return features_df, watermark_features
    
    def _ensure_numeric_sequence(self, sequence):
        """
        确保序列是数值型
        
        参数:
            sequence: 输入序列
            
        返回:
            数值型序列
        """
        arr = np.asarray(sequence)
        arr = arr.reshape(-1)

        # 若本身就是数值类型，直接转 float；否则逐元素转成数值（失败则变 NaN）
        if not np.issubdtype(arr.dtype, np.number):
            try:
                arr = pd.to_numeric(arr, errors='coerce')
            except Exception:
                # 极端情况下，逐元素强转
                arr = np.array([pd.to_numeric(x, errors='coerce') for x in arr], dtype=float)

        arr = arr.astype(float, copy=False)
        arr = arr[np.isfinite(arr)]

        # 特征函数依赖至少一个数
        return arr if arr.size > 0 else np.array([0.0], dtype=float)
    
    def _extract_statistical_features(self, sequence):
        """
        提取统计特征
        
        参数:
            sequence: 输入序列
            
        返回:
            统计特征
        """
        sequence = np.asarray(sequence, dtype=float).reshape(-1)
        if sequence.size == 0:
            return [0] * 12

        features = []
        
        # 基本统计量
        features.append(float(np.mean(sequence)))
        features.append(float(np.std(sequence)))
        try:
            features.append(float(skew(sequence)))
        except Exception:
            features.append(0)
        try:
            features.append(float(kurtosis(sequence)))
        except Exception:
            features.append(0)
        features.append(float(np.min(sequence)))
        features.append(float(np.max(sequence)))
        
        # 分位数特征
        features.append(float(np.percentile(sequence, 25)))
        features.append(float(np.percentile(sequence, 50)))
        features.append(float(np.percentile(sequence, 75)))
        features.append(float(np.percentile(sequence, 75) - np.percentile(sequence, 25)))  # IQR
        
        # 绝对中位差
        med = float(np.median(sequence))
        features.append(float(np.median(np.abs(sequence - med))))
        
        # 近似熵（序列复杂度）
        if len(sequence) > 10:
            try:
                features.append(float(entropy(np.histogram(sequence, bins=20)[0])))
            except:
                features.append(0)
        else:
            features.append(0)
            
        return features
    
    def _extract_frequency_features(self, sequence):
        """
        提取频域特征
        
        参数:
            sequence: 输入序列
            
        返回:
            频域特征
        """
        features = []
        
        # 快速傅里叶变换
        n = len(sequence)
        if n < 2:
            frequency_bands = self.config.get('frequency_bands', [(0.01, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3)])
            return [0] * (len(frequency_bands) * 3)
        
        fft_vals = np.abs(fft.fft(sequence - np.mean(sequence))[:n//2])
        freqs = fft.fftfreq(n, d=1)[:n//2]
        
        frequency_bands = self.config.get('frequency_bands', [(0.01, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3)])
        for low, high in frequency_bands:
            # 计算频带内的能量
            mask = (freqs >= low) & (freqs <= high)
            band_power = np.sum(fft_vals[mask]**2) if np.any(mask) else 0
            
            # 总能量
            total_power = np.sum(fft_vals**2)
            
            # 能量比
            power_ratio = band_power / total_power if total_power > 0 else 0
            
            # 峰值频率
            if np.any(mask):
                peak_freq = freqs[mask][np.argmax(fft_vals[mask])]
            else:
                peak_freq = 0
            
            features.extend([band_power, power_ratio, peak_freq])
        
        return features
    
    def _extract_time_domain_features(self, sequence):
        """
        提取时域特征
        
        参数:
            sequence: 输入序列
            
        返回:
            时域特征
        """
        features = []
        n = len(sequence)
        
        if n < 2:
            return [0] * 9
        
        # 自相关特征
        try:
            acf = np.correlate(sequence - np.mean(sequence), 
                             sequence - np.mean(sequence), mode='full')
            acf = acf[acf.size//2:] / acf[acf.size//2]
            features.append(acf[1] if len(acf) > 1 else 0)  # lag1
            features.append(acf[5] if len(acf) > 5 else 0)  # lag5
        except:
            features.extend([0, 0])
        
        # 偏自相关（简化计算）
        try:
            pacf = self._calculate_pacf(sequence, 1)
            features.append(pacf[0] if len(pacf) > 0 else 0)
        except:
            features.append(0)
        
        # 过零率
        zero_crossings = np.sum(np.diff(np.sign(sequence)) != 0)
        features.append(zero_crossings / n)
        
        # 转折点数
        turning_points = np.sum(np.diff(np.sign(np.diff(sequence))) != 0)
        features.append(turning_points / n)
        
        # 波动率
        returns = np.diff(sequence) / sequence[:-1] if np.any(sequence[:-1] != 0) else np.zeros(n-1)
        features.append(np.std(returns) if len(returns) > 1 else 0)
        
        # 赫斯特指数（简化计算）
        try:
            hurst = self._calculate_hurst(sequence)
            features.append(hurst)
        except:
            features.append(0.5)
        
        # 李雅普诺夫指数（近似）
        try:
            lyapunov = self._estimate_lyapunov(sequence)
            features.append(lyapunov)
        except:
            features.append(0)
        
        # 复杂度度量
        features.append(self._calculate_complexity(sequence))
        
        return features
    
    def _extract_correlation_features(self, sequence):
        """
        提取相关性特征
        
        参数:
            sequence: 输入序列
            
        返回:
            相关性特征
        """
        features = []
        n = len(sequence)
        
        if n < 3:
            return [0] * 5
        
        # 自相关性度量
        try:
            diff_seq = np.diff(sequence)
            auto_corr = np.corrcoef(sequence[:-1], sequence[1:])[0, 1]
            features.append(auto_corr)
        except:
            features.append(0)
        
        # 偏自相关
        try:
            partial_corr = self._calculate_partial_correlation(sequence, 2)
            features.append(partial_corr)
        except:
            features.append(0)
        
        # 去趋势波动分析（简化）
        try:
            dfa = self._calculate_dfa(sequence)
            features.append(dfa)
        except:
            features.append(0)
        
        # 多重分形特征（简化）
        try:
            mf = self._calculate_multifractal(sequence)
            features.append(mf)
        except:
            features.append(0)
        
        # 非线性相关性
        try:
            nonlinear = self._calculate_nonlinear_correlation(sequence)
            features.append(nonlinear)
        except:
            features.append(0)
        
        return features
    
    def _extract_watermark_specific_features(self, sequence):
        """
        提取水印特定特征
        
        参数:
            sequence: 输入序列
            
        返回:
            水印特定特征
        """
        sequence = np.asarray(sequence, dtype=float).reshape(-1)
        if sequence.size < 2:
            return [0] * 6

        features = []
        
        # 幅度调制特征
        denom = float(np.mean(np.abs(sequence)))
        amplitude_variation = (float(np.std(np.abs(sequence))) / denom) if denom != 0 else 0
        features.append(amplitude_variation)
        
        # 相位特征
        phase_variation = 0
        freq_variation = 0
        try:
            # hilbert 需要数值序列；对于长度过短的输入，直接跳过
            analytic_signal = signal.hilbert(sequence)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            if instantaneous_phase.size > 2:
                phase_variation = float(np.std(np.diff(instantaneous_phase)))
                instantaneous_frequency = np.diff(instantaneous_phase) / (2 * np.pi)
                if instantaneous_frequency.size > 0:
                    freq_variation = float(np.std(instantaneous_frequency))
        except Exception:
            phase_variation = 0
            freq_variation = 0
        features.append(phase_variation)
        
        # 频率调制特征
        features.append(freq_variation)
        
        # 模式特征
        try:
            mid = len(sequence) // 2
            if mid == 0 or mid == len(sequence):
                pattern_correlation = 0
            else:
                pattern_correlation = np.corrcoef(sequence[:mid], sequence[mid:])[0, 1]
            features.append(pattern_correlation)
        except:
            features.append(0)
        
        # 能量分布
        energy_ratio = float(np.sum(sequence**2) / len(sequence))
        features.append(energy_ratio)
        
        # 信噪比特征
        signal_power = float(np.mean(sequence**2))
        noise_power = float(np.mean((sequence - np.convolve(sequence, np.ones(5)/5, mode='same'))**2))
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 and signal_power > 0 else 0
        features.append(snr)
        
        return features
    
    def _extract_text_features(self, sequence):
        """
        提取文本特征
        
        参数:
            sequence: 输入序列
            
        返回:
            文本特征
        """
        features = []
        text = str(sequence)
        
        # 1. 基础文本特征
        # 文本长度
        text_length = len(text)
        features.append(text_length)
        
        # 文本复杂度（字符多样性）
        if len(text) > 0:
            unique_chars = len(set(text))
            complexity = unique_chars / len(text)
        else:
            complexity = 0
        features.append(complexity)
        
        # 词汇量大小
        words = text.split()
        vocab_size = len(set(words)) if len(words) > 0 else 0
        features.append(vocab_size)
        
        # 2. 词向量特征
        word2vec_features = self._extract_word2vec_features(text)
        features.extend(word2vec_features)
        
        # 3. 主题模型特征
        lda_features = self._extract_lda_features(text)
        features.extend(lda_features)
        
        # 4. 情感分析特征
        sentiment_features = self._extract_sentiment_features(text)
        features.extend(sentiment_features)
        
        # 5. 句法分析特征
        syntax_features = self._extract_syntax_features(text)
        features.extend(syntax_features)
        
        return features
    
    def _extract_word2vec_features(self, text):
        """
        提取词向量特征
        
        参数:
            text: 文本
            
        返回:
            词向量特征
        """
        try:
            if NLTK_AVAILABLE and word_tokenize and stopwords and GENSIM_AVAILABLE and Word2Vec:
                # 简单的词向量特征提取
                words = word_tokenize(text.lower())
                stop_words = set(stopwords.words('english'))
                words = [word for word in words if word.isalnum() and word not in stop_words]
                
                if len(words) > 1:
                    # 训练简单的Word2Vec模型
                    model = Word2Vec([words], vector_size=10, window=2, min_count=1, workers=1)
                    # 获取词向量平均值
                    vectors = [model.wv[word] for word in words if word in model.wv]
                    if vectors:
                        avg_vector = np.mean(vectors, axis=0)
                        return avg_vector.tolist()
        except:
            pass
        # 如果失败，返回0向量
        return [0] * 10
    
    def _extract_lda_features(self, text):
        """
        提取主题模型特征
        
        参数:
            text: 文本
            
        返回:
            主题模型特征
        """
        try:
            if NLTK_AVAILABLE and word_tokenize and stopwords and GENSIM_AVAILABLE and LdaModel and Dictionary:
                # 简单的LDA主题特征提取
                words = word_tokenize(text.lower())
                stop_words = set(stopwords.words('english'))
                words = [word for word in words if word.isalnum() and word not in stop_words]
                
                if len(words) > 1:
                    # 创建词典和语料库
                    dictionary = Dictionary([words])
                    corpus = [dictionary.doc2bow(words)]
                    
                    # 训练LDA模型
                    lda_model = LdaModel(corpus, num_topics=3, id2word=dictionary, passes=1)
                    
                    # 获取文档主题分布
                    topic_dist = lda_model.get_document_topics(corpus[0], minimum_probability=0)
                    topic_dist = [prob for _, prob in sorted(topic_dist, key=lambda x: x[0])]
                    # 确保返回3个主题的概率
                    while len(topic_dist) < 3:
                        topic_dist.append(0)
                    return topic_dist[:3]
        except:
            pass
        # 如果失败，返回0向量
        return [0, 0, 0]
    
    def _extract_sentiment_features(self, text):
        """
        提取情感分析特征
        
        参数:
            text: 文本
            
        返回:
            情感分析特征
        """
        try:
            if NLTK_AVAILABLE and word_tokenize:
                # 简单的情感分析
                words = word_tokenize(text.lower())
                # 情感词典（简化版）
                positive_words = set(['good', 'great', 'excellent', 'positive', 'happy', 'success', 'profit', 'gain'])
                negative_words = set(['bad', 'poor', 'terrible', 'negative', 'sad', 'failure', 'loss', 'losses'])
                
                positive_count = sum(1 for word in words if word in positive_words)
                negative_count = sum(1 for word in words if word in negative_words)
                
                # 计算情感得分
                total_words = len(words)
                if total_words > 0:
                    sentiment_score = (positive_count - negative_count) / total_words
                    positive_ratio = positive_count / total_words
                    negative_ratio = negative_count / total_words
                else:
                    sentiment_score = 0
                    positive_ratio = 0
                    negative_ratio = 0
                
                return [sentiment_score, positive_ratio, negative_ratio]
        except:
            pass
        # 如果失败，返回0值
        return [0, 0, 0]
    
    def _extract_syntax_features(self, text):
        """
        提取句法分析特征
        
        参数:
            text: 文本
            
        返回:
            句法分析特征
        """
        try:
            if SPACY_AVAILABLE and nlp:
                doc = nlp(text)
                
                # 计算各种词性的数量
                pos_counts = {}
                for token in doc:
                    pos = token.pos_
                    pos_counts[pos] = pos_counts.get(pos, 0) + 1
                
                # 提取关键句法特征
                noun_count = pos_counts.get('NOUN', 0)
                verb_count = pos_counts.get('VERB', 0)
                adjective_count = pos_counts.get('ADJ', 0)
                adverb_count = pos_counts.get('ADV', 0)
                
                # 计算句子数量
                sentence_count = len(list(doc.sents))
                
                return [noun_count, verb_count, adjective_count, adverb_count, sentence_count]
        except:
            pass
        # 如果失败，返回0值
        return [0, 0, 0, 0, 0]
    
    def _identify_watermark_features(self, features, y):
        """
        识别水印相关特征
        
        参数:
            features: 特征矩阵
            y: 标签数据
            
        返回:
            水印相关特征
        """
        if y is None or len(np.unique(y)) < 2:
            return {}
        
        watermark_features = {}
        
        # 计算每个特征与水印标签的相关性
        for col in features.columns:
            try:
                if features[col].dtype in ['float64', 'int64']:
                    corr = np.corrcoef(features[col], y)[0, 1]
                    if not np.isnan(corr) and abs(corr) > 0.1:
                        watermark_features[col] = {
                            'correlation': corr,
                            'importance': abs(corr)
                        }
            except:
                continue
        
        # 按重要性排序
        watermark_features = dict(sorted(watermark_features.items(), 
                                       key=lambda x: abs(x[1]['importance']), 
                                       reverse=True)[:20])
        
        return watermark_features
    
    def _calculate_pacf(self, series, lag):
        """
        计算偏自相关
        
        参数:
            series: 输入序列
            lag: 滞后阶数
            
        返回:
            偏自相关值
        """
        if len(series) <= lag:
            return [0]
        
        pacf = []
        for k in range(1, lag + 1):
            if len(series) > k:
                try:
                    pacf.append(np.corrcoef(series[k:], series[:-k])[0, 1])
                except:
                    pacf.append(0)
            else:
                pacf.append(0)
        return pacf
    
    def _calculate_hurst(self, series):
        """
        计算赫斯特指数
        
        参数:
            series: 输入序列
            
        返回:
            赫斯特指数
        """
        lags = range(2, min(20, len(series)//2))
        tau = []
        
        for lag in lags:
            n = len(series) // lag
            if n < 2:
                continue
            rs = []
            for i in range(n):
                segment = series[i*lag:(i+1)*lag]
                if len(segment) > 1:
                    mean = np.mean(segment)
                    cumsum = np.cumsum(segment - mean)
                    r = np.max(cumsum) - np.min(cumsum)
                    s = np.std(segment)
                    if s > 0:
                        rs.append(r / s)
            
            if rs:
                tau.append(np.mean(rs))
            else:
                tau.append(0)
        
        if len(tau) > 1:
            lags_log = np.log(lags[:len(tau)])
            tau_log = np.log(tau)
            hurst, _ = np.polyfit(lags_log, tau_log, 1)
            return hurst
        return 0.5
    
    def _estimate_lyapunov(self, series, embedding_dim=2, delay=1):
        """
        估计李雅普诺夫指数
        
        参数:
            series: 输入序列
            embedding_dim: 嵌入维度
            delay: 延迟
            
        返回:
            李雅普诺夫指数
        """
        n = len(series)
        if n < embedding_dim + 1:
            return 0
        
        embedded = np.array([series[i:i+embedding_dim] 
                           for i in range(0, n-embedding_dim+1, delay)])
        
        if len(embedded) < 2:
            return 0
        
        distances = []
        for i in range(len(embedded)-1):
            diff = np.linalg.norm(embedded[i+1:] - embedded[i], axis=1)
            if len(diff) > 0:
                distances.append(np.min(diff[diff > 0]))
        
        if len(distances) > 1:
            distances = np.array(distances)
            lyapunov = np.mean(np.log(distances[1:]/distances[:-1]))
            return lyapunov
        return 0
    
    def _calculate_complexity(self, series):
        """
        计算序列复杂度
        
        参数:
            series: 输入序列
            
        返回:
            复杂度
        """
        if len(series) < 3:
            return 0
        
        threshold = np.median(series)
        binary = (series > threshold).astype(int)
        
        patterns = []
        for i in range(len(binary)-2):
            pattern = tuple(binary[i:i+3])
            patterns.append(pattern)
        
        unique_patterns = len(set(patterns))
        complexity = unique_patterns / len(patterns) if patterns else 0
        
        return complexity
    
    def _calculate_partial_correlation(self, series, lag):
        """
        计算偏自相关
        
        参数:
            series: 输入序列
            lag: 滞后阶数
            
        返回:
            偏自相关值
        """
        if len(series) < lag + 2:
            return 0
        
        try:
            from sklearn.linear_model import LinearRegression
            
            X = np.column_stack([series[i:-(lag-i)] for i in range(lag)])
            y = series[lag:]
            
            if len(X) < lag + 1:
                return 0
            
            model = LinearRegression()
            model.fit(X, y)
            partial_corr = model.coef_[-1] if len(model.coef_) > 0 else 0
            
            return partial_corr
        except:
            return 0
    
    def _calculate_dfa(self, series):
        """
        计算去趋势波动分析指数
        
        参数:
            series: 输入序列
            
        返回:
            DFA指数
        """
        n = len(series)
        if n < 4:
            return 0.5
        
        scales = [4, 8, 16, 32, 64]
        scales = [s for s in scales if s <= n//4]
        
        if len(scales) < 2:
            return 0.5
        
        fluctuations = []
        
        for scale in scales:
            n_segments = n // scale
            
            if n_segments < 1:
                continue
            
            f2 = 0
            for v in range(n_segments):
                segment = series[v*scale:(v+1)*scale]
                
                if len(segment) < 2:
                    continue
                
                y = np.cumsum(segment - np.mean(segment))
                
                x = np.arange(len(y))
                coeffs = np.polyfit(x, y, 1)
                trend = np.polyval(coeffs, x)
                y_detrended = y - trend
                
                f2 += np.mean(y_detrended**2)
            
            if n_segments > 0:
                f2 /= n_segments
                fluctuations.append(np.sqrt(f2))
            else:
                fluctuations.append(0)
        
        if len(fluctuations) > 1:
            fluctuations = np.array(fluctuations)
            valid_idx = (fluctuations > 0) & (~np.isnan(fluctuations))
            if np.sum(valid_idx) >= 2:
                scales_log = np.log(np.array(scales)[valid_idx])
                fluct_log = np.log(fluctuations[valid_idx])
                alpha, _ = np.polyfit(scales_log, fluct_log, 1)
                return alpha
        
        return 0.5
    
    def _calculate_multifractal(self, series):
        """
        计算多重分形特征
        
        参数:
            series: 输入序列
            
        返回:
            多重分形特征
        """
        if len(series) < 10:
            return 0
        
        q_values = [-5, -2, 2, 5]
        tau_q = []
        
        for q in q_values:
            if q == 0:
                continue
            
            fluctuations = []
            scales = [4, 8, 16]
            
            for scale in scales:
                if len(series) >= scale * 4:
                    n_segments = len(series) // scale
                    fq = 0
                    
                    for v in range(n_segments):
                        segment = series[v*scale:(v+1)*scale]
                        if len(segment) > 1:
                            fq += (np.std(segment) ** q)
                    
                    if n_segments > 0:
                        fq /= n_segments
                        fluctuations.append(fq ** (1/q) if q != 0 else np.exp(0.5 * np.mean(np.log([np.std(s) for s in segment]))))
            
            if len(fluctuations) >= 2:
                scales_log = np.log(scales[:len(fluctuations)])
                fluct_log = np.log(fluctuations)
                hq, _ = np.polyfit(scales_log, fluct_log, 1)
                tau_q.append(hq * q - 1)
        
        if len(tau_q) >= 2:
            h_range = max(tau_q) - min(tau_q)
            return h_range
        
        return 0
    
    def _calculate_nonlinear_correlation(self, series, lag=1):
        """
        计算非线性相关性
        
        参数:
            series: 输入序列
            lag: 滞后阶数
            
        返回:
            非线性相关值
        """
        if len(series) < lag + 10:
            return 0
        
        n_bins = min(20, len(series) // 10)
        
        hist, x_edges, y_edges = np.histogram2d(series[:-lag], series[lag:], bins=n_bins)
        
        pxy = hist / hist.sum()
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)
        
        pxy = pxy[pxy > 0]
        px_py = np.outer(px, py)
        px_py = px_py[px_py > 0]
        
        if len(pxy) > 0 and len(px_py) > 0:
            mi = np.sum(pxy * np.log(pxy / px_py[:len(pxy)]))
            return mi
        else:
            return 0
