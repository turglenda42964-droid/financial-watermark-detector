import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from datetime import datetime

class Evaluator:
    """模型评估器"""
    
    def __init__(self, config=None):
        """
        初始化评估器
        
        参数:
            config: 配置字典
        """
        self.config = config or {}
        # 创建可视化目录
        os.makedirs('可视化', exist_ok=True)
    
    def evaluate_models(self, models, results, y_test=None, detailed=True):
        """
        评估模型
        
        参数:
            models: 模型字典
            results: 模型结果字典
            y_test: 测试标签
            detailed: 是否输出详细评估报告
            
        返回:
            评估结果
        """
        if not models:
            print("警告: 没有训练好的模型")
            return None
        
        # 无监督模式没有真实标签，跳过监督评估与可视化
        if y_test is None:
            print("警告: 无标签数据，跳过监督评估与可视化")
            evaluation_results = {}
            for method, result in results.items():
                evaluation_results[method] = {
                    'accuracy': result.get('accuracy', 0.0),
                    'precision': result.get('precision', 0.0),
                    'recall': result.get('recall', 0.0),
                    'f1': result.get('f1', 0.0),
                    'auc': result.get('auc', 0.0)
                }
            return evaluation_results
        
        print("模型评估结果:")
        print("=" * 80)
        
        evaluation_results = {}
        
        for method, result in results.items():
            print(f"\n{method.upper()} 模型:")
            print(f"  准确率: {result['accuracy']:.4f}")
            print(f"  精确率: {result['precision']:.4f}")
            print(f"  召回率: {result['recall']:.4f}")
            print(f"  F1分数: {result['f1']:.4f}")
            print(f"  AUC分数: {result['auc']:.4f}")
            
            evaluation_results[method] = {
                'accuracy': result['accuracy'],
                'precision': result['precision'],
                'recall': result['recall'],
                'f1': result['f1'],
                'auc': result['auc']
            }
            
            if detailed:
                print("\n  分类报告:")
                print(classification_report(y_test, result['y_pred'], 
                                          target_names=['无水印', '有水印']))
        
        # 绘制ROC曲线
        self.plot_roc_curves(results, y_test)
        
        # 绘制混淆矩阵
        self.plot_confusion_matrices(results, y_test)
        
        return evaluation_results
    
    def plot_roc_curves(self, results, y_test):
        """
        绘制ROC曲线
        
        参数:
            results: 模型结果字典
            y_test: 测试标签
        """
        plt.figure(figsize=(10, 8))
        
        for method, result in results.items():
            if 'y_pred_proba' in result:
                fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, 
                        label=f'{method.upper()} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Watermark Detection Models')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.savefig('可视化/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrices(self, results, y_test):
        """
        绘制混淆矩阵
        
        参数:
            results: 模型结果字典
            y_test: 测试标签
        """
        n_models = len(results)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        if n_models == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, (method, result) in enumerate(results.items()):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            cm = confusion_matrix(y_test, result['y_pred'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f'{method.upper()} Confusion Matrix')
            ax.set_xticklabels(['No WM', 'WM'])
            ax.set_yticklabels(['No WM', 'WM'])
        
        for idx in range(len(results), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('可视化/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, watermark_features, top_n=20):
        """
        绘制特征重要性
        
        参数:
            watermark_features: 水印相关特征
            top_n: 显示前N个特征
        """
        if not watermark_features:
            print("没有可用的水印特征重要性数据")
            return
        
        sorted_features = sorted(watermark_features.items(), 
                               key=lambda x: abs(x[1]['importance']), 
                               reverse=True)[:top_n]
        
        features = [f[0] for f in sorted_features]
        importances = [f[1]['importance'] for f in sorted_features]
        correlations = [f[1]['correlation'] for f in sorted_features]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        colors = ['red' if corr > 0 else 'blue' for corr in correlations]
        y_pos = np.arange(len(features))
        
        ax1.barh(y_pos, importances, color=colors)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(features)
        ax1.set_xlabel('Feature Importance')
        ax1.set_title(f'Top {top_n} Watermark-related Features')
        ax1.invert_yaxis()
        
        ax2.scatter(correlations, importances, alpha=0.6)
        for i, (feature, corr, imp) in enumerate(zip(features, correlations, importances)):
            ax2.annotate(feature.split('_')[0], (corr, imp), fontsize=8, alpha=0.7)
        
        ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Correlation with Watermark')
        ax2.set_ylabel('Feature Importance')
        ax2.set_title('Feature Correlation vs Importance')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('可视化/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, detector, output_path='watermark_detection_report.txt'):
        """
        生成检测报告
        
        参数:
            detector: 检测器实例
            output_path: 输出路径
            
        返回:
            报告内容
        """
        report = []
        report.append("=" * 80)
        report.append("金融时序数据水印检测报告")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        report.append("")
        
        report.append("1. 数据概况")
        report.append("-" * 40)
        if hasattr(detector, 'X'):
            report.append(f"样本数量: {len(detector.X)}")
            if hasattr(detector, 'feature_extractor') and hasattr(detector.feature_extractor, 'feature_names'):
                report.append(f"特征数量: {len(detector.feature_extractor.feature_names)}")
        if hasattr(detector, 'y') and detector.y is not None:
            watermark_count = np.sum(detector.y)
            total_count = len(detector.y)
            report.append(f"水印样本: {watermark_count}/{total_count} ({watermark_count/total_count*100:.2f}%)")
        report.append("")
        
        report.append("2. 模型性能对比")
        report.append("-" * 40)
        if hasattr(detector, 'model_factory') and hasattr(detector.model_factory, 'results') and detector.model_factory.results:
            headers = ["模型", "准确率", "精确率", "召回率", "F1分数", "AUC"]
            report.append(" ".join(f"{h:>10}" for h in headers))
            report.append("-" * 80)
            
            for method, result in detector.model_factory.results.items():
                row = [
                    f"{method:>10}",
                    f"{result['accuracy']:>10.4f}",
                    f"{result['precision']:>10.4f}",
                    f"{result['recall']:>10.4f}",
                    f"{result['f1']:>10.4f}",
                    f"{result['auc']:>10.4f}"
                ]
                report.append(" ".join(row))
        report.append("")
        
        report.append("3. 水印相关特征重要性 (Top 10)")
        report.append("-" * 40)
        if hasattr(detector, 'watermark_features') and detector.watermark_features:
            sorted_features = sorted(detector.watermark_features.items(), 
                                   key=lambda x: abs(x[1]['importance']), 
                                   reverse=True)[:10]
            
            report.append(f"{'特征':<30} {'重要性':<12} {'相关性':<12}")
            report.append("-" * 60)
            for feature, info in sorted_features:
                report.append(f"{feature:<30} {info['importance']:<12.4f} {info['correlation']:<12.4f}")
        report.append("")
        
        report.append("4. 水印检测结果汇总")
        report.append("-" * 40)
        if hasattr(detector, 'X_test') and hasattr(detector, 'y_test') and detector.y_test is not None:
            if hasattr(detector, 'model_factory') and hasattr(detector.model_factory, 'results'):
                for method, result in detector.model_factory.results.items():
                    cm = confusion_matrix(detector.y_test, result['y_pred'])
                    tn, fp, fn, tp = cm.ravel()
                    
                    report.append(f"\n{method.upper()} 模型:")
                    report.append(f"  真阴性(TN): {tn:>5}  假阳性(FP): {fp:>5}")
                    report.append(f"  假阴性(FN): {fn:>5}  真阳性(TP): {tp:>5}")
                    report.append(f"  特异性(Specificity): {tn/(tn+fp):.4f}")
                    report.append(f"  FPR: {fp/(fp+tn):.4f}, FNR: {fn/(fn+tp):.4f}")
        
        report.append("\n5. 结论与建议")
        report.append("-" * 40)
        if hasattr(detector, 'model_factory') and hasattr(detector.model_factory, 'results') and detector.model_factory.results:
            best_model = max(detector.model_factory.results.items(), key=lambda x: x[1]['f1'])[0]
            best_auc = max(detector.model_factory.results.items(), key=lambda x: x[1]['auc'])[0]
            
            report.append(f"推荐模型: {best_model.upper()} (基于F1分数)")
            report.append(f"最佳AUC模型: {best_auc.upper()} (基于AUC)")
            report.append("")
            
            if detector.model_factory.results[best_model]['f1'] > 0.8:
                report.append("模型性能优秀，可用于实际水印检测。")
            elif detector.model_factory.results[best_model]['f1'] > 0.6:
                report.append("模型性能良好，建议结合其他方法进行验证。")
            else:
                report.append("模型性能一般，建议优化特征工程或尝试其他算法。")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report))
        
        print(f"检测报告已保存到: {output_path}")
        
        print("\n" + "\n".join(report[:50]) + "\n...")
        
        return report
