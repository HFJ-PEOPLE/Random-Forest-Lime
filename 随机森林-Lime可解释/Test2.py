import pandas as pd
import numpy as np
# from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_curve, auc, precision_recall_curve, confusion_matrix,
    ConfusionMatrixDisplay, precision_score, recall_score, f1_score,
    roc_auc_score, balanced_accuracy_score, matthews_corrcoef,
    cohen_kappa_score, log_loss, average_precision_score,
    brier_score_loss, fbeta_score, jaccard_score, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interp
from itertools import cycle

#######################python 3.8(DB606)############################
####Test2比Test1多了一些对模型的评估指标###########

# 加载 combined_data.csv 文件
file_path = 'data_save/Zn-data.csv'
df = pd.read_csv(file_path, header=None)

###########################################################
# # 假设第一列是标签，剩余的列是 (x, y) 对形式的特征
# y = df.iloc[:, -1]  # 标签（第一列）
# X = df.iloc[:, 1:]  # 特征（剩余列）
#
# # 将每行的 (x, y) 点依次连接为一条线
# X_lines = []
# for idx in range(len(X)):
#     points = X.iloc[idx].values.reshape(-1, 2)  # 将一行数据分割为多个 (x, y) 点
#     lines = []
#     for i in range(len(points) - 1):
#         line_segment = np.concatenate([points[i], points[i + 1]])
#         lines.append(line_segment)
#     X_lines.append(np.concatenate(lines))
#
# X_lines = np.array(X_lines)
#
# # 确认特征名列表长度与特征数据一致
# num_segments = X_lines.shape[1] // 4
# feature_names = [
#     f'line{(i // 4) + 1} x {["start", "end"][(i % 4) // 2]}, line{(i // 4) + 1} y {["start", "end"][i % 2]}' for i in
#     range(X_lines.shape[1])]
#
# assert len(feature_names) == X_lines.shape[1], "特征名列表长度与特征数据不匹配，请检查特征名列表的生成过程。"
# # 随机划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X_lines, y, test_size=0.3, random_state=42)

###############################################################
# #最后一列为标签，每列一个特征
y = pd.to_numeric(df.iloc[1:, 1], errors='coerce').astype(int)  # 转为整数，标签（最后一列），跳过第一行
X_processed = df.iloc[1:, 2:].values  # 特征（其他列），跳过第一行，第一列和最后一行
#print(f"特征前5行前5列: {X_processed[:5, :5]}")
# 生成特征名（从1开始编号，因为跳过了第一列）
feature_names = [f'Feature_{i}' for i in range(1, X_processed.shape[1]+1)]

# 随机划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=42)
##################################################################

# 训练 RandomForestClassifier 模型
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 保存所有测试样本的预测结果
predictions = model.predict(X_test)
y_proba = model.predict_proba(X_test)  # 获取预测概率


######################### 综合模型评估指标 ############################
def evaluate_model(y_true, y_pred, y_proba, classes):
    """
    计算多种模型评估指标并返回结果字典
    基础分类指标:
    精确率 (Precision)含义：预测为正类的样本中实际为正类的比例,公式：TP/(TP+FP),范围：0~1
    召回率 (Recall)含义：实际为正类的样本中被正确预测的比例,公式：TP/(TP+FN),范围：0~1
    准确率 (Accuracy)含义：模型预测正确的样本比例,公式：(TP+TN)/(TP+TN+FP+FN),范围：0~1;>0.9：极好（平衡数据集）;0.7~0.9：可接受 ;<0.6：需改进
    平衡准确率 (Balanced Accuracy)含义：各类别召回率的平均值，解决数据不平衡问题,范围：0~1; >0.8：优秀；0.6~0.8：中等；<0.5：差于随机猜测
    F1分数 (F1-Score)含义：精确率和召回率的调和平均数含义：精确率和召回率的调和平均数,公式：2(PrecisionRecall)/(Precision+Recall),范围：0~1： >0.8：优秀；0.6~0.8：中等；<0.5：需优化
    概率评估指标:
    ROC AUC含义：模型区分正负样本的能力，与阈值无关,范围：0.5~1：0.9~1.0：极强区分能力；0.7~0.9：有用区分度；0.5~0.7：几乎无效；<0.5：反向预测
    对数损失 (Log Loss)含义：预测概率与实际标签的差异惩罚值,范围：0~∞（完美=0）二分类参考：<0.1：极好；0.1~0.3：良好；>0.5：差
    布赖尔分数 (Brier Score)含义：概率预测的均方误差,范围：0~1（完美=0）:   <0.1：概率预测极准;0.1~0.3：可用 ；>0.4：不可靠
    一致性指标:
    马修斯相关系数 (MCC)含义：综合所有混淆矩阵元素的平衡指标,范围：-1~1：  >0.5：强相关 ；0.2~0.5：弱相关 ；<0：反向预测
    科恩卡帕系数 (Cohen's Kappa)含义：评估模型预测与随机预测的一致性,范围：-1~1 ：>0.8：几乎完全一致；0.6~0.8：显著一致；<0.4：一致性差
    Jaccard相似系数含义：预测正确的正样本占所有相关样本的比例,范围：0~1：>0.7：高度相似；0.4~0.7：中等相似 ；<0.3：差异大
    多分类扩展指标:
    Fβ分数 (F2-Score)β=2：更重视召回率,判断标准同F1，但阈值可降低5%~10%

    分类报告：
    指标	           公式	                    含义	                                            判断标准
    Precision	TP / (TP + FP)	   预测为正类的样本中实际为正类的比例（"不冤枉好人"的能力）	       >0.7可接受，>0.9优秀
    Recall	    TP / (TP + FN)	   实际为正类的样本中被正确预测的比例（"不漏掉坏人"的能力）	       >0.8可接受，>0.9优秀
    F1-Score	2(PrecisionRecall)/(Precision+Recall)	精确率和召回率的调和平均数（综合衡量）   >0.7可接受，>0.85优秀
    Support	       -	            该类别的真实样本数量	                                   检查类别是否平衡
    weighted avg：按样本量加权的平均（大类别影响更大）
    若两者差异大（如macro=0.7, weighted=0.9），说明小类别表现差。

    """
    results = {}

    # 基本分类指标
    results['accuracy'] = accuracy_score(y_true, y_pred)
    results['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)

    # 二分类特定指标
    if len(classes) == 2:
        # 二分类指标
        results['precision_binary'] = precision_score(y_true, y_pred, average='binary', pos_label=classes[1])
        results['recall_binary'] = recall_score(y_true, y_pred, average='binary', pos_label=classes[1])
        results['f1_binary'] = f1_score(y_true, y_pred, average='binary', pos_label=classes[1])

        if y_proba is not None:
            results['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            results['average_precision'] = average_precision_score(y_true, y_proba[:, 1])
            results['brier_score'] = brier_score_loss(y_true, y_proba[:, 1])
    else:
        # 多分类指标
        if y_proba is not None:
            try:
                results['roc_auc_ovr'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
                results['roc_auc_ovo'] = roc_auc_score(y_true, y_proba, multi_class='ovo')
            except:
                pass

    # 其他通用指标
    results['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    results['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)

    if y_proba is not None:
        try:
            results['log_loss'] = log_loss(y_true, y_proba)
        except:
            pass

    # Jaccard相似系数
    results['jaccard_macro'] = jaccard_score(y_true, y_pred, average='macro')

    # F-beta分数（平衡精确率和召回率）
    results['f2_score'] = fbeta_score(y_true, y_pred, beta=2, average='macro')

    # 分类报告（包含每个类别的指标）
    results['classification_report'] = classification_report(y_true, y_pred, target_names=[str(c) for c in classes])

    return results


# 创建保存评估结果的函数
def save_metrics_to_txt(metrics, file_path='data_save/model_evaluation_metrics.txt'):
    """
    将评估指标保存到TXT文件
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("模型综合评估指标\n")
        f.write("=" * 50 + "\n\n")

        # 写入基础指标
        f.write("基础分类指标:\n")
        f.write("-" * 50 + "\n")
        f.write(f"准确率 (Accuracy): {metrics['accuracy']:.4f}\n")
        f.write(f"平衡准确率 (Balanced Accuracy): {metrics['balanced_accuracy']:.4f}\n")

        if 'precision_binary' in metrics:
            f.write(f"精确率 (Precision): {metrics['precision_binary']:.4f}\n")
            f.write(f"召回率 (Recall): {metrics['recall_binary']:.4f}\n")
            f.write(f"F1分数: {metrics['f1_binary']:.4f}\n")

        # 写入概率评估指标
        f.write("\n概率评估指标:\n")
        f.write("-" * 50 + "\n")
        if 'roc_auc' in metrics:
            f.write(f"ROC AUC: {metrics['roc_auc']:.4f}\n")
        if 'average_precision' in metrics:
            f.write(f"平均精确率 (Average Precision): {metrics['average_precision']:.4f}\n")
        if 'brier_score' in metrics:
            f.write(f"布赖尔分数 (Brier Score): {metrics['brier_score']:.4f}\n")
        if 'log_loss' in metrics:
            f.write(f"对数损失 (Log Loss): {metrics['log_loss']:.4f}\n")

        # 写入一致性指标
        f.write("\n一致性指标:\n")
        f.write("-" * 50 + "\n")
        f.write(f"马修斯相关系数 (MCC): {metrics['matthews_corrcoef']:.4f}\n")
        f.write(f"科恩卡帕系数 (Cohen's Kappa): {metrics['cohen_kappa']:.4f}\n")
        f.write(f"Jaccard相似系数 (Macro): {metrics['jaccard_macro']:.4f}\n")
        f.write(f"F2分数: {metrics['f2_score']:.4f}\n")

        # 写入分类报告
        f.write("\n分类报告 (Classification Report):\n")
        f.write("-" * 50 + "\n")
        f.write(metrics['classification_report'])

        # 写入指标解释
        f.write("\n\n" + "=" * 50 + "\n")
        f.write("指标解释\n")
        f.write("=" * 50 + "\n")
        f.write("""
基础分类指标:
- 准确率 (Accuracy): 模型预测正确的样本比例
- 平衡准确率 (Balanced Accuracy): 各类别召回率的平均值，解决数据不平衡问题
- 精确率 (Precision): 预测为正类的样本中实际为正类的比例
- 召回率 (Recall): 实际为正类的样本中被正确预测的比例
- F1分数: 精确率和召回率的调和平均数

概率评估指标:
- ROC AUC: 模型区分正负样本的能力，与阈值无关
- 平均精确率 (Average Precision): 精确率-召回率曲线下的面积
- 布赖尔分数 (Brier Score): 概率预测的均方误差
- 对数损失 (Log Loss): 预测概率与实际标签的差异惩罚值

一致性指标:
- 马修斯相关系数 (MCC): 综合所有混淆矩阵元素的平衡指标
- 科恩卡帕系数 (Cohen's Kappa): 评估模型预测与随机预测的一致性
- Jaccard相似系数: 预测正确的正样本占所有相关样本的比例
- F2分数: 更重视召回率的F-beta分数(beta=2)
""")

# 获取类别列表
classes = np.unique(y_train)

# 计算所有评估指标
metrics = evaluate_model(y_test, predictions, y_proba, classes)

# 保存评估指标到TXT文件
save_metrics_to_txt(metrics)
print("\n所有评估指标已保存到 model_evaluation_metrics.txt 文件.")


# 计算测试准确度
accuracy = accuracy_score(y_test, predictions)
print(f"\n模型在测试集上的准确度为: {accuracy:.4f}")

# 将测试结果和准确度保存到文件
results_df = pd.DataFrame({'Prediction': predictions, 'True Label': y_test})
results_df['Accuracy'] = accuracy  # 添加准确度列
results_df.to_csv('data_save/model_predictions.csv', index=False)
print(f"模型预测结果和准确度已保存到 model_predictions.csv 文件.")

##########################获得整体数据集上每个线段特征的相对重要性#####################
# 获取特征重要性
feature_importance = model.feature_importances_

# 只选择重要性排名前20的特征进行可视化
top_feature_indices = np.argsort(feature_importance)[::-1][:20]
top_feature_names = [feature_names[idx] for idx in top_feature_indices]
top_feature_importance = feature_importance[top_feature_indices]

# 可视化全局特征重要性前20的特征
plt.figure(figsize=(10, 6))
plt.barh(top_feature_names, top_feature_importance, align='center')
plt.xlabel('Feature Importance')
plt.title('Top 20 Global Feature Importance')
plt.gca().invert_yaxis()  # 反转 y 轴，使得重要性高的特征显示在顶部
plt.tight_layout()
plt.savefig('data_save/top20_global_feature_importance.png')
plt.close()

# 保存所有全局特征重要性到文件
global_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
global_importance_df.to_csv('data_save/global_feature_importance.csv', index=False)
print(f"全局特征重要性已保存到 global_feature_importance.csv 文件.")

#########################绘制 ROC 曲线############################
# 计算 ROC 曲线
fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1])
roc_auc = auc(fpr, tpr)

# 绘制 ROC 曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('data_save/roc_curve.png')
plt.close()

#########################绘制 PR 曲线############################
# 计算 PR 曲线
precision, recall, _ = precision_recall_curve(y_test, y_proba[:, 1])
average_precision = average_precision_score(y_test, y_proba[:, 1])

# 绘制 PR 曲线
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {average_precision:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve')
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig('data_save/pr_curve.png')
plt.close()

#########################显示混淆矩阵############################
# 计算混淆矩阵
cm = confusion_matrix(y_test, predictions)

# 绘制混淆矩阵
plt.figure(figsize=(6, 5))
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
cm_display.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('data_save/confusion_matrix.png')
plt.close()

#########################绘制 Sensitivity - (100 - Specificity) 图#########################
# 计算 Sensitivity 和 100 - Specificity
sensitivity = tpr * 100  # Sensitivity = TPR
fpr_100 = fpr * 100  # 100 - Specificity = FPR * 100

# 绘制 Sensitivity vs 100 - Specificity 图
plt.figure(figsize=(8, 6))
plt.plot(fpr_100, sensitivity, color='green', lw=2)
plt.xlabel('100 - Specificity (False Positive Rate)')
plt.ylabel('Sensitivity (True Positive Rate)')
plt.title('Sensitivity vs 100 - Specificity')
plt.grid(True)
plt.tight_layout()
plt.savefig('data_save/sensitivity_vs_fpr.png')
plt.close()



