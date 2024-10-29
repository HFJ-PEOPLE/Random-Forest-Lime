import pandas as pd
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# 加载 combined_data.csv 文件
file_path = 'data_save/Ag.csv'
df = pd.read_csv(file_path, header=None)

# 假设第一列是标签，剩余的列是 (x, y) 对形式的特征
y = df.iloc[:, 0]  # 标签（第一列）
X = df.iloc[:, 1:]  # 特征（剩余列）

# 将每行的 (x, y) 点依次连接为一条线
X_lines = []
for idx in range(len(X)):
    points = X.iloc[idx].values.reshape(-1, 2)  # 将一行数据分割为多个 (x, y) 点
    lines = []
    for i in range(len(points) - 1):
        line_segment = np.concatenate([points[i], points[i + 1]])
        lines.append(line_segment)
    X_lines.append(np.concatenate(lines))

X_lines = np.array(X_lines)

# 确认特征名列表长度与特征数据一致，特征名称中的 "start" 和 "end" 分别代表每个线段的起始点和结束点。
#例如，如果第 97 条线段由 (x1, y1) 和 (x2, y2) 两个点连接而成，那么：
#"line97 x start" 和 "line97 y start" 分别表示点 (x1, y1) 的坐标；
#"line97 x end" 和 "line97 y end" 分别表示点 (x2, y2) 的坐标。
num_segments = X_lines.shape[1] // 4
feature_names = [f'line{(i // 4) + 1} x {["start", "end"][(i % 4) // 2]}, line{(i // 4) + 1} y {["start", "end"][i % 2]}' for i in range(X_lines.shape[1])]

assert len(feature_names) == X_lines.shape[1], "特征名列表长度与特征数据不匹配，请检查特征名列表的生成过程。"

# 随机划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_lines, y, test_size=0.3, random_state=42)
# 检查是否有NaN值
# print(df.isnull().sum())

# 训练 RandomForestClassifier 模型
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 保存所有测试样本的预测结果
predictions = model.predict(X_test)
# 计算测试准确度
accuracy = accuracy_score(y_test, predictions)
print(f"模型在测试集上的准确度为: {accuracy:.4f}")
# 将测试结果和准确度保存到文件
results_df = pd.DataFrame({'Prediction': predictions, 'True Label': y_test})
results_df['Accuracy'] = accuracy  # 添加准确度列
results_df.to_csv('data_save/model_predictions.csv', index=False)
print(f"模型预测结果和准确度已保存到 model_predictions.csv 文件.")

##########################获得整体数据集上每个线段特征的相对重要性#####################
# 获取特征重要性,最终的特征重要性是基于所有树中特征重要性的平均或累积值
#特征重要性分析是基于训练集的特征命名和特征重要性值
feature_importance = model.feature_importances_

# # 打印每个特征的重要性
# print("全局特征重要性:")
# for i, name in enumerate(feature_names):
#     print(f"{name}: {feature_importance[i]}")

# 只选择重要性排名前5的特征进行可视化
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
plt.show()

# 保存所有全局特征重要性到文件
global_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
global_importance_df.to_csv('data_save/global_feature_importance.csv', index=False)
print(f"全局特征重要性已保存到 global_feature_importance.csv 文件.")

#########################针对每个测试样本的局部解释###############################################
# # 使用 LIME 进行解释性分析
# explainer = LimeTabularExplainer(
#     training_data=X_train,  # 训练集特征
#     feature_names=feature_names,  # 特征列名
#     class_names=np.unique(y_train),  # 标签类别名称
#     mode='classification'  # 分类模式
# )
# #最终得到的解释性结果时，它们是基于每条线段的四个特征（起始点 x、y 和 结束点 x、y）来进行解释的。
# #每个解释性结果都会列出对应特征的权重，这些权重反映了每个特征对模型预测的贡献
#
# # 计算测试集中的样本数量
# num_samples = min(20, len(X_test))  # 最多选择5个样本，但不超过测试集中的样本数量
#
# # 创建一个空的列表来保存解释性结果
# explanations = []
# # 创建一个空的列表来保存条形图
# bar_plots = []
# # 随机选择样本索引
# random_indices = np.random.choice(len(X_test), size=num_samples, replace=False)
# for idx in random_indices:
#     data_test = X_test[idx].reshape(1, -1)
#     prediction = model.predict(data_test)[0]
#     y_true = y_test.iloc[idx]
#     print(f'测试集中的 {idx} 号样本，模型预测为 {prediction}，真实类别为 {y_true}')
#
#     # 解释模型预测
#     # LIME 解释性分析：对于每个选定的测试样本，调用 explain_instance 方法来生成特征权重列表，展示了每个线段特征对模型预测的贡献程度。
#     #这些权重反映了每个特征对于局部预测的贡献程度。
#     #正的权重表示该特征增加了预测值，负的权重表示减少了预测值。权重的大小反映了特征对于模型预测的相对重要性，较大的权重表示特征对预测有更大的影响。
#     #权重值的大小取决于模型在局部区域内对特征的敏感度和该特征值在样本中的变化。
#     #条件中的区间范围（如 <= 0.04 或 <= 1.87）反映了在该范围内特征的变化对于模型预测的影响
#     #解释性结果中条件的大小和权重的大小不是简单的线性关系，而是由LIME算法通过对模型的局部理解生成的相对结果。
#     #因此，不同条件的权重值大小并不意味着条件大小的绝对重要性，而是相对于局部预测而言的重要性。
#     exp = explainer.explain_instance(
#         data_row=data_test[0],
#         predict_fn=model.predict_proba,
#         num_features=len(feature_names)  # 每个线段有多个特征
#     )
#
#     # 获取特征权重并按权重值排序
#     explanation = exp.as_list()
#     # 过滤出正权重的特征
#     positive_features = [(feat, weight) for feat, weight in explanation if weight > 0]
#
#     # 按权重值降序排序
#     positive_features.sort(key=lambda x: x[1], reverse=True)
#
#     # 只打印正权重的前5个特征
#     top_positive_features = positive_features[:5]
#     print("解释性结果:")
#     for feature, weight in top_positive_features:
#         print(f"{feature} : {weight}")
#     print("\n")
#
#     # 将所有特征的解释结果保存到列表中
#     explanations.append(explanation)
#
#     # 可视化解释性结果（仅展示前5个特征）
#     fig, ax = plt.subplots()
#     features, weights = zip(*top_positive_features)
#     y_pos = np.arange(len(features))
#     ax.barh(y_pos, weights, align='center')
#     ax.set_yticks(y_pos)
#     ax.set_yticklabels(features)
#     ax.invert_yaxis()  # 反转 y 轴，使得重要性高的特征显示在顶部
#     ax.set_xlabel('weight')
#     # 修改标题为包含样本索引、预测结果和真实结果的格式
#     ax.set_title(f'Sample {idx},prediction: {prediction},True: {y_true},Top 20 ')
#     # 保存图形为 JPG 文件
#     plt.savefig(f'data_save/lime_explanation_sample_{idx}.jpg', bbox_inches='tight')
#     plt.show()
#
# # 将解释性结果保存到文件
# explanations_df = pd.DataFrame(explanations)
# explanations_df.to_csv('data_save/lime_explanations.csv', index=False, header=False)  # 添加 header=False 避免保存时添加额外行
# print(f"解释性结果已保存到 lime_explanations.csv 文件.")

