import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib
# 解决中文乱码问题
matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 优先用黑体/微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
# 1. 加载新闻数据集（只取两类：科技和娱乐）
categories = ['sci.space', 'rec.sport.baseball']
newsgroups = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))
# 2. 文本特征提取（TF-IDF）
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(newsgroups.data)
y = newsgroups.target
# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 4. 训练朴素贝叶斯分类器
model = MultinomialNB()
model.fit(X_train, y_train)
# 5. 预测并评估
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
# 6. 可视化混淆矩阵
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=newsgroups.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title(f'朴素贝叶斯文本分类\n准确率：{acc:.2f}')
plt.show()
print(f"测试集准确率：{acc:.2f}")