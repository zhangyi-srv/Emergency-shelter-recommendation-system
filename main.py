import pandas as pd
import numpy as np
from matplotlib.ticker import MultipleLocator
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from sklearn import metrics


sns.set()
df = pd.read_csv("E:/重要项目/safe/避难位置数据1.csv")

# DBSCAN数据聚类
df = df[['long', 'lati']].dropna(axis=0, how='all')
data = np.array(df)

db = DBSCAN(eps=0.5, min_samples=2).fit(data)
labels = db.labels_
radio = len(labels[labels[:] == -1]) / len(labels)  # 求噪声点个数占总数的比例
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # 获取分簇的数目
score = metrics.silhouette_score(data, labels)  # 计算轮廓系数

df['label'] = labels
print(radio)
print(n_clusters_)
print(score)


p = sns.lmplot(x='long', y='lati', data=df, hue='label', fit_reg=False)

plt.savefig('E:/重要项目/safe/0.5,2.png')
plt.show()


map_ = folium.Map(location=[22.552763, 114.097022], zoom_start=12,
                  tiles='https://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=7&x={x}&y={y}&z={z}',
                  control_scale=True,
                  attr='default')

colors = ['#DC143C', '#FFB6C1', '#DB7093', '#C71585', '#8B008B', '#4B0082', '#7B68EE',
          '#0000FF', '#B0C4DE', '#708090', '#00BFFF', '#5F9EA0', '#00FFFF', '#7FFFAA',
          '#008000', '#FFFF00', '#808000', '#FFD700', '#FFA500', '#FF6347', '#DC143C',
          '#FFB6C1', '#DB7093', '#C71585', '#8B008B', '#4B0082', '#7B68EE',
          '#0000FF', '#B0C4DE', '#708090', '#00BFFF', '#5F9EA0', '#00FFFF', '#7FFFAA',
          '#008000', '#FFFF00', '#808000', '#FFD700', '#FFA500', '#FF6347',
          '#DC143C', '#FFB6C1', '#DB7093', '#C71585', '#8B008B', '#4B0082', '#7B68EE',
          '#0000FF', '#B0C4DE', '#708090', '#00BFFF', '#5F9EA0', '#00FFFF', '#7FFFAA',
          '#008000', '#FFFF00', '#808000', '#FFD700', '#FFA500', '#FF6347',
          '#DC143C', '#FFB6C1', '#DB7093', '#C71585', '#8B008B', '#4B0082', '#7B68EE',
          '#0000FF', '#B0C4DE', '#708090', '#00BFFF', '#5F9EA0', '#00FFFF', '#7FFFAA',
          '#008000', '#FFFF00', '#808000', '#FFD700', '#FFA500', '#FF6347', '#DC143C',
          '#FFB6C1', '#DB7093', '#C71585', '#8B008B', '#4B0082', '#7B68EE',
          '#0000FF', '#B0C4DE', '#708090', '#00BFFF', '#5F9EA0', '#00FFFF', '#7FFFAA',
          '#008000', '#FFFF00', '#808000', '#FFD700', '#FFA500', '#FF6347',
          '#DC143C', '#FFB6C1', '#DB7093', '#C71585', '#8B008B', '#4B0082', '#7B68EE',
          '#0000FF', '#B0C4DE', '#708090', '#00BFFF', '#5F9EA0', '#00FFFF', '#7FFFAA',
          '#008000', '#FFFF00', '#808000', '#FFD700', '#FFA500', '#FF6347', '#000000']

for i in range(len(data)):
    folium.CircleMarker(location=(data[i][1], data[i][0]),
                        radius=5, popup='popup',
                        color=colors[labels[i]], fill=True,
                        fill_color=colors[labels[i]]).add_to(map_)

map_.save('cluster.html')
