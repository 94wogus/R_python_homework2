import matplotlib.pyplot as plt
import mglearn
import math

def section(str, start=False):
    l = (120 - len(str)) / 2
    if not start:
        print('\n')
    print("= "*math.ceil(l) + str.upper() + " ="*math.floor(l))


section("Q3 K-means Clustering 알고리즘", start=True)
section("# 3.1 make X data")
# Pandas를 사용해 wine_data.csv파일을 읽어 들입니다.
from pandas import read_csv
csv_path = './wine_data.csv'
wine_df = read_csv(csv_path)

# Class column 제거
y = wine_df.pop('Class')
X = wine_df

section("# 3.2 Train K-means Clustering Model")
# K-means Clustering Model을 구성합니다.
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from pandas import crosstab
import numpy

# KMeans 모델을 학습 시킵니다.
X_normalized = MinMaxScaler().fit(X).transform(X)
model = KMeans(n_clusters=3, algorithm='auto')
model.fit(X_normalized)

# 분류 모델과 실제 라벨을 비교 해봅니다.
pt = crosstab(y, model.labels_)
print(pt)

# Figure를 출력합니다.
plt.figure(1)
mglearn.discrete_scatter(X_normalized[:, 0], X_normalized[:, 1], model.labels_, markers='o')
mglearn.discrete_scatter(
    model.cluster_centers_[:, 0],
    model.cluster_centers_[:, 1],
    y=numpy.unique(model.labels_),
    markers='^',
    markeredgewidth=2  # marker, 두께
)
plt.savefig('./result/3_n_clusters.png')

section("# 3.3 find cluster number")
ks = range(1, 10)
inertias = []

# 클러스터의 갯수를 변화시키면서 inertia를 확인합니다.
for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(X_normalized)
    inertias.append(model.inertia_)
print(inertias)

# 그래프로 출력합니다.
plt.figure(2)
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.savefig('./result/inertia.png')




