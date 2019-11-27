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
wine_df.pop('Class')
X = wine_df
print(X)

section("# 3.2 make K-means Clustering Model")
# K-means Clustering Model을 구성한다.
from sklearn.cluster import KMeans
import matplotlib.pyplot  as plt
import pandas
model = KMeans(n_clusters=3,algorithm='auto')
model.fit(X)

predict = pandas.DataFrame(model.predict(X))
predict.columns = ['predict']
print(predict)