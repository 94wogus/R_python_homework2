import math
import pandas as pd

# 출력 결과물 도와주는 함수 정의하였습니다.
def section(str, start=False):
    l = (120 - len(str)) / 2
    if not start:
        print('\n')
    print("= "*math.ceil(l) + str.upper() + " ="*math.floor(l))


section("Q1 와인 클래스에 대한 kNN 알고리즘 적용", start=True)
section("# 1.1 make wine dataframe")
# 1.1 Pandas를 사용해 wine_data.csv파일을 wine 데이터프레임을 만듭니다.
from pandas import read_csv
csv_path = './wine_data.csv'
wine_df = read_csv(csv_path)

print(wine_df)

section("1.2 wine describe")
# 1.2 Wine 데이터에 describe 메소드를 사용하여 요약통계량을 구합니다.
print(wine_df.describe())

section("1.3 train test set")
# pop을 활용하여 DataFrame에서 Class Column을 지움과 동시에 y 변수에 할당합니다.
from sklearn.model_selection import train_test_split
y = wine_df.pop('Class')

# train test set를 0.3의 비율로 분리합니다.
# 또한 데이터의 비율을 y의 비율과 일치 시키기 위해 stratify를 설정합니다.
X_train, X_test, y_train, y_test = train_test_split(wine_df, y, test_size=0.3, stratify=y)
print(X_train.head(), "X_train", '\n')
print(X_test.head(), "X_test", '\n')
print(y_train.head(), "y_train", '\n')
print(y_test.head(), "y_test", '\n')
#
# section("1.4 ~ 1.6 train KNeighborsClassifier model / n_neighbors=5")
# # Scikit-learn의 KNeighborsClassifier를 사용하여 70%인 X_train과 y_train을 바탕으로 모형을 트레이닝 시킵니다.
# from sklearn.neighbors import KNeighborsClassifier
# n_neighbors = 5
# print("n_neighbors: {}".format(n_neighbors))
# wine_knn_5 = KNeighborsClassifier(n_neighbors=n_neighbors, p=2, metric='minkowski')
# wine_knn_5.fit(X_train, y_train)
#
# # Model에 대하여 Train Set의 예측값을 출력 합니다.
# train_score = wine_knn_5.score(X_train, y_train)
# print("Score_with_train_set: {}%".format(round(train_score*100, 2)))
# print(wine_knn_5.score(X_train, y_train))
#
# # Model에 대하여 Test Set의 예측값을 출력 합니다.
# test_score = wine_knn_5.score(X_test, y_test)
# print("Score_with_test_set: {}%".format(round(test_score*100, 2)))
# print(wine_knn_5.score(X_test, y_test))
#
# section("1.7 train KNeighborsClassifier model / n_neighbors=3")
# from sklearn.neighbors import KNeighborsClassifier
# n_neighbors = 3
# print("n_neighbors: {}".format(n_neighbors))
# wine_knn_3 = KNeighborsClassifier(n_neighbors=n_neighbors, p=2, metric='minkowski')
# wine_knn_3.fit(X_train, y_train)
#
# # Model에 대하여 Train Set의 예측값을 출력 합니다.
# train_score = wine_knn_3.score(X_train, y_train)
# print("Score_with_train_set: {}%".format(round(train_score*100, 2)))
# print(wine_knn_5.score(X_train, y_train))
#
# # Model에 대하여 Test Set의 예측값을 출력 합니다.
# test_score = wine_knn_3.score(X_test, y_test)
# print("Score_with_train_set: {}%".format(round(test_score*100, 2)))
# print(wine_knn_5.score(X_test, y_test))
#
#
# section("1.8 Change X data")
# # 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash' 행만 X로 설정한다.
# X = wine_df.loc[:, ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash']]
# print('X')
#
# # train test set를 0.3의 비율로 분리한다"
# # y의 비율을 유지하며 나누기 위해 stratify를 설정한다.
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
# print(X_train.head())
# print(X_test.head())
# print(y_train.head())
# print(y_test.head())
#
# # Scikit-learn의 KNeighborsClassifier를 사용하여 70%인 X_train과 y_train을 바탕으로 모형을 트레이닝 시킨다.
# n_neighbors = 5
# print('n_neighbors')
# wine_knn_5 = KNeighborsClassifier(n_neighbors=n_neighbors, p=2, metric='minkowski')
# wine_knn_5.fit(X_train, y_train)
#
# # Model에 대하여 Train Set의 예측값을 출력 합니다.
# train_score = wine_knn_5.score(X_train, y_train)
# print("Score_with_train_set: {}%".format(round(train_score*100, 2)))
# print(wine_knn_5.score(X_train, y_train))
#
# # Model에 대하여 Test Set의 예측값을 출력 합니다.
# test_score = wine_knn_5.score(X_test, y_test)
# print("Score_with_train_set: {}%".format(round(test_score*100, 2)))
# print(wine_knn_5.score(X_test, y_test))
#
#
# n_neighbors = 3
# print("n_neighbors: {}".format(n_neighbors))
# wine_knn_3 = KNeighborsClassifier(n_neighbors=n_neighbors, p=2, metric='minkowski')
# wine_knn_3.fit(X_train, y_train)
#
# # Model에 대하여 Train Set의 예측값을 출력 합니다.
# train_score = wine_knn_3.score(X_train, y_train)
# print("Score_with_train_set: {}%".format(round(train_score*100, 2)))
# print(wine_knn_3.score(X_train, y_train))
#
# # Model에 대하여 Test Set의 예측값을 출력 합니다.
# test_score = wine_knn_3.score(X_test, y_test)
# print("Score_with_train_set: {}%".format(round(test_score*100, 2)))
# print(wine_knn_3.score(X_test, y_test))