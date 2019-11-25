# R+Python 컴퓨팅: Homework 2

## 목차
1. [와인 클래스에 대한 kNN 알고리즘 적용](#1-와인-클래스에-대한-knn-알고리즘-적용)  
1.1. [문제 개요](#11-문제-개요)  
1.2. [분석 진행](#12-분석-진행)
2. [새 알고리즘을 적용해야 할 상황](#2-새-알고리즘을-적용해야-할-상황)  
2.1. [문제 개요](#21-문제-개요)  
2.2. [분석 진행](#22-분석-진행)
3. [K-means Clustering 알고리즘](#3-k-means-clustering-알고리즘)  
3.1. [문제 개요](#31-문제-개요)  
3.2. [분석 진행](#32-분석-진행)
### a
## 1. 와인 클래스에 대한 kNN 알고리즘 적용
### 1.1. 문제 개요
[wine_data.csv](https://github.com/94wogus/R_python_homework2/blob/master/wine_data.csv)
에는 3 가지 종류의 와인에서 발견 된 다른 성분들에 대해 13 가지의 특성을 기록이 되어 있습니다.  
이것을 활용하여 분석을 진행하려 합니다.

분석 과정에서 출력 내용을 조금 보기 쉽게 하기 위하여 다음과 같은 함수를 지정하였습니다.
```python
import math
def section(str, start=False):
    l = (120 - len(str)) / 2
    if not start:
        print('\n')
    print("= "*math.ceil(l) + str.upper() + " ="*math.floor(l))
``` 
### a
### a
### a
### 1.2. 분석 진행
#### 1.2.1. Make Wine Dataframe
Pandas를 사용해 wine_data.csv파일을 wine 데이터프레임을 만듭니다.
```python
from pandas import read_csv
csv_path = './wine_data.csv'
wine_df = read_csv(csv_path)

print(wine_df)
```
```text
     Class  Alcohol  Malic acid   Ash  Alcalinity of ash  Magnesium  Total phenols  Flavanoids  Nonflavanoid phenols  Proanthocyanins  Color intensity   Hue  OD280/OD315 of diluted wines  Proline
0        1    14.23        1.71  2.43               15.6        127           2.80        3.06                  0.28             2.29             5.64  1.04                          3.92     1065
1        1    13.20        1.78  2.14               11.2        100           2.65        2.76                  0.26             1.28             4.38  1.05                          3.40     1050
2        1    13.16        2.36  2.67               18.6        101           2.80        3.24                  0.30             2.81             5.68  1.03                          3.17     1185
3        1    14.37        1.95  2.50               16.8        113           3.85        3.49                  0.24             2.18             7.80  0.86                          3.45     1480
4        1    13.24        2.59  2.87               21.0        118           2.80        2.69                  0.39             1.82             4.32  1.04                          2.93      735
..     ...      ...         ...   ...                ...        ...            ...         ...                   ...              ...              ...   ...                           ...      ...
173      3    13.71        5.65  2.45               20.5         95           1.68        0.61                  0.52             1.06             7.70  0.64                          1.74      740
174      3    13.40        3.91  2.48               23.0        102           1.80        0.75                  0.43             1.41             7.30  0.70                          1.56      750
175      3    13.27        4.28  2.26               20.0        120           1.59        0.69                  0.43             1.35            10.20  0.59                          1.56      835
176      3    13.17        2.59  2.37               20.0        120           1.65        0.68                  0.53             1.46             9.30  0.60                          1.62      840
177      3    14.13        4.10  2.74               24.5         96           2.05        0.76                  0.56             1.35             9.20  0.61                          1.60      560

[178 rows x 14 columns]
```
#### 1.2.2. Wine Dataframe Describe
Wine Dataframe에서 describe 메소드를 사용하여 요약통계량을 구합니다.
```python
print(wine_df.describe())
```
```text
= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 1.2 WINE DESCRIBE = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
            Class     Alcohol  Malic acid         Ash  Alcalinity of ash   Magnesium  Total phenols  Flavanoids  Nonflavanoid phenols  Proanthocyanins  Color intensity         Hue  OD280/OD315 of diluted wines      Proline
count  178.000000  178.000000  178.000000  178.000000         178.000000  178.000000     178.000000  178.000000            178.000000       178.000000       178.000000  178.000000                    178.000000   178.000000
mean     1.938202   13.000618    2.336348    2.366517          19.494944   99.741573       2.295112    2.029270              0.361854         1.590899         5.058090    0.957449                      2.611685   746.893258
std      0.775035    0.811827    1.117146    0.274344           3.339564   14.282484       0.625851    0.998859              0.124453         0.572359         2.318286    0.228572                      0.709990   314.907474
min      1.000000   11.030000    0.740000    1.360000          10.600000   70.000000       0.980000    0.340000              0.130000         0.410000         1.280000    0.480000                      1.270000   278.000000
25%      1.000000   12.362500    1.602500    2.210000          17.200000   88.000000       1.742500    1.205000              0.270000         1.250000         3.220000    0.782500                      1.937500   500.500000
50%      2.000000   13.050000    1.865000    2.360000          19.500000   98.000000       2.355000    2.135000              0.340000         1.555000         4.690000    0.965000                      2.780000   673.500000
75%      3.000000   13.677500    3.082500    2.557500          21.500000  107.000000       2.800000    2.875000              0.437500         1.950000         6.200000    1.120000                      3.170000   985.000000
max      3.000000   14.830000    5.800000    3.230000          30.000000  162.000000       3.880000    5.080000              0.660000         3.580000        13.000000    1.710000                      4.000000  1680.000000
```
### a
#### 1.2.3. train test set 만들기
```python
# pop을 활용하여 DataFrame에서 Class Column을 지움과 동시에 y 변수에 할당합니다.
from sklearn.model_selection import train_test_split
y = wine_df.pop('Class')

# train test set를 0.3의 비율로 분리합니다.
# 또한 데이터의 비율을 y의 비율과 일치 시키기 위해 stratify를 설정합니다.
X_train, X_test, y_train, y_test = train_test_split(wine_df, y, test_size=0.3, stratify=y)
```
### a
#### 1.2.4. Train KNN model(neighbors=5)
Scikit-learn의 KNeighborsClassifier를 사용하여 70%인 X_train과 y_train을 바탕으로 모형을 트레이닝 시킵니다.  
n_neighbors은 5로 설정 하였습니다.
```python
from sklearn.neighbors import KNeighborsClassifier
n_neighbors = 5
print("n_neighbors: {}".format(n_neighbors))
wine_knn_5 = KNeighborsClassifier(n_neighbors=n_neighbors, p=2, metric='minkowski')
wine_knn_5.fit(X_train, y_train)

# Model에 대하여 Train Set의 예측값을 출력 합니다.
train_score = wine_knn_5.score(X_train, y_train)
print("Score_with_train_set: {}%".format(round(train_score*100, 2)))
print(wine_knn_5.score(X_train, y_train))

# Model에 대하여 Test Set의 예측값을 출력 합니다.
test_score = wine_knn_5.score(X_test, y_test)
print("Score_with_test_set: {}%".format(round(test_score*100, 2)))
print(wine_knn_5.score(X_test, y_test))
```
```text
n_neighbors: 5
Score_with_train_set: 79.84%
0.7983870967741935
Score_with_test_set: 68.52%
0.6851851851851852
```
#### 1.2.5. Train KNN model(neighbors=3)
다시한번 Scikit-learn의 KNeighborsClassifier를 사용하여 70%인 X_train과 y_train을 바탕으로 모형을 트레이닝 시킵니다.  
n_neighbors은 3으로 설정 하였습니다.
```python
n_neighbors = 3
print("n_neighbors: {}".format(n_neighbors))
wine_knn_3 = KNeighborsClassifier(n_neighbors=n_neighbors, p=2, metric='minkowski')
wine_knn_3.fit(X_train, y_train)

# Model에 대하여 Train Set의 예측값을 출력 합니다.
train_score = wine_knn_3.score(X_train, y_train)
print("Score_with_train_set: {}%".format(round(train_score*100, 2)))
print(wine_knn_5.score(X_train, y_train))

# Model에 대하여 Test Set의 예측값을 출력 합니다.
test_score = wine_knn_3.score(X_test, y_test)
print("Score_with_train_set: {}%".format(round(test_score*100, 2)))
print(wine_knn_5.score(X_test, y_test))
```
```text
n_neighbors: 3
Score_with_train_set: 87.1%
0.7983870967741935
Score_with_train_set: 70.37%
0.6851851851851852
```

#### 1.2.6. Change X data
X 데이터를 Alchol, Malic acid, Ash, Alcalinity of ash 네 변수만 활용하여 다시 수행합니다
```python
# 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash' 행만 X로 설정합니다.
X = wine_df.loc[:, ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash']]
print(X)

# train test set를 0.3의 비율로 분리합니다"
# y의 비율을 유지하며 나누기 위해 stratify를 설정합니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

# Scikit-learn의 KNeighborsClassifier를 사용하여 
# 70%인 X_train과 y_train을 바탕으로모형을 트레이닝 시킵니다.
n_neighbors = 5
print("n_neighbors: {}".format(n_neighbors))
wine_knn_5 = KNeighborsClassifier(n_neighbors=n_neighbors, p=2, metric='minkowski')
wine_knn_5.fit(X_train, y_train)

# Model에 대하여 Train Set의 예측값을 출력 합니다.
train_score = wine_knn_5.score(X_train, y_train)
print("Score_with_train_set: {}%".format(round(train_score*100, 2)))
print(wine_knn_5.score(X_train, y_train))

# Model에 대하여 Test Set의 예측값을 출력 합니다.
test_score = wine_knn_5.score(X_test, y_test)
print("Score_with_train_set: {}%".format(round(test_score*100, 2)))
print(wine_knn_5.score(X_test, y_test))


n_neighbors = 3
print("n_neighbors: {}".format(n_neighbors))
wine_knn_3 = KNeighborsClassifier(n_neighbors=n_neighbors, p=2, metric='minkowski')
wine_knn_3.fit(X_train, y_train)

# Model에 대하여 Train Set의 예측값을 출력 합니다.
train_score = wine_knn_3.score(X_train, y_train)
print("Score_with_train_set: {}%".format(round(train_score*100, 2)))
print(wine_knn_3.score(X_train, y_train))

# Model에 대하여 Test Set의 예측값을 출력 합니다.
test_score = wine_knn_3.score(X_test, y_test)
print("Score_with_train_set: {}%".format(round(test_score*100, 2)))
print(wine_knn_3.score(X_test, y_test))
```
```text
[X data]
     Alcohol  Malic acid   Ash  Alcalinity of ash
0      14.23        1.71  2.43               15.6
1      13.20        1.78  2.14               11.2
2      13.16        2.36  2.67               18.6
3      14.37        1.95  2.50               16.8
4      13.24        2.59  2.87               21.0
..       ...         ...   ...                ...
173    13.71        5.65  2.45               20.5
174    13.40        3.91  2.48               23.0
175    13.27        4.28  2.26               20.0
176    13.17        2.59  2.37               20.0
177    14.13        4.10  2.74               24.5

[178 rows x 4 columns]

n_neighbors: 5
Score_with_train_set: 84.68%
0.8467741935483871
Score_with_train_set: 79.63%
0.7962962962962963

n_neighbors: 3
Score_with_train_set: 88.71%
0.8870967741935484
Score_with_train_set: 70.37%
0.7037037037037037
```
### 2. 새 알고리즘을 적용해야 할 상황
### 2.1. 문제 개요
문제 상황: A라는 학생 본인은 인턴쉽 도중 와인데이터를 활용한 결과 Q1에 대해서 B라는 대리에게 보고했다.  
B라는 대리는 support vector machine이라는 기법이란 것이 있으며, 자기가 모형 파라미터 값만 줄 테니   
조금 다른 모형을 예측만 해서 결과값만 알려달라고 하였다. B대리는 다음과 같은 분석 조건에 내걸었다.

1.	모형” from sklearn.svm import SVC
2.	SVC 파라미터는 (kernel='linear', C=1.0, gamma='auto')
3.	주어진 데이터에 첫 변수, class를 우리가 예측하는 타겟 변수, y, 나머지 모든 변수를 예측 변수 X로 활용하고, Train/Test 표본을 70% 와 30%가 되도록 데이터를 구성한다. 각각의 명칭은 X_train/X_test, y_train/y_test로 칭한다. 
4.	SVC로  Training 된 모형을 바탕으로 Training Set에서 예측해보고 원래의 라벨을 얼마나 정확하게 예측하는지 percentage 단위로 출력한다. 
5.	이 Training 된 모형을 바탕으로 Test Set에서 원래의 라벨을 얼마나 정확하게 예측하는지 percentage 단위로 출력한다

### 2.2. 분석 진행
#### 2.2.1. Make Wine Dataframe
Pandas를 사용해 wine_data.csv파일을 wine 데이터프레임을 만듭니다.
```python
from pandas import read_csv
csv_path = './wine_data.csv'
wine_df = read_csv(csv_path)

print(wine_df)
```
```text
     Class  Alcohol  Malic acid   Ash  Alcalinity of ash  Magnesium  Total phenols  Flavanoids  Nonflavanoid phenols  Proanthocyanins  Color intensity   Hue  OD280/OD315 of diluted wines  Proline
0        1    14.23        1.71  2.43               15.6        127           2.80        3.06                  0.28             2.29             5.64  1.04                          3.92     1065
1        1    13.20        1.78  2.14               11.2        100           2.65        2.76                  0.26             1.28             4.38  1.05                          3.40     1050
2        1    13.16        2.36  2.67               18.6        101           2.80        3.24                  0.30             2.81             5.68  1.03                          3.17     1185
3        1    14.37        1.95  2.50               16.8        113           3.85        3.49                  0.24             2.18             7.80  0.86                          3.45     1480
4        1    13.24        2.59  2.87               21.0        118           2.80        2.69                  0.39             1.82             4.32  1.04                          2.93      735
..     ...      ...         ...   ...                ...        ...            ...         ...                   ...              ...              ...   ...                           ...      ...
173      3    13.71        5.65  2.45               20.5         95           1.68        0.61                  0.52             1.06             7.70  0.64                          1.74      740
174      3    13.40        3.91  2.48               23.0        102           1.80        0.75                  0.43             1.41             7.30  0.70                          1.56      750
175      3    13.27        4.28  2.26               20.0        120           1.59        0.69                  0.43             1.35            10.20  0.59                          1.56      835
176      3    13.17        2.59  2.37               20.0        120           1.65        0.68                  0.53             1.46             9.30  0.60                          1.62      840
177      3    14.13        4.10  2.74               24.5         96           2.05        0.76                  0.56             1.35             9.20  0.61                          1.60      560

[178 rows x 14 columns]
```
#### 2.2.2. train test set 만들기
```python
# pop을 활용하여 DataFrame에서 Class Column을 지움과 동시에 y 변수에 할당합니다.
from sklearn.model_selection import train_test_split
y = wine_df.pop('Class')

# train test set를 0.3의 비율로 분리합니다.
# 또한 데이터의 비율을 y의 비율과 일치 시키기 위해 stratify를 설정합니다.
X_train, X_test, y_train, y_test = train_test_split(wine_df, y, test_size=0.3, stratify=y)
```
#### 2.2.3. SVC 모델 학습 및 예측값 출력하기
```python
# Scikit-learn의 SVC 사용하여 70%인 X_train과 y_train을 바탕으로 모형을 트레이닝 시킵니다.
from sklearn.svm import SVC
SVC_model = SVC(kernel='linear', C=1.0, gamma='auto')
SVC_model.fit(X_train, y_train)

# Model에 대하여 Train Set의 예측값을 출력 합니다.
train_score = SVC_model.score(X_train, y_train)
print("Score_with_train_set: {}%".format(round(train_score*100, 2)))
print(train_score)

# Model에 대하여 Test Set의 예측값을 출력 합니다.
test_score = SVC_model.score(X_test, y_test)
print("Score_with_test_set: {}%".format(round(test_score*100, 2)))
print(test_score)
```
```text
Score_with_train_set: 100.0%
1.0
Score_with_test_set: 94.44%
0.9444444444444444
```

## 3. K-means Clustering 알고리즘
### 3.1. 문제 개요
K-means Clustering을 활용하여 분석을 진행한다.

1.	Wine 주어진 데이터에 첫 변수, class를 제외한 나머지 데이터를 X로 결정한다. 
2.	Cluster를 세개로 구성하여 K-means Clustering 알고리즘을 적용한다. 
3.	X 데이터를 바탕으로 라벨을 예측해보고, pd. Crosstab 명령어를 사용하여, 실제 라벨값과 예측된 라벨값을 비교한다. 
4.	k=1 부터 9까지 변형시키면서 각 모형의 inertia를 구해본다. 

### 3.2. 분석 진행