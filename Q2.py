import math

def section(str, start=False):
    l = (120 - len(str)) / 2
    if not start:
        print('\n')
    print("= "*math.ceil(l) + str.upper() + " ="*math.floor(l))

section("Q2 새 알고리즘을 적용해야 할 상황", start=True)
section("# 2.1 make wine dataframe")
# 2.1 Pandas를 사용해 wine_data.csv파일을 wine 데이터프레임을 만듭니다.
from pandas import read_csv
csv_path = './wine_data.csv'
wine_df = read_csv(csv_path)
print(wine_df)

section("2.2 make train test set")
# pop을 활용하여 DataFrame에서 Class Column을 지움과 동시에 y 변수에 할당합니다.
from sklearn.model_selection import train_test_split
y = wine_df.pop('Class')

# train test set를 0.3의 비율로 분리합니다.
# 또한 데이터의 비율을 y의 비율과 일치 시키기 위해 stratify를 설정합니다.
X_train, X_test, y_train, y_test = train_test_split(wine_df, y, test_size=0.3, stratify=y)
print("[X_train]")
print(X_train, '\n')
print("[X_test]")
print(X_test, '\n')
print("[y_train]")
print(y_train, '\n')
print("[y_test]")
print(y_test, '\n')

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