#20171622 박건후 과제 #7

## 소감
- [x] 이전까지 회귀 문제에 있어서 목표 데이터가 연속된 수치였지만, 분류 문제에서 목표 데이터는 클래스라는 점을 염두에 두고 확률의 개념을 도입해보았습니다. 이전까지는 예측 값을 출력했지만 확률을 출력해봄으로써 로지스틱 회귀라는 것을 배워볼 수 있었습니다.
- [x] 로지스틱 회귀 모델의 식과 평균 교차 엔트로피 오차함수, 편미분 등을 사용하고, 최종적으로 경사하강법을 통해 결정 경계를 구해볼 수 있었습니다.
- [x] 입력 데이터가 2차원인 경우로 확장하여 로지스틱 회귀 모델의 출력을 등고선 형태로 출력을 해보았고, 평균 교차 엔트로피 오차가 최소가 되도록 로지스틱 회귀 모델의 매개 변수를 구하고, 결과 표시까지 해볼 수 있었습니다.
- [x] 주어진 입력 데이터 x에 대해 라벨 데이터 t가 생성될 확률이 가장 커지는 w를 추정치로 하는 최대가능도법과 가능도에 로그를 취하는 이유를 미분 과정에서 알게 되었습니다.



##### 곤충 N마리의 데이터를 하나의 예로 듭니다. 각각의 무게를 xn으로, 각각의 성별을 tn으로 나타냅니다. tn은 0 또는 1을 갖는 변수로 0이면 암컷, 1이면 수컷을 나타내고 있다고 가정합니다. 이 데이터를 기초로 무게를 통해 성별을 예측하는 모델을 만드는 것이 목적입니다. [리스트 6-1-(1)]로 인공 데이터를 만듭니다.

```python
# 리스트 6-1-(1)
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
# 데이터 생성 --------------------------------
np.random.seed(seed=0) # 난수를 고정
X_min = 0
X_max = 2.5
X_n = 30
X_col = ['cornflowerblue', 'gray']
X = np.zeros(X_n) # 입력 데이터
T = np.zeros(X_n, dtype=np.uint8) # 목표 데이터
Dist_s = [0.4, 0.8] # 분포의 시작 지점
Dist_w = [0.8, 1.6] # 분포의 폭
Pi = 0.5 # 클래스 0의 비율
for n in range(X_n):
    wk = np.random.rand()
    T[n] = 0 * (wk < Pi) + 1 * (wk >= Pi) # (A)
    X[n] = np.random.rand() * Dist_w[T[n]] + Dist_s[T[n]] # (B)
# 데이터 표시 --------------------------------
print('X=' + str(np.round(X, 2)))
print('T=' + str(T))
```

    X=[1.94 1.67 0.92 1.11 1.41 1.65 2.28 0.47 1.07 2.19 2.08 1.02 0.91 1.16
     1.46 1.02 0.85 0.89 1.79 1.89 0.75 0.9  1.87 0.5  0.69 1.5  0.96 0.53
     1.21 0.6 ]
    T=[1 1 0 0 1 1 1 0 0 1 1 0 0 0 1 0 0 0 1 1 0 1 1 0 0 1 1 0 1 0]



##### 다음 [리스트 6-1-(2)]로 작성한 데이터를 표시합니다.

```python
# 리스트 6-1-(2)
# 데이터 분포 표시 ----------------------------
def show_data1(x, t):
    K = np.max(t) + 1
    for k in range(K): # (A)
        plt.plot(x[t == k], t[t == k], X_col[k], alpha=0.5,
                 linestyle='none', marker='o') # (B)
        plt.grid(True)
        plt.ylim(-.5, 1.5)
        plt.xlim(X_min, X_max)
        plt.yticks([0, 1])


# 메인 ------------------------------------
fig = plt.figure(figsize=(3, 3))
show_data1(X, T)
plt.show()
```


![png](output_1_0.png)



 ##### 로지스틱 회귀 모델을 정의합니다. 로지스틱 회계 모델을 결정 경계와 함께 표시하는 함수를 [리스트 6-1-(4)]에서 만듭니다. 실행하면 로지스틱 회귀 모델과 결정 경계가 표시되고, 결정 경계의 값도 출력됩니다.

```python
# 리스트 6-1-(3)
def logistic(x, w):
    y = 1 / (1 + np.exp(-(w[0] * x + w[1])))
    return y
```


```python
# 리스트 6-1-(4)
def show_logistic(w):
    xb = np.linspace(X_min, X_max, 100)
    y = logistic(xb, w)
    plt.plot(xb, y, color='gray', linewidth=4)
    # 결정 경계
    i = np.min(np.where(y > 0.5)) # (A)
    B = (xb[i - 1] + xb[i]) / 2 # (B)
    plt.plot([B, B], [-.5, 1.5], color='k', linestyle='--')
    plt.grid(True)
    return B


# test
W = [8, -10]
show_logistic(W)
```


    1.25




![png](output_3_1.png)



 ##### 지금까지의 평균 제곱 오차와 마찬가지로 오차가 ‘최소’가 되는 매개 변수를 구하면 됩니다. 그리고 교차 엔트로피를 N으로 나눈 ‘평균 교차 엔트로피 오차’를 로 정의합니다. [리스트 6-1-(5)]에서 평균 교차 엔트로피 오차를 계산하는 함수 cee_logistic(w, x, t) 를 만듭니다.

```python
# 리스트 6-1-(5)
# 평균 교차 엔트로피 오차 ---------------------
def cee_logistic(w, x, t):
    y = logistic(x, w)
    cee = 0
    for n in range(len(y)):
        cee = cee - (t[n] * np.log(y[n]) + (1 - t[n]) * np.log(1 - y[n]))
    cee = cee / X_n
    return cee


# test
W=[1,1]
cee_logistic(W, X, T)
```


    1.0288191541851066



##### 평균 교차 엔트로피 오차가 어떤 모양인지, 형태를 [리스트 6-1-(6)]에서 확인합니다.


```python
# 리스트 6-1-(6)
from mpl_toolkits.mplot3d import Axes3D


# 계산 --------------------------------------
xn = 80 # 등고선 표시 해상도
w_range = np.array([[0, 15], [-15, 0]])
x0 = np.linspace(w_range[0, 0], w_range[0, 1], xn)
x1 = np.linspace(w_range[1, 0], w_range[1, 1], xn)
xx0, xx1 = np.meshgrid(x0, x1)
C = np.zeros((len(x1), len(x0)))
w = np.zeros(2)
for i0 in range(xn):
    for i1 in range(xn):
        w[0] = x0[i0]
        w[1] = x1[i1]
        C[i1, i0] = cee_logistic(w, X, T)


# 표시 --------------------------------------
plt.figure(figsize=(12, 5))
#plt.figure(figsize=(9.5, 4))
plt.subplots_adjust(wspace=0.5)
ax = plt.subplot(1, 2, 1, projection='3d')
ax.plot_surface(xx0, xx1, C, color='blue', edgecolor='black',
                rstride=10, cstride=10, alpha=0.3)
ax.set_xlabel('$w_0$', fontsize=14)
ax.set_ylabel('$w_1$', fontsize=14)
ax.set_xlim(0, 15)
ax.set_ylim(-15, 0)
ax.set_zlim(0, 8)
ax.view_init(30, -95)


plt.subplot(1, 2, 2)
cont = plt.contour(xx0, xx1, C, 20, colors='black',
                   levels=[0.26, 0.4, 0.8, 1.6, 3.2, 6.4])
cont.clabel(fmt='%1.1f', fontsize=8)
plt.xlabel('$w_0$', fontsize=14)
plt.ylabel('$w_1$', fontsize=14)
plt.grid(True)
plt.show()
```


![png](output_5_0.png)



```python
# 리스트 6-1-(7)
# 평균 교차 엔트로피 오차의 미분 --------------
def dcee_logistic(w, x, t):
    y = logistic(x, w)
    dcee = np.zeros(2)
    for n in range(len(y)):
        dcee[0] = dcee[0] + (y[n] - t[n]) * x[n]
        dcee[1] = dcee[1] + (y[n] - t[n])
    dcee = dcee / X_n
    return dcee


# --- test
W=[1, 1]
dcee_logistic(W, X, T)
```




    array([0.30857905, 0.39485474])



##### 이제 경사 하강법으로 로지스틱 회귀 모델의 매개 변수를 찾습니다. 실행하는 명령은 [리스트 6-1-(8)]입니다. 여기에서는 scipy.optimize 라이브러리에 포함된 minimize() 함수로 경사 하강법을 시도합니다. (A)


```python
# 리스트 6-1-(8)
from scipy.optimize import minimize


# 매개 변수 검색
def fit_logistic(w_init, x, t):
    res1 = minimize(cee_logistic, w_init, args=(x, t),
                    jac=dcee_logistic, method="CG") # (A)
    return res1.x


# 메인 ------------------------------------
plt.figure(1, figsize=(3, 3))
W_init=[1,-1]
W = fit_logistic(W_init, X, T)
print("w0 = {0:.2f}, w1 = {1:.2f}".format(W[0], W[1]))
B=show_logistic(W)
show_data1(X, T)
plt.ylim(-.5, 1.5)
plt.xlim(X_min, X_max)
cee = cee_logistic(W, X, T)
print("CEE = {0:.2f}".format(cee))
print("Boundary = {0:.2f} g".format(B))
plt.show()
```

    w0 = 8.18, w1 = -9.38
    CEE = 0.25
    Boundary = 1.15 g



![png](output_7_1.png)



```python
%reset
```

    Once deleted, variables cannot be recovered. Proceed (y/[n])? y



##### [리스트 6-2-(1)]에서 2클래스의 분류와 3클래스의 분류 데이터를 함께 만듭니다.

```python
# 리스트 6-2-(1)
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline


# 데이터 생성 --------------------------------
np.random.seed(seed=1)  # 난수를 고정
N = 100 # 데이터의 수
K = 3 # 분포 수
T3 = np.zeros((N, 3), dtype=np.uint8)
T2 = np.zeros((N, 2), dtype=np.uint8)
X = np.zeros((N, 2))
X_range0 = [-3, 3] # X0 범위 표시 용
X_range1 = [-3, 3] # X1의 범위 표시 용
Mu = np.array([[-.5, -.5], [.5, 1.0], [1, -.5]]) # 분포의 중심
Sig = np.array([[.7, .7], [.8, .3], [.3, .8]]) # 분포의 분산
Pi = np.array([0.4, 0.8, 1]) # (A) 각 분포에 대한 비율 0.4 0.8 1
for n in range(N):
    wk = np.random.rand()
    for k in range(K): # (B)
        if wk < Pi[k]:
            T3[n, k] = 1
            break
    for k in range(2):
        X[n, k] = (np.random.randn() * Sig[T3[n, :] == 1, k]
                   + Mu[T3[n, :] == 1, k])
T2[:, 0] = T3[:, 0]
T2[:, 1] = T3[:, 1] | T3[:, 2]
```


```python
# 리스트 6-2-(2)
print(X[:5,:])
```

    [[-0.14173827  0.86533666]
     [-0.86972023 -1.25107804]
     [-2.15442802  0.29474174]
     [ 0.75523128  0.92518889]
     [-1.10193462  0.74082534]]



```python
# 리스트 6-2-(3)
print(T2[:5,:])
```

    [[0 1]
     [1 0]
     [1 0]
     [0 1]
     [1 0]]



 ##### 목적 변수 벡터 tn의 K번째 요소만 1로, 그 외에는 0으로 표기하는 방법을 1-of-K 부호화라고 합니다.

```python
# 리스트 6-2-(4)
print(T3[:5,:])
```

    [[0 1 0]
     [1 0 0]
     [1 0 0]
     [0 1 0]
     [1 0 0]]



#####  [리스트 6-2-(5)]로 T2와 T3를 그림으로 그려봅니다.

```python
# 리스트 6-2-(5)
# 데이터 표시 --------------------------
def show_data2(x, t):
    wk, K = t.shape
    c = [[.5, .5, .5], [1, 1, 1], [0, 0, 0]]
    for k in range(K):
        plt.plot(x[t[:, k] == 1, 0], x[t[:, k] == 1, 1],
                 linestyle='none', markeredgecolor='black',
                 marker='o', color=c[k], alpha=0.8)
        plt.grid(True)


# 메인 ------------------------------
plt.figure(figsize=(7.5, 3))
plt.subplots_adjust(wspace=0.5)
plt.subplot(1, 2, 1)
show_data2(X, T2)
plt.xlim(X_range0)
plt.ylim(X_range1)


plt.subplot(1, 2, 2)
show_data2(X, T3)
plt.xlim(X_range0)
plt.ylim(X_range1)
plt.show()
```


![png](output_13_0.png)

##### [리스트 6-2-(7)]은 모델과 데이터를 3D로 표시하기 위한 것입니다. 실행하면 W=[-1, -1, -1]을 선택한 경우의 2차원 로지스틱 회귀 모델과 데이터를 3차원으로 표시합니다.

```python
# 리스트 6-2-(6)
# 로지스틱 회귀 모델 -----------------
def logistic2(x0, x1, w):
    y = 1 / (1 + np.exp(-(w[0] * x0 + w[1] * x1 + w[2])))
    return y
```


```python
# 리스트 6-2-(7)
# 모델 3D보기 ------------------------------
from mpl_toolkits.mplot3d import axes3d


def show3d_logistic2(ax, w):
    xn = 50
    x0 = np.linspace(X_range0[0], X_range0[1], xn)
    x1 = np.linspace(X_range1[0], X_range1[1], xn)
    xx0, xx1 = np.meshgrid(x0, x1)
    y = logistic2(xx0, xx1, w)
    ax.plot_surface(xx0, xx1, y, color='blue', edgecolor='gray',
                    rstride=5, cstride=5, alpha=0.3)


def show_data2_3d(ax, x, t):
    c = [[.5, .5, .5], [1, 1, 1]]
    for i in range(2):
        ax.plot(x[t[:, i] == 1, 0], x[t[:, i] == 1, 1], 1 - i,
                marker='o', color=c[i], markeredgecolor='black',
                linestyle='none', markersize=5, alpha=0.8)
    Ax.view_init(elev=25, azim=-30)


# test ---
Ax = plt.subplot(1, 1, 1, projection='3d')
W=[-1, -1, -1]
show3d_logistic2(Ax, W)
show_data2_3d(Ax,X,T2)
```


![png](output_15_0.png)



##### 모델의 등고선 표시도 [리스트 6-2-(8)]로 만듭니다. 실행하면 W=[-1, -1, -1]을 선택한 경 우의 로지스틱 회귀 모델의 출력이 등고선으로 표시됩니다.

```python
# 리스트 6-2-(8)
# 모델 등고선 2D 표시 ------------------------


def show_contour_logistic2(w):
    xn = 30 # 파라미터의 분할 수
    x0 = np.linspace(X_range0[0], X_range0[1], xn)
    x1 = np.linspace(X_range1[0], X_range1[1], xn)
    xx0, xx1 = np.meshgrid(x0, x1)
    y = logistic2(xx0, xx1, w)
    cont = plt.contour(xx0, xx1, y, levels=(0.2, 0.5, 0.8),
                       colors=['k', 'cornflowerblue', 'k'])
    cont.clabel(fmt='%1.1f', fontsize=10)
    plt.grid(True)


# test ---
plt.figure(figsize=(3,3))
W=[-1, -1, -1]
show_contour_logistic2(W)
```


![png](output_16_0.png)



##### [리스트 6-2-(9)]로 상호 엔트로피 오차를 계산하는 함수를 정의합니다. [리스트 6-2-(10)]에서 편미분을 계산하는 함수를 정의합니다. 실행하면 W=[-1, -1, -1]의 경우 편미분값이 반환됩니다.

```python
# 리스트 6-2-(9)
# 크로스 엔트로피 오차 ------------
def cee_logistic2(w, x, t):
    X_n = x.shape[0]
    y = logistic2(x[:, 0], x[:, 1], w)
    cee = 0
    for n in range(len(y)):
        cee = cee - (t[n, 0] * np.log(y[n]) +
                     (1 - t[n, 0]) * np.log(1 - y[n]))
    cee = cee / X_n
    return cee
```


```python
# 리스트 6-2-(10)
# 크로스 엔트로피 오차의 미분 ------------
def dcee_logistic2(w, x, t):
    X_n=x.shape[0]
    y = logistic2(x[:, 0], x[:, 1], w)
    dcee = np.zeros(3)
    for n in range(len(y)):
        dcee[0] = dcee[0] + (y[n] - t[n, 0]) * x[n, 0]
        dcee[1] = dcee[1] + (y[n] - t[n, 0]) * x[n, 1]
        dcee[2] = dcee[2] + (y[n] - t[n, 0])
    dcee = dcee / X_n
    return dcee


# test ---
W=[-1, -1, -1]
dcee_logistic2(W, X, T2)
```


    array([ 0.10272008,  0.04450983, -0.06307245])





##### 평균 교차 엔트로피 오차가 최소가 되도록 로지스틱 회귀 모델의 매개 변수를 구하 고, 결과를 표시합니다. (리스트 6-2-(11) )


```python
# 리스트 6-2-(11)
from scipy.optimize import minimize


# 로지스틱 회귀 모델의 매개 변수 검색 -
def fit_logistic2(w_init, x, t):
    res = minimize(cee_logistic2, w_init, args=(x, t),
                   jac=dcee_logistic2, method="CG")
    return res.x


# 메인 ------------------------------------
plt.figure(1, figsize=(7, 3))
plt.subplots_adjust(wspace=0.5)


Ax = plt.subplot(1, 2, 1, projection='3d')
W_init = [-1, 0, 0]
W = fit_logistic2(W_init, X, T2)
print("w0 = {0:.2f}, w1 = {1:.2f}, w2 = {2:.2f}".format(W[0], W[1], W[2]))
show3d_logistic2(Ax, W)


show_data2_3d(Ax, X, T2)
cee = cee_logistic2(W, X, T2)
print("CEE = {0:.2f}".format(cee))


Ax = plt.subplot(1, 2, 2)
show_data2(X, T2)
show_contour_logistic2(W)
plt.show()
```

    w0 = -3.70, w1 = -2.54, w2 = -0.28
    CEE = 0.22



![png](output_19_1.png)



##### [리스트 6-2-(12)]에서 3클래스용 로지스틱 회귀 모델 logistic3을 구현합니다.

```python
# 리스트 6-2-(12)
# 3 클래스 용 로지스틱 회귀 모델 -----------------


def logistic3(x0, x1, w):
    K = 3
    w = w.reshape((3, 3))
    n = len(x1)
    y = np.zeros((n, K))
    for k in range(K):
        y[:, k] = np.exp(w[k, 0] * x0 + w[k, 1] * x1 + w[k, 2])
    wk = np.sum(y, axis=1)
    wk = y.T / wk
    y = wk.T
    return y


# test ---
W = np.array([1, 2, 3, 4 ,5, 6, 7, 8, 9])
y = logistic3(X[:3, 0], X[:3, 1], W)
print(np.round(y, 3))
```

    [[0.    0.006 0.994]
     [0.965 0.033 0.001]
     [0.925 0.07  0.005]]



##### [리스트 6-2-(13)]에서 교차 엔트로피 오차를 계산하는 함수인 cee_logistic3을 정의합니다. 9개 요소의 배열 W와, X, T3를 인수로 스칼라 값을 출력합니다.

```python
# 리스트 6-2-(13)
# 교차 엔트로피 오차 ------------
def cee_logistic3(w, x, t):
    X_n = x.shape[0]
    y = logistic3(x[:, 0], x[:, 1], w)
    cee = 0
    N, K = y.shape
    for n in range(N):
        for k in range(K):
            cee = cee - (t[n, k] * np.log(y[n, k]))
    cee = cee / X_n
    return cee


# test ----
W = np.array([1, 2, 3, 4 ,5, 6, 7, 8, 9])
cee_logistic3(W, X, T3)
```


    3.9824582404787288



##### [리스트 6-2-(14)]는 각 매개 변수에 대한 미분값을 출력하는 함수 dcee_logistic3입니다.


```python
# 리스트 6-2-(14)
# 교차 엔트로피 오차의 미분 ------------
def dcee_logistic3(w, x, t):
    X_n = x.shape[0]
    y = logistic3(x[:, 0], x[:, 1], w)
    dcee = np.zeros((3, 3)) # (클래스의 수 K) x (x의 차원 D+1)
    N, K = y.shape
    for n in range(N):
        for k in range(K):
            dcee[k, :] = dcee[k, :] - (t[n, k] - y[n, k])* np.r_[x[n, :], 1]
    dcee = dcee / X_n
    return dcee.reshape(-1)


# test ----
W = np.array([1, 2, 3, 4 ,5, 6, 7, 8, 9])
dcee_logistic3(W, X, T3)

```


    array([ 0.03778433,  0.03708109, -0.1841851 , -0.21235188, -0.44408101,
           -0.38340835,  0.17456754,  0.40699992,  0.56759346])



##### 이전의 출력은 OE/Owki에 대응한  요소 수 9개의 배열입니다. 이를 minimize ()에 전달하여 매개 변수 검색을 수행하는 함수를 만듭니다.(리스트 6-2- (15) ). 등고선에 결과를 표시하는 함수 show_contour_logistic3도 만들어둡니다.(리스트 6-2- (16) )


```python
# 리스트 6-2-(15)
# 매개 변수 검색 -----------------
def fit_logistic3(w_init, x, t):
    res = minimize(cee_logistic3, w_init, args=(x, t),
                   jac=dcee_logistic3, method="CG")
    return res.x
```


```python
# 리스트 6-2-(16)
# 모델 등고선 2D 표시 --------------------
def show_contour_logistic3(w):
    xn = 30 # 파라미터의 분할 수
    x0 = np.linspace(X_range0[0], X_range0[1], xn)
    x1 = np.linspace(X_range1[0], X_range1[1], xn)


    xx0, xx1 = np.meshgrid(x0, x1)
    y = np.zeros((xn, xn, 3))
    for i in range(xn):
        wk = logistic3(xx0[:, i], xx1[:, i], w)
        for j in range(3):
            y[:, i, j] = wk[:, j]
    for j in range(3):
        cont = plt.contour(xx0, xx1, y[:, :, j],
                           levels=(0.5, 0.9),
                           colors=['cornflowerblue', 'k'])
        cont.clabel(fmt='%1.1f', fontsize=9)
    plt.grid(True)
```


##### [리스트 6-2-(16)]의 show_contour_logistic3은 가중치 매개 변수 w를 전달하면, 표시할 입력 공간을 30×30으로 분할하여 모든 입력에 대해 네트워크의 출력을 확인합니다. 그리고 각각의 카테고리에서 0.5 또는 0.9 이상의 출력을 얻을 수 영역을 등고선으로 표시합니다. [리스트 6-2-(17)]에 피팅합니다.


```python
# 리스트 6-2-(17)
# 메인 ------------------------------------
W_init = np.zeros((3, 3))
W = fit_logistic3(W_init, X, T3)
print(np.round(W.reshape((3, 3)),2))
cee = cee_logistic3(W, X, T3)
print("CEE = {0:.2f}".format(cee))


plt.figure(figsize=(3, 3))
show_data2(X, T3)
show_contour_logistic3(W)
plt.show()
```

    [[-3.2  -2.69  2.25]
     [-0.49  4.8  -0.69]
     [ 3.68 -2.11 -1.56]]
    CEE = 0.23



![png](output_25_1.png)

