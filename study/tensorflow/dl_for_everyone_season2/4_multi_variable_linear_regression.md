# multi variable linear regression
전의 예에서는 변수가 하나뿐이었다 (공부한 시간)  
이번에는 변수가 여러개라고 생각 해보자 (quiz1, quiz2, midterm)  
행렬을 사용하면 표현식이 간단해진다. 프로그램 상에서 xw로 표현이 가능하다.  
x의 컬럼과 w의 열의 수가 같아야 하기 때문에 XW로 표현을 한다.(텐서플로에서도 xw로 표현함)  
입력은 (데이터의 갯수,데이터 안의 변수 갯수)  
  
## 간단한 형태의 w1, w2, w3을 계산하는 코드  
~~~python
# data and label
x1 = [ 73.,  93.,  89.,  96.,  73.]
x2 = [ 80.,  88.,  91.,  98.,  66.]
x3 = [ 75.,  93.,  90., 100.,  70.]
Y  = [152., 185., 180., 196., 142.]

# random weights
w1 = tf.Variable(tf.random_normal([1]))
w2 = tf.Variable(tf.random_normal([1]))
w3 = tf.Variable(tf.random_normal([1]))
b  = tf.Variable(tf.random_normal([1]))

learning_rate = 0.000001

for i in range(1000+1):
    # tf.GradientTape() to record the gradient of the cost function
    with tf.GradientTape() as tape:
        hypothesis = w1 * x1 +  w2 * x2 + w3 * x3 + b
        cost = tf.reduce_mean(tf.square(hypothesis - Y))
    # calculates the gradients of the cost
    w1_grad, w2_grad, w3_grad, b_grad = tape.gradient(cost, [w1, w2, w3, b])
    
    # update w1,w2,w3 and b
    w1.assign_sub(learning_rate * w1_grad)
    w2.assign_sub(learning_rate * w2_grad)
    w3.assign_sub(learning_rate * w3_grad)
    b.assign_sub(learning_rate * b_grad)

    if i % 50 == 0:
      print("{:5} | {:12.4f}".format(i, cost.numpy()))
~~~  
  
## 위에서 본 각각의 w1, w2, w3을 계산하는 코드의 문제점
각각의 w1, w2, w3을 계산하는 코드는 x1,x2,x3,y의 값을 직접 입력해야 하는 번거로움이 있다.  
또, 코드의 크기가 커지고, 데이터의 크기도 커지면서 유지보수의 어려움이 생기는 문제가 있다.  
~~~python
for i in range(1000+1):
	with tf.GradientTape() as tape:
		hypothesis = w1 * x1 + w2 * x2 + w3 * x3 + b
		#COST함수를 구한다.
		cost = tf.reduce_mean(tf.square(hypothesis - Y))
	
	# tape에 기록된 하나의 값을 꺼내서 cost함수에 대한 W1,W2,W3,B의 각각 개별의 미분값을 가져온다.
	w1_grad, w2_grad, w3_grad, b_grad = tape.gradient(cost, [w1, w2, w3, b])
	
	# W1값을 업데이트한다.
	w1.assign_sub(learning_rate * w1_grad)
	w2.assign_sub(learning_rate * w2_grad)
	w3.assign_sub(learning_rate * w3_grad)
	
	if i % 50 == 0:
		print("{:5} | {:12.4f}".format(i, cost.numpy()))
~~~  
  
## 행렬(numpy)를 이용해서 x1,x2,x3,y 값을 정의
numpy slicing  
~~~python
data = np.array([
	#X1, X2, X3, y
	[73., 80., 75., 152.],
	[93., 88., 93., 185.],
	[89., 91., 90., 180.],
	[96., 98., 100., 196.],
	[73., 66., 70., 142.]
], dtype=np.float32)


# slice data
X = data[:, : -1]
y = data[:, [-1]]
~~~

## numpy를 사용하면 아래처럼 더 깔끔하게 코드를 구현이 가능하다  
~~~python
data = np.array([
    # X1,   X2,    X3,   y
    [ 73.,  80.,  75., 152. ],
    [ 93.,  88.,  93., 185. ],
    [ 89.,  91.,  90., 180. ],
    [ 96.,  98., 100., 196. ],
    [ 73.,  66.,  70., 142. ]
], dtype=np.float32)

# slice data
X = data[:, :-1]
y = data[:, [-1]]

W = tf.Variable(tf.random_normal([3, 1]))
b = tf.Variable(tf.random_normal([1]))

learning_rate = 0.000001

# hypothesis, prediction function
def predict(X):
    return tf.matmul(X, W) + b

print("epoch | cost")

n_epochs = 2000
for i in range(n_epochs+1):
    # tf.GradientTape() to record the gradient of the cost function
    with tf.GradientTape() as tape:
        cost = tf.reduce_mean((tf.square(predict(X) - y)))

    # calculates the gradients of the loss
    W_grad, b_grad = tape.gradient(cost, [W, b])

    # updates parameters (W and b)
    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)
    
    if i % 100 == 0:
        print("{:5} | {:10.4f}".format(i, cost.numpy()))
~~~
