# multi variable linear regression
전의 예에서는 변수가 하나뿐이었다 (공부한 시간)
이번에는 변수가 여러개라고 생각 해보자 (quiz1, quiz2, midterm)
행렬을 사용하면 표현식이 간단해진다. 프로그램 상에서 xw로 표현이 가능하다.
x의 컬럼과 w의 열의 수가 같아야 하기 때문에 XW로 표현을 한다.(텐서플로에서도 xw로 표현함)
입력은 (데이터의 갯수,데이터 안의 변수 갯수)


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
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE0MTE3NjUxOTBdfQ==
-->