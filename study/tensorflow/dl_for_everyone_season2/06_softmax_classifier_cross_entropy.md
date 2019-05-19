# Softmax function
Softmax function은 다중 분류를 하기 위한 함수로 H(x)를 확률 값으로 표현해준다.  
아래 그림에서 볼 수 있듯이 각각의 입력을 확률로 바꿔주고 있다.  
<img src="./img/softmax_function.png" width="80%">  
  
## Cost function  
Softmax의 Costfunction으로는 cross entropy를 사용한다.  

<img src="./img/softmax_cost_function.png" width="80%">



## Implementation

### cost function: cross entropy

<img src="./img/softmax_cross_entropy_with_logits.png" width="80%">

### H(x), cost function

<img src="./img/softmax_hypothesis_cost_function.png" width="80%">



### gradient function, accuracy

<img src="./img/gradient_function_accuracy.png" width="80%">

## Training !!!!

<img src="./img/softmax_training.png" width="80%">





