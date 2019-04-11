# ML lec01 기본적인 Machine Learning 의 용어와 개념 설명

머신러닝, 딥러닝의 목표는 룰을 찾아서 예측을 할수 있는 시스템을 만드는 것이 목표다.  
이 목표를 찾기 위해서는 학습을 시켜서 룰을 찾는데 데이터는 라벨링된 데이터를 사용한다.  
라벨링된 데이터를 사용하기 때문에 Supervised Learning이라고 한다.  
라벨링되지 않은 데이터를 사용하는 것을 Unsupervised learning이라고 한다.  

여기서는 주로 Supervised learning에 대해 살펴보게 될 것이다.  
예를 들어서 Supervised learning을 구현할 수 있는 종류가 뭐가 있는지 보자. 
얼만큼 시간을 공부했느냐에 따라 Pass, 점수(score)를 예측하고자 한다 (regression)  
얼만큼 시간을 공부했느냐에 따라 Pass, Non-Pass를 예측하고자 한다 (Binary classification)  
얼만큼 시간을 공부했느냐에 따라 등급 (A, B, C, ~ ,F)를 예측하고자 한다 (multi-label classification)  

linear regression을 어떻게 모델을 만들고 어떻게 학습을 시키는지..