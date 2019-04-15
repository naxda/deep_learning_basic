# Convolutional Neural Network

[참고]  
https://hunkim.github.io/ml/lec11.pdf  
https://drive.google.com/drive/folders/1twBsdLkI2P15J0DgYs77_E_EVKt7Ghav  

image 분류에 가장 많이 사용되는 분류 방법인데, 뇌가 이미지를 인식하는 방법에서 힌트를 얻었다고 한다.

아래 그림을 보면, 그림의 한 부분에 대해 인식하는 뇌의 부분이 모두 다른 것을 보여주고 있는데, Convolution neural network도 이와 비슷한 원리로 구현된다.

기본적으로 Convolution, Pooling, Fully Connected layer로 구성되어 있다.

<img src="./img/cnn_overview.jpg" width="50%">



아래 그림의 Convolution, Pooling layer는 특정한 특징을 추출하는 layer이고.(feature extraction)

Fully connected layer는 분류를 하는 layer이다. (사진이 강아지인지 고양이인지, classification)
<img src="./img/convolution_neural_network.jpg"/>


## Convolution layer
--------

- filter의 갯수 
  filter를 통과하면 하나의 숫자가 나온다.  
  filter의 갯수에 따라 output map의 갯수가 결정되고, 다음 layer의 input map의 갯수가 된다.  
  
<img src="./img/filter.jpg" width="50%">  

stride가 1이라고 가정했을 때, filter의 갯수에 따라 출력맵의 갯수가 결정되는 것을 볼 수 있다.(필터의 갯수와 출력맵의 갯수는 동일)  
<img src="./img/convolution_layer_one_filter.jpg" width=200>
<img src="./img/convolution_layer_two_filter.jpg" width=270>
<img src="./img/convolution_layer_six_filter.jpg" width=350>

<img src="./img/convolution_layer_filters.jpg" width=500>  

- filter computing
convolution 연산의 계산은 아래와 같이 이미지의 픽셀 값과 필터의 각 원소들의 합으로 계산된다.
<img src="./img/filter_computing.jpg" float:left/>

- stride, padding
stride란 filter가 움직이는 간격의 크기를 의미한다.  
stride 값과 filter, input map의 크기에 따라 생성되는 출력 맵의 크기가 달라지게 되는데,  
출력되는 맵의 크기는 (width - filter width)/stride + 1이다.  
filter를 거치면서 out map의 크기가 점점 작아지게 되는데, 이로 인해 data의 손실이 일어날 수 있다.
이를 방지하기 위해 padding을 사용하는데, 아래 그림처럼 filter의 크기에 따라 padding을 달리하면 outmap의 크기가  
input map의 크기와 같아지는 것을 볼 수 있다.  
<img src="./img/padding_stride.png" width=500/>  


## Pooling Layer
-----

