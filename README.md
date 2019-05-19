# Docker build  
 - cd docker  
 - docker build -t deeplearning_keras:imx-1 .
  
# Run Docker container 
  - docker run -p 8888:8888 -it --volume="$PWD/..:/workdir/deeplearning_keras_study" -v ${PWD}/build:/workdir/build deeplearning_keras:imx-1
  - or ./run_container.sh

# Docker 컨테이너, 이미지 삭제
 - Docker 컨테이너 모두 삭제
 docker rm $(docker ps -a -q)
 - Docker 이미지 모두 삭제
 docker rmi $(docker images -q)
 - 특정 이미지만 삭제
 (image 리스트 확인)
 docker images
 docker rmi 이미지id

# execute jupyter notebook in Docker Container
  - /workdir/run_jupyter.sh --allow-root &

# 특이점
  - ubuntu에서 docker build시에 plotly 패키지를 설치하면서 error가 발생해서 repository를 통해서 설치하도록 수정했다.