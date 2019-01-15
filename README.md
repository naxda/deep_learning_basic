# Docker build  
 - docker build -t deeplearning_keras:imx-1 .
  
# Run Docker container 
  - docker run -p 8888:8888 -it --volume="$PWD/..:/workdir/deeplearning_keras_study" -v ${PWD}/build:/workdir/build deeplearning_keras:imx-1
  - or ./run_container.sh

# execute jupyter notebook in Docker Container
  - /workdir/run_jupyter.sh --allow-root &

