#!/bin/sh
docker run -p 8888:8888 -it --volume="$PWD/..:/workdir/deeplearning_keras_study" --volume="$PWD/build:/workdir/build" deeplearning_keras:imx-1
