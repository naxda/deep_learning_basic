#!/bin/sh
docker run -p 8888:8888 -it --volume="$PWD/../study:/workdir/study" deeplearning_keras:imx-1
