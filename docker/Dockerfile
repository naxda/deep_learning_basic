FROM ubuntu:16.04

RUN sed -i s/archive.ubuntu.com/ftp.daumkakao.com/g /etc/apt/sources.list
RUN apt-get clean
RUN apt-get update 
RUN apt-get install -y make 
RUN apt-get install -y git unzip 
RUN apt-get install -y wget
RUN apt-get install -y build-essential

# install python3.6, BeutifulSoup4 package
# for use "add-apt-repository command"
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:jonathonf/python-3.6
# RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
# RUN apt-get install -y python3.6
RUN apt-get install -y python3.6-dev
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install requests
RUN apt-get install -y python3-bs4

# Install BLAS Library to boost Tensor-Operation
RUN apt-get install -y build-essential \
cmake git unzip pkg-config libopenblas-dev liblapack-dev

# Install HDF5(대용량 수치 데이터파일을 효율적인 이진포맷으로 저장해준다.)
RUN apt-get install -y libhdf5-serial-dev python-h5py

# for jupyter, tensorflow, keras package
# RUN pip3 install tensorflow scikit-learn jupyter matplotlib pandas keras seaborn plotly
RUN pip3 install tensorflow scikit-learn jupyter matplotlib pandas keras seaborn

# # Install numpy, scipy, matplotlib
RUN pip3 install numpy scipy pyyaml

RUN pip3 uninstall -y tornado
RUN pip3 install tornado==5.1.1
# RUN pip3 install tornado

RUN pip3 uninstall -y pillow
RUN pip3 install pillow

# Set environment variables
# UBUNTU korean locale
RUN apt-get install -y locales
ENV LANGUAGE=ko_KR.UTF-8 
ENV LANG=ko_KR.UTF-8
RUN locale-gen ko_KR.UTF-8
ENV LC_ALL ko_KR.UTF-8

# make shared directory with HOST and set working directory
WORKDIR /workdir

# Set up our notebook config.
COPY jupyter_notebook_config.py /root/.jupyter/
COPY run_jupyter.sh /workdir
EXPOSE 8888
CMD ["/bin/bash"]
