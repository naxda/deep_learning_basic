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
RUN apt-get update
RUN apt-get install -y python3.6
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install requests
RUN apt-get install -y python3-bs4

# for jupyter, tensorflow, keras package
RUN pip3 install tensorflow scikit-learn jupyter matplotlib pandas keras seaborn plotly

# excute jupyter notebook server
RUN pip3 install pip -U \
    && pip3 install jupyter_contrib_nbextensions \
    && rm -r /root/.cache/pip
RUN jupyter contrib nbextension install

# Set environment variables
# UBUNTU korean locale
RUN apt-get install -y locales
ENV LANGUAGE=ko_KR.UTF-8 
ENV LANG=ko_KR.UTF-8
RUN locale-gen ko_KR.UTF-8
ENV LC_ALL ko_KR.UTF-8

# make shared directory with HOST and set working directory
RUN mkdir -p /workdir/build
VOLUME /workdir/deeplearning_keras_study
<<<<<<< HEAD

WORKDIR /workdir
=======
VOLUME /workdir/study
WORKDIR /workdir/study
>>>>>>> 4493def2ee4b37107d0ba533d536f1625c39d3d1

# Set up our notebook config.
COPY jupyter_notebook_config.py /root/.jupyter/
COPY run_jupyter.sh /workdir
EXPOSE 8888
CMD ["/bin/bash"]