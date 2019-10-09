FROM tensorflow/tensorflow:1.15.0rc2-gpu-py3

ADD sources.list /etc/apt/sources.list
RUN apt-get install  apt-transport-https
RUN apt-get update && apt-get install -y \
  libsm6 \
  libxrender1 \
  libfontconfig1 \
  libxext6 \
  git \
  vim \
  wget

COPY requirements.txt .
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pip -U \
 && pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install --upgrade cython
RUN pip install --no-cache-dir -r requirements.txt
ENTRYPOINT bash