# Must use a Cuda version 11+
FROM python:3.8.10
FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime
#FROM nvidia/cuda:11.0-base
WORKDIR /
ADD / .
# Install git
RUN apt-get update && apt-get install -y git
RUN apt install -y gcc clang clang-tools cmake python3
RUN apt install nvidia-cuda-toolkit -y
RUN apt install ninja-build -y
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8 -y
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8

# Install python packages
RUN pip3 install --upgrade pip
RUN pip3 install dbus-next
RUN apt-get -y install cmake
RUN pip3 install pyyaml==5.4.1
ADD preq.txt preq.txt
RUN pip3 install -r preq.txt


# RUN pip3 install torch==1.13.0+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip3 install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip3 install boto3

# We add the banana boilerplate here
ADD server.py .

# Add your model weight files
# (in this case we have a python script)
#ADD download.py .
#RUN python3 download.py


# Add your custom app code, init() and inference()
ADD app.py .

EXPOSE 8000

CMD python3 -u server.py
