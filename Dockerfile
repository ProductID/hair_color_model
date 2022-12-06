ARG AARCH64_BASE_IMAGE=nvidia/cuda:11.4.0-devel-ubuntu20.04
FROM ${AARCH64_BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_CROSS_VERSION=11-4
ENV CUDA_CROSS_VERSION_DOT=11.4

WORKDIR /
ADD / .
CMD lsmod | grep nvidia
CMD rmmod nvidia_drm
CMD rmmod nvidia_modeset
CMD rmmod nvidia_uvm
CMD rmmod nvidia

RUN apt-get update && apt-get install -y git
RUN apt install -y gcc clang clang-tools cmake python3 python3-pip
RUN apt install nvidia-cuda-toolkit -y

RUN pip3 install --upgrade pip
RUN pip3 install dbus-next
RUN apt-get -y install cmake

RUN pip3 install -r preq.txt


# ENV FORCE_CUDA="1"
RUN pip3 install torch==1.13.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
# RUN pip3 install cuda-python
# Run pip3 install torch==1.11.0+cu115 torchvision==0.12.0+cu115 torchaudio==0.11.0+cu115 -f
#RUN pip3 install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
# RUN pip3 install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip3 install boto3
# We add the banana boilerplate here
ADD server.py .

# Add your model weight files
# (in this case we have a python script)
ADD download.py .
#RUN python3 download.py

# Add your custom app code, init() and inference()
ADD app.py .

EXPOSE 8000

CMD python3 -u server.py
