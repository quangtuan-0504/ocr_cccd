# Sử dụng image Python 3.11 cơ bản
FROM python:3.11.13-slim
 
# Đặt thư mục làm việc
WORKDIR /app
ARG PROXY_USER
ARG PROXY_PASS
ARG PROXY_HOST
ARG PROXY_PORT
 
ENV HTTP_PROXY="http://${PROXY_USER}:${PROXY_PASS}@${PROXY_HOST}:${PROXY_PORT}"
ENV HTTPS_PROXY="http://${PROXY_USER}:${PROXY_PASS}@${PROXY_HOST}:${PROXY_PORT}"
 
 
# Cài đặt các phụ thuộc hệ thống cần thiết
RUN apt-get update --option Acquire::HTTP::Proxy=http://${PROXY_USER}:${PROXY_PASS}@${PROXY_HOST}:${PROXY_PORT} && DEBIAN_FRONTEND=noninteractive apt-get install\
    libgl1\
    libgl1-mesa-glx \
    libglib2.0-0 -y  --option Acquire::HTTP::Proxy=http://${PROXY_USER}:${PROXY_PASS}@${PROXY_HOST}:${PROXY_PORT}  && \
    rm -rf /var/lib/apt/lists/*
 
# Liên kết python3 với python
RUN ln -s /usr/bin/python3.11 /usr/bin/python
 
RUN pip config set global.trusted-host "pypi.org files.pythonhosted.org pypi.python.org https://www.paddlepaddle.org.cn www.paddlepaddle.org.cn download.pytorch.org"
 
 
# Cài đặt các thư viện Python từ các nguồn cụ thể
ENV PIP_NO_VERIFY_CERTS=1
# RUN pip install --upgrade pip
 
 
 
RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
# RUN python -m pip install --no-cache-dir paddlepaddle-gpu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/ --trusted-host www.paddlepaddle.org.cn
RUN pip install --no-cache-dir paddlepaddle==3.1.0
RUN pip install --no-cache-dir paddleocr==3.1.0
RUN pip install --no-cache-dir fastapi[standard]
RUN pip install --no-cache-dir ultralytics==8.3.169 --no-deps
RUN pip install --no-cache-dir onnx==1.18.0
RUN pip install --no-cache-dir onnxruntime-gpu==1.22.0
 
 
 
ENV HTTP_PROXY=""
ENV HTTPS_PROXY=""
 
# Copy toàn bộ source code vào container
# COPY src/ ./src/
 
# Thiết lập biến môi trường
ENV PYTHONPATH=/app/src
 
# Mở port 8000 cho FastAPI
EXPOSE 8000
 
# Lệnh chạy ứng dụng
CMD ["python3", "src/main.py"]
 
# main instruction
# https://grok.com/chat/363ad30c-43fb-4128-bd51-a0d157d8256f
# maker pdf experiments
 
# Fix bug: ko authen được apt-get
# how to set apt proxy command line
# https://unix.stackexchange.com/questions/590437/specifying-apt-proxy-from-the-command-line
 
# get proxy từ .env
# https://chatgpt.com/c/688adbc3-c6bc-8012-b2de-5e59e8194d69
 
 
# Fix bug: ko authen được pip install
# pip config set global.trusted-host
# https://chatgpt.com/c/688ae974-7044-8012-b42b-f7d9f71bfd5e
# how to set pip proxy command line
# https://stackoverflow.com/questions/19080352/how-to-get-pip-to-work-behind-a-proxy-server
# https://chatgpt.com/c/688ae0a2-2994-8012-80d5-5ede6e3879b7
 
 
# BUILD IMAGE COMMAND:
# docker build \
#   --build-arg PROXY_USER=$(grep PROXY_USER .env-docker | cut -d '=' -f2) \
#   --build-arg PROXY_PASS=$(grep PROXY_PASS .env-docker | cut -d '=' -f2) \
#   --build-arg PROXY_HOST=$(grep PROXY_HOST .env-docker | cut -d '=' -f2) \
#   --build-arg PROXY_PORT=$(grep PROXY_PORT .env-docker | cut -d '=' -f2) \
#   -t ocr-app .
# RUN CONTAINER COMMAND
# docker run --gpus all \
#   -v $(pwd)/model:/app/model \
#   -v $(pwd)/src:/app/src \
#   -p 8000:8000 \
#   --name ocr-container \
#   ocr-app
