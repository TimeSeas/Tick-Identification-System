# 后端镜像：Python + FastAPI + CPU 版 PyTorch（模型推理用）
FROM python:3.10-slim

#所有操作在容器内的/app目录
WORKDIR /app

# 先复制依赖文件再安装：利用 Docker 层缓存，代码改动不触发重新装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 \
       --index-url https://download.pytorch.org/whl/cpu

# 中文字体（标注图需要）：容器内没有 Windows 字体，必须随镜像携带
RUN apt-get update && apt-get install -y --no-install-recommends fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
