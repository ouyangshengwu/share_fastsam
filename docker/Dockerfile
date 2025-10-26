# 使用最小的Python 3.12 slim镜像
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 更换系统源为阿里源，加速系统依赖安装
RUN sed -i s@/deb.debian.org/@/mirrors.aliyun.com/@g /etc/apt/sources.list && \
    sed -i s@/security.debian.org/@/mirrors.aliyun.com/debian-security/@g /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 设置pip阿里源，加速Python包安装
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip config set install.trusted-host mirrors.aliyun.com

# 安装Python依赖（新增flask-cors）
RUN pip install --no-cache-dir \
    opencv-python \
    numpy \
    ultralytics \
    flask \
    flask-cors

# 复制你的代码（如果需要）
# COPY . .

# 暴露Flask默认端口
# EXPOSE 5000

# 容器启动命令（示例：运行名为app.py的Flask应用）
# CMD ["python", "app.py"]