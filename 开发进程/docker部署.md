后端 Dockerfile 逐行讲（web/backend/Dockerfile）

FROM python:3.10-slim          # ① 起点：官方 Python 基础镜像（slim=精简版，没有乱七八糟的额外软件）
WORKDIR /app                   # ② 之后所有操作都在容器内的 /app 目录

COPY requirements.txt .        # ③ 先只复制依赖清单（注意：不是复制全部代码！）
RUN pip install -r requirements.txt   # ④ 安装依赖
COPY app ./app                 # ⑤ 最后才复制代码

CMD ["uvicorn", "app.main:app", ...]  # ⑥ 容器启动时执行的命令
Docker的每一行命令生成一个层，层可以被缓存
如果直接复制代码，改代码就要重装依赖


前端 Dockerfile：多阶段构建（体现水平的关键设计）

FROM node:22-alpine AS build    # 阶段一：完整 Node 环境，负责把源码编译成 dist/
...
FROM nginx:alpine               # 阶段二：全新干净环境，只要 nginx + 编译产物
COPY --from=build /app/dist /usr/share/nginx/html

把node_modules 源码等编译，镜像体积小部署快