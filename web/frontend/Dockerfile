# 前端镜像：多阶段构建
# 阶段 1：Node 构建产物（npm run build → dist/）
FROM node:22-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# 阶段 2：nginx 只带静态产物（最终镜像很小，不含 Node）
FROM nginx:alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
