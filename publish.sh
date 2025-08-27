#!/bin/bash

# RAGFlow 发布脚本
# 用法: ./publish.sh <版本号>
# 例如: ./publish.sh v1.0.0

set -e  # 遇到错误时退出

# 检查参数
if [ $# -ne 1 ]; then
    echo "错误: 请提供版本号参数"
    echo "用法: $0 <版本号>"
    echo "例如: $0 v1.0.0"
    exit 1
fi

VERSION=$1
REMOTE_HOST="root@116.62.22.201"
REMOTE_PATH="/mnt/code/ragflow/docker"
REGISTRY="visionfin-registry.cn-hangzhou.cr.aliyuncs.com/ai/ragflow"

echo "开始发布 RAGFlow 版本: $VERSION"
echo "=================================="

# 步骤1: 获取 ragflow-dotsocr:latest 的镜像ID
echo "步骤1: 获取 ragflow-dotsocr:latest 镜像ID..."
IMAGE_ID=$(docker images ragflow-dotsocr:latest --format "{{.ID}}")

if [ -z "$IMAGE_ID" ]; then
    echo "错误: 未找到 ragflow-dotsocr:latest 镜像"
    echo "请先构建镜像: docker build -f Dockerfile.dotsocr -t ragflow-dotsocr:latest ."
    exit 1
fi

echo "找到镜像ID: $IMAGE_ID"

# 步骤2: 执行 docker tag
echo "步骤2: 创建标签 $REGISTRY:$VERSION..."
docker tag "$IMAGE_ID" "$REGISTRY:$VERSION"
echo "标签创建成功"

# 步骤3: 执行 docker push
echo "步骤3: 推送镜像到远程仓库..."
docker push "$REGISTRY:$VERSION"
echo "镜像推送成功"

# 步骤4-7: SSH 连接并执行远程操作
echo "步骤4-7: 连接远程服务器并更新配置..."

# 使用免密SSH连接远程服务器
ssh -o StrictHostKeyChecking=no "$REMOTE_HOST" << 'EOF'
    set -e
    
    echo "已连接到远程服务器"
    
    # 步骤5: 切换到指定目录
    echo "切换到目录: $REMOTE_PATH"
    cd "$REMOTE_PATH"
    
    # 步骤6: 修改 .env 文件中 RAGFLOW_IMAGE 的版本号
    echo "更新 .env 文件中的镜像版本..."
    
    # 备份原文件
    cp .env .env.backup.$(date +%Y%m%d_%H%M%S)
    
    # 使用 sed 替换版本号（冒号后面的部分）
    # 匹配 RAGFLOW_IMAGE=xxx:yyy 格式，将 yyy 替换为新的版本号
    sed -i "s/^RAGFLOW_IMAGE=.*:.*/RAGFLOW_IMAGE=$REGISTRY:$VERSION/" .env
    
    echo "版本号已更新为: $REGISTRY:$VERSION"
    
    # 验证更新
    echo "更新后的配置:"
    grep "^RAGFLOW_IMAGE=" .env
    
    # 步骤7: 重启 ragflow 容器
    echo "重启 ragflow 容器..."
    
    # 检查 docker-compose 文件
    if [ -f "docker-compose.yml" ]; then
        echo "使用 docker-compose.yml 重启服务..."
        docker compose down
        docker compose up -d
    elif [ -f "docker-compose-ocr.yml" ]; then
        echo "使用 docker-compose-ocr.yml 重启服务..."
        docker compose -f docker-compose-ocr.yml down
        docker compose -f docker-compose-ocr.yml up -d
    else
        echo "警告: 未找到 docker-compose 文件，请手动重启容器"
        echo "可用的 docker-compose 文件:"
        ls -la docker-compose*.yml 2>/dev/null || echo "无"
    fi
    
    echo "容器重启完成"
    
    # 检查服务状态
    echo "检查服务状态..."
    docker ps --filter "name=ragflow" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    
    echo "远程操作完成"
EOF

echo "=================================="
echo "发布完成！"
echo "镜像已推送到: $REGISTRY:$VERSION"
echo "远程服务器配置已更新"
echo "容器已重启"
