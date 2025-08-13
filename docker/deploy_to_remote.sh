#!/bin/bash

# 部署脚本：将本地修改的文件拷贝到远程服务器
# 远程服务器信息
REMOTE_USER="root"
REMOTE_HOST="116.62.22.201"
REMOTE_BASE_PATH="/mnt/code/ragflow"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}开始部署文件到远程服务器 ${REMOTE_USER}@${REMOTE_HOST}${NC}"

# 定义需要拷贝的文件数组
LOCAL_FILES=(
    "./entrypoint.sh"
    "../api/apps/sdk/doc.py"
    "../api/db/services/file_service.py"
    "../rag/utils/minio_conn.py"
    "../rag/svr/task_executor.py"
    "../deepdoc/parser/monkeyocr_parser.py"
    "../deepdoc/parser/__init__.py"
    "../rag/app/naive.py"
    "../rag/app/presentation.py"
    "../rag/nlp/__init__.py"
    "../api/db/services/document_service.py"
    "../api/db/services/document_content_service.py"
    "../api/db/db_models.py"
    "../api/db/services/__init__.py"
    "../api/db/services/task_service.py"
    "../api/db/services/conversation_service.py"
    "../api/db/services/canvas_service.py"
    "../api/db/__init__.py"
    "../api/apps/api_app.py"
    "../api/apps/conversation_app.py"
    "../api/apps/sdk/session.py"
    "../rag/app/table.py"
    "../deepdoc/parser/excel_parser.py"
    "../rag/app/qa.py"
    "./docker-compose.yml"
    "../api/utils/api_utils.py"
    "../rag/app/report.py"
    "../api/utils/validation_utils.py"
    "../api/apps/sdk/dataset.py"
)

REMOTE_FILES=(
    "${REMOTE_BASE_PATH}/docker/entrypoint.sh"
    "${REMOTE_BASE_PATH}/api/apps/sdk/doc.py"
    "${REMOTE_BASE_PATH}/api/db/services/file_service.py"
    "${REMOTE_BASE_PATH}/rag/utils/minio_conn.py"
    "${REMOTE_BASE_PATH}/rag/svr/task_executor.py"
    "${REMOTE_BASE_PATH}/deepdoc/parser/monkeyocr_parser.py"
    "${REMOTE_BASE_PATH}/deepdoc/parser/__init__.py"
    "${REMOTE_BASE_PATH}/rag/app/naive.py"
    "${REMOTE_BASE_PATH}/rag/app/presentation.py"
    "${REMOTE_BASE_PATH}/rag/nlp/__init__.py"
    "${REMOTE_BASE_PATH}/api/db/services/document_service.py"
    "${REMOTE_BASE_PATH}/api/db/services/document_content_service.py"
    "${REMOTE_BASE_PATH}/api/db/db_models.py"
    "${REMOTE_BASE_PATH}/api/db/services/__init__.py"
    "${REMOTE_BASE_PATH}/api/db/services/task_service.py"
    "${REMOTE_BASE_PATH}/api/db/services/conversation_service.py"
    "${REMOTE_BASE_PATH}/api/db/services/canvas_service.py"
    "${REMOTE_BASE_PATH}/api/db/__init__.py"
    "${REMOTE_BASE_PATH}/api/apps/api_app.py"
    "${REMOTE_BASE_PATH}/api/apps/conversation_app.py"
    "${REMOTE_BASE_PATH}/api/apps/sdk/session.py"
    "${REMOTE_BASE_PATH}/rag/app/table.py"
    "${REMOTE_BASE_PATH}/deepdoc/parser/excel_parser.py"
    "${REMOTE_BASE_PATH}/rag/app/qa.py"
    "${REMOTE_BASE_PATH}/docker/docker-compose.yml"
    "${REMOTE_BASE_PATH}/api/utils/api_utils.py"
    "${REMOTE_BASE_PATH}/rag/app/report.py"
    "${REMOTE_BASE_PATH}/api/utils/validation_utils.py"
    "${REMOTE_BASE_PATH}/api/apps/sdk/dataset.py"
)

# 校验本地与远程文件数组长度是否一致，防止错位拷贝
if [[ ${#LOCAL_FILES[@]} -ne ${#REMOTE_FILES[@]} ]]; then
    echo -e "${RED}错误: LOCAL_FILES 与 REMOTE_FILES 数量不一致 (${#LOCAL_FILES[@]} vs ${#REMOTE_FILES[@]})，已中止${NC}"
    exit 1
fi

# 检查本地文件是否存在
echo -e "${YELLOW}检查本地文件...${NC}"
for local_file in "${LOCAL_FILES[@]}"; do
    if [[ ! -f "$local_file" ]]; then
        echo -e "${RED}错误: 本地文件 $local_file 不存在${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓${NC} $local_file"
done

# 测试远程连接
echo -e "${YELLOW}测试远程连接...${NC}"
if ! ssh -o ConnectTimeout=5 ${REMOTE_USER}@${REMOTE_HOST} "echo '连接成功'" > /dev/null 2>&1; then
    echo -e "${RED}错误: 无法连接到远程服务器 ${REMOTE_USER}@${REMOTE_HOST}${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 远程连接测试成功${NC}"

# 拷贝文件
echo -e "${YELLOW}开始拷贝文件...${NC}"
for i in "${!LOCAL_FILES[@]}"; do
    local_file="${LOCAL_FILES[$i]}"
    remote_file="${REMOTE_FILES[$i]}"
    echo -e "${YELLOW}拷贝: $local_file → $remote_file${NC}"
    
    # 创建远程目录（如果不存在）
    remote_dir=$(dirname "$remote_file")
    ssh ${REMOTE_USER}@${REMOTE_HOST} "mkdir -p $remote_dir"
    
    # 拷贝文件
    if scp "$local_file" "${REMOTE_USER}@${REMOTE_HOST}:$remote_file"; then
        echo -e "${GREEN}✓ 拷贝成功: $local_file${NC}"
    else
        echo -e "${RED}✗ 拷贝失败: $local_file${NC}"
        exit 1
    fi
done

echo -e "${GREEN}所有文件拷贝完成！${NC}"

# 询问是否重启远程服务
echo -e "${YELLOW}是否需要重启远程服务器上的RAGFlow服务？(y/n)${NC}"
read -r restart_choice
if [[ "$restart_choice" =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}重启远程RAGFlow服务...${NC}"
    ssh ${REMOTE_USER}@${REMOTE_HOST} "cd ${REMOTE_BASE_PATH}/docker && docker compose down && docker compose up -d"
    echo -e "${GREEN}服务重启完成！${NC}"
fi

echo -e "${GREEN}部署完成！${NC}" 