#!/bin/bash
#===============================================================================
# Kaggle项目准备脚本
#
# 功能:
#   1. 清理不必要的文件（缓存、checkpoints等）
#   2. 压缩项目为 MCM_kaggle.zip
#   3. 准备上传到Kaggle数据集
#
# 使用方法:
#   cd MCM  # 进入项目根目录
#   bash scripts/configs/kaggle_hyperparam_search/prepare_for_kaggle.sh
#===============================================================================

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}===============================================================================${NC}"
echo -e "${BLUE}Kaggle项目准备脚本${NC}"
echo -e "${BLUE}===============================================================================${NC}"
echo ""

# 检查是否在项目根目录
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}错误: 请在项目根目录运行此脚本${NC}"
    echo -e "${YELLOW}当前目录: $(pwd)${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 当前目录: $(pwd)${NC}"
echo ""

# 1. 清理Python缓存
echo -e "${BLUE}[1/5] 清理Python缓存...${NC}"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
echo -e "${GREEN}✓ Python缓存已清理${NC}"
echo ""

# 2. 清理checkpoints（可选，节省空间）
echo -e "${BLUE}[2/5] 清理checkpoints目录...${NC}"
read -p "是否删除checkpoints目录？(y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf checkpoints/*
    echo -e "${GREEN}✓ checkpoints已清理${NC}"
else
    echo -e "${YELLOW}⊘ 保留checkpoints${NC}"
fi
echo ""

# 3. 清理日志文件（可选）
echo -e "${BLUE}[3/5] 清理日志文件...${NC}"
read -p "是否删除log目录？(y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf log/*
    echo -e "${GREEN}✓ 日志已清理${NC}"
else
    echo -e "${YELLOW}⊘ 保留日志${NC}"
fi
echo ""

# 4. 清理.git（可选，大幅减小体积）
echo -e "${BLUE}[4/5] 清理.git目录...${NC}"
read -p "是否删除.git目录？(这会删除Git历史！) (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf .git
    echo -e "${GREEN}✓ .git已清理${NC}"
else
    echo -e "${YELLOW}⊘ 保留.git${NC}"
fi
echo ""

# 5. 压缩项目
echo -e "${BLUE}[5/5] 压缩项目...${NC}"

OUTPUT_ZIP="MCM_kaggle.zip"

# 删除旧的压缩包
if [ -f "$OUTPUT_ZIP" ]; then
    rm "$OUTPUT_ZIP"
fi

# 压缩（排除不必要的文件）
zip -r "$OUTPUT_ZIP" . \
    -x "*.git*" \
    -x "*__pycache__*" \
    -x "*.pyc" \
    -x "*checkpoints/*" \
    -x "*log/*" \
    -x "*.zip" \
    -x "*test_outputs/*" \
    -x "*.ipynb_checkpoints*" \
    -q

FILE_SIZE=$(du -h "$OUTPUT_ZIP" | cut -f1)

echo -e "${GREEN}✓ 项目已压缩: $OUTPUT_ZIP (大小: $FILE_SIZE)${NC}"
echo ""

# 显示后续步骤
echo -e "${BLUE}===============================================================================${NC}"
echo -e "${BLUE}准备完成！${NC}"
echo -e "${BLUE}===============================================================================${NC}"
echo ""
echo -e "${GREEN}下一步:${NC}"
echo -e "  1. 访问 https://www.kaggle.com/datasets"
echo -e "  2. 点击 'New Dataset'"
echo -e "  3. 上传 $OUTPUT_ZIP"
echo -e "  4. 设置数据集名称（例如: mcm-project）"
echo -e "  5. 选择 Private（私有）"
echo -e "  6. 点击 Create"
echo ""
echo -e "${YELLOW}注意:${NC}"
echo -e "  - Kaggle会自动解压zip文件"
echo -e "  - 你的项目会在 /kaggle/input/<数据集名称>/ 下"
echo -e "  - 上传可能需要一些时间，取决于文件大小"
echo ""
echo -e "${BLUE}===============================================================================${NC}"
