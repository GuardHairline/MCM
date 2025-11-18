#!/bin/bash
#===============================================================================
# Kaggle代码打包脚本 - 分离上传模式
#
# 功能:
#   1. 只打包代码部分（不含data和downloaded_model）
#   2. 压缩为 MCM_code.zip
#   3. 适合频繁修改代码后重新上传
#
# 使用方法:
#   cd MCM  # 进入项目根目录
#   bash scripts/configs/kaggle_hyperparam_search/prepare_code_only.sh
#
# 配合使用:
#   - 首次: prepare_data_only.sh (上传一次即可)
#   - 每次修改代码后: prepare_code_only.sh
#===============================================================================

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}===============================================================================${NC}"
echo -e "${BLUE}Kaggle代码打包脚本 - 只打包代码部分${NC}"
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
echo -e "${BLUE}[1/3] 清理Python缓存...${NC}"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
echo -e "${GREEN}✓ Python缓存已清理${NC}"
echo ""

# 2. 清理checkpoints和日志（可选）
echo -e "${BLUE}[2/3] 清理临时文件...${NC}"
read -p "是否删除checkpoints目录？(y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf checkpoints/*
    echo -e "${GREEN}✓ checkpoints已清理${NC}"
else
    echo -e "${YELLOW}⊘ 保留checkpoints${NC}"
fi

read -p "是否删除log目录？(y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf log/*
    echo -e "${GREEN}✓ log已清理${NC}"
else
    echo -e "${YELLOW}⊘ 保留log${NC}"
fi
echo ""

# 3. 压缩代码（排除data和downloaded_model）
echo -e "${BLUE}[3/3] 压缩代码...${NC}"

OUTPUT_ZIP="MCM_code.zip"

# 删除旧的压缩包
if [ -f "$OUTPUT_ZIP" ]; then
    rm "$OUTPUT_ZIP"
fi

# 压缩（排除data、downloaded_model和其他不必要文件）
zip -r "$OUTPUT_ZIP" . \
    -x "*.git*" \
    -x "*__pycache__*" \
    -x "*.pyc" \
    -x "*checkpoints/*" \
    -x "*log/*" \
    -x "*.zip" \
    -x "*test_outputs/*" \
    -x "*.ipynb_checkpoints*" \
    -x "data/*" \
    -x "downloaded_model/*" \
    -q

FILE_SIZE=$(du -h "$OUTPUT_ZIP" | cut -f1)

echo -e "${GREEN}✓ 代码已压缩: $OUTPUT_ZIP (大小: $FILE_SIZE)${NC}"
echo ""

# 显示后续步骤
echo -e "${BLUE}===============================================================================${NC}"
echo -e "${BLUE}代码打包完成！${NC}"
echo -e "${BLUE}===============================================================================${NC}"
echo ""
echo -e "${GREEN}文件信息:${NC}"
echo -e "  压缩包: $OUTPUT_ZIP"
echo -e "  大小: $FILE_SIZE (应该很小，几MB到几十MB)"
echo ""
echo -e "${GREEN}下一步:${NC}"
echo -e "  1. 访问 https://www.kaggle.com/datasets"
echo -e "  2. 找到你的 'mcm-code' 数据集"
echo -e "  3. 点击 'New Version' (新版本)"
echo -e "  4. 上传 $OUTPUT_ZIP"
echo -e "  5. 写版本说明（如：修复bug、添加功能等）"
echo -e "  6. 点击 'Create' 更新数据集"
echo ""
echo -e "${YELLOW}注意:${NC}"
echo -e "  • 数据集 'mcm-data' 不需要重新上传（除非data或模型文件改变）"
echo -e "  • 在Kaggle Notebook中需要同时添加两个数据集:"
echo -e "    - mcm-code (代码)"
echo -e "    - mcm-data (数据)"
echo ""
echo -e "${BLUE}===============================================================================${NC}"

