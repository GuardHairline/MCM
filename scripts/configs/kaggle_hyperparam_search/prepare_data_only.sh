#!/bin/bash
#===============================================================================
# Kaggle数据打包脚本 - 分离上传模式
#
# 功能:
#   1. 只打包data和downloaded_model目录
#   2. 压缩为 MCM_data.zip
#   3. 只需上传一次（除非数据改变）
#
# 使用方法:
#   cd MCM  # 进入项目根目录
#   bash scripts/configs/kaggle_hyperparam_search/prepare_data_only.sh
#
# 配合使用:
#   - 首次: prepare_data_only.sh (本脚本，上传一次)
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
echo -e "${BLUE}Kaggle数据打包脚本 - 只打包data和模型${NC}"
echo -e "${BLUE}===============================================================================${NC}"
echo ""

# 检查是否在项目根目录
if [ ! -d "data" ]; then
    echo -e "${RED}错误: 未找到data目录，请在项目根目录运行此脚本${NC}"
    echo -e "${YELLOW}当前目录: $(pwd)${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 当前目录: $(pwd)${NC}"
echo ""

# 检查data目录
echo -e "${BLUE}[1/2] 检查数据目录...${NC}"

if [ -d "data" ]; then
    DATA_SIZE=$(du -sh data | cut -f1)
    echo -e "${GREEN}✓ data目录存在 (大小: $DATA_SIZE)${NC}"
else
    echo -e "${RED}✗ data目录不存在${NC}"
    exit 1
fi

if [ -d "downloaded_model" ]; then
    MODEL_SIZE=$(du -sh downloaded_model | cut -f1)
    echo -e "${GREEN}✓ downloaded_model目录存在 (大小: $MODEL_SIZE)${NC}"
else
    echo -e "${YELLOW}⚠ downloaded_model目录不存在${NC}"
fi
echo ""

# 压缩数据
echo -e "${BLUE}[2/2] 压缩数据和模型...${NC}"

OUTPUT_ZIP="MCM_data.zip"

# 删除旧的压缩包
if [ -f "$OUTPUT_ZIP" ]; then
    rm "$OUTPUT_ZIP"
fi

# 只打包data和downloaded_model
echo -e "${YELLOW}正在压缩，这可能需要几分钟...${NC}"

if [ -d "downloaded_model" ]; then
    zip -r "$OUTPUT_ZIP" data downloaded_model -q
else
    zip -r "$OUTPUT_ZIP" data -q
fi

FILE_SIZE=$(du -h "$OUTPUT_ZIP" | cut -f1)

echo -e "${GREEN}✓ 数据已压缩: $OUTPUT_ZIP (大小: $FILE_SIZE)${NC}"
echo ""

# 显示后续步骤
echo -e "${BLUE}===============================================================================${NC}"
echo -e "${BLUE}数据打包完成！${NC}"
echo -e "${BLUE}===============================================================================${NC}"
echo ""
echo -e "${GREEN}文件信息:${NC}"
echo -e "  压缩包: $OUTPUT_ZIP"
echo -e "  大小: $FILE_SIZE (应该很大，几GB)"
echo ""
echo -e "${GREEN}下一步:${NC}"
echo -e "  1. 访问 https://www.kaggle.com/datasets"
echo -e "  2. 点击 'New Dataset'"
echo -e "  3. 上传 $OUTPUT_ZIP"
echo -e "  4. 设置数据集名称: mcm-data"
echo -e "  5. 设置为 Private（私有）"
echo -e "  6. 点击 'Create'"
echo ""
echo -e "${YELLOW}注意:${NC}"
echo -e "  • 这个数据集只需上传一次"
echo -e "  • 上传可能需要较长时间（取决于文件大小和网速）"
echo -e "  • 上传后Kaggle会自动解压"
echo -e "  • 除非data或模型文件改变，否则不需要重新上传"
echo ""
echo -e "${GREEN}完成数据上传后:${NC}"
echo -e "  运行 prepare_code_only.sh 打包代码部分"
echo ""
echo -e "${BLUE}===============================================================================${NC}"

