source /d/ProgramData/anaconda3/etc/profile.d/conda.sh
conda activate pytorch
#!/bin/bash

# 运行所有本地配置的训练脚本
# 自动运行scripts/configs/中所有local_开头的配置文件

# 增加文件描述符限制，解决"Too many open files"问题
ulimit -n 65536

echo "=== 运行所有本地配置训练 ==="

# 检查configs目录是否存在
if [ ! -d "scripts/configs" ]; then
    echo "错误: scripts/configs 目录不存在"
    echo "请先运行 scripts/generate_all_local_configs.sh 生成配置文件"
    exit 1
fi

# 获取所有local配置文件
local_configs=$(ls scripts/configs/local_*.json 2>/dev/null)

if [ -z "$local_configs" ]; then
    echo "错误: 没有找到local配置文件"
    echo "请先运行 scripts/generate_all_local_configs.sh 生成配置文件"
    exit 1
fi

echo "找到以下本地配置文件:"
echo "$local_configs"
echo ""

# 计数器
total_configs=0
completed_configs=0
failed_configs=0

# 创建日志目录
mkdir -p scripts/log/local_training

# 记录开始时间
start_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "开始时间: $start_time"
echo ""

# 运行每个配置文件
for config_file in $local_configs; do
    total_configs=$((total_configs + 1))
    
    # 提取配置文件名（不含路径和扩展名）
    config_name=$(basename "$config_file" .json)
    
    echo "=== 运行配置: $config_name ==="
    echo "配置文件: $config_file"
    
    # 创建日志文件
    log_file="scripts/log/local_training/${config_name}.log"
    
    # 运行训练脚本
    echo "开始训练..."
    echo "日志文件: $log_file"
    
    # 使用timeout防止无限运行，设置最大运行时间为2小时
    timeout 7200 python -m scripts.train_with_zero_shot --config "$config_file" > "$log_file" 2>&1
    
    # 检查运行结果
    if [ $? -eq 0 ]; then
        echo "✓ 配置 $config_name 训练完成"
        completed_configs=$((completed_configs + 1))
        
        # 删除生成的.pt文件以节省空间
        echo "清理生成的模型文件..."
        # 提取配置名称（去掉路径和扩展名）
        config_basename=$(basename "$config_file" .json)
        # 删除该配置生成的所有模型文件
        rm -f "checkpoints/model_${config_basename}.pt"
        rm -f "checkpoints/model_${config_basename}_task_heads.pt"
        rm -f "checkpoints/label_embedding_${config_basename}.pt"
        rm -f "checkpoints/train_info_${config_basename}.json"
        echo "模型文件清理完成"
    else
        echo "✗ 配置 $config_name 训练失败"
        failed_configs=$((failed_configs + 1))
        echo "查看日志: tail -f $log_file"
        
        # 即使失败也清理.pt文件
        echo "清理生成的模型文件..."
        # 提取配置名称（去掉路径和扩展名）
        config_basename=$(basename "$config_file" .json)
        # 删除该配置生成的所有模型文件
        rm -f "checkpoints/model_${config_basename}.pt"
        rm -f "checkpoints/model_${config_basename}_task_heads.pt"
        rm -f "checkpoints/label_embedding_${config_basename}.pt"
        rm -f "checkpoints/train_info_${config_basename}.json"
        echo "模型文件清理完成"
    fi
    
    echo ""
done

# 记录结束时间
end_time=$(date +"%Y-%m-%d %H:%M:%S")

echo "=== 训练完成统计 ==="
echo "开始时间: $start_time"
echo "结束时间: $end_time"
echo "总配置文件数: $total_configs"
echo "成功完成: $completed_configs"
echo "失败数量: $failed_configs"
echo ""

if [ $failed_configs -gt 0 ]; then
    echo "失败的配置文件:"
    for config_file in $local_configs; do
        config_name=$(basename "$config_file" .json)
        log_file="scripts/log/local_training/${config_name}.log"
        if [ -f "$log_file" ]; then
            # 检查日志文件最后几行是否有错误
            if tail -n 5 "$log_file" | grep -q "error\|Error\|ERROR\|Exception\|Traceback"; then
                echo "  - $config_name"
            fi
        fi
    done
    echo ""
    echo "查看详细日志: ls -la scripts/log/local_training/"
fi

echo "所有训练日志保存在: scripts/log/local_training/" 