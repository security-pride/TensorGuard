#!/bin/bash

# 设置工作目录和输出目录
MODELS_DIR="LLM_work/cluster_feature/models"
OUTPUT_DIR="LLM_work/cluster_feature/models/merged"

# 遍历models目录下的所有子目录
for MODEL_DIR in "$MODELS_DIR"/*/ ; do
    # 跳过merged目录
    if [[ "$MODEL_DIR" == *"/merged/"* ]]; then
        continue
    fi
    
    # 获取模型名称（目录名）
    MODEL_NAME=$(basename "$MODEL_DIR")
    echo "处理模型: $MODEL_NAME"
    
    # 检查是否已经存在合并后的文件（在merged目录或merged/*-ft目录中）
    if [ -f "$OUTPUT_DIR/$MODEL_NAME.safetensors" ]; then
        echo "模型 $MODEL_NAME 已经在merged目录中合并，跳过"
        continue
    fi

    # 检查merged目录下的merge_model子目录
    if [ -d "$OUTPUT_DIR/merge_model" ] && [ -f "$OUTPUT_DIR/merge_model/$MODEL_NAME.safetensors" ]; then
        echo "模型 $MODEL_NAME 已经在merge_model目录中合并，跳过"
        continue
    fi
    
    # 检查merged目录下的所有*-ft子目录
    FOUND_IN_FT=false
    for FT_DIR in "$OUTPUT_DIR"/*-ft/ ; do
        if [ -d "$FT_DIR" ] && [ -f "$FT_DIR/$MODEL_NAME.safetensors" ]; then
            echo "模型 $MODEL_NAME 已经在 $(basename "$FT_DIR") 目录中合并，跳过"
            FOUND_IN_FT=true
            break
        fi
    done
    
    if [ "$FOUND_IN_FT" = true ]; then
        continue
    fi
    
    # 使用merge_model.py脚本进行合并
    echo "开始合并模型: $MODEL_NAME"
    python LLM_work/cluster_feature/utils/merge_model.py --model_dir "$MODEL_DIR" --output_dir "$OUTPUT_DIR"
    
    echo "模型 $MODEL_NAME 处理完成"
    echo "----------------------------------------"
done

echo "所有模型处理完成！"