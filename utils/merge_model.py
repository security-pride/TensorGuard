import os
import re
import json
import shutil
from tqdm import tqdm
from safetensors import safe_open
from safetensors.torch import save_file
import argparse

def merge_from_index_json(model_dir, output_dir):
    # 查找索引文件
    index_file = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.exists(index_file):
        raise ValueError(f"在目录 {model_dir} 中未找到索引文件 model.safetensors.index.json")
    
    # 读取索引文件
    with open(index_file, 'r', encoding='utf-8') as f:
        index_data = json.load(f)
    
    # 提取权重映射
    weight_map = index_data.get("weight_map", {})
    if not weight_map:
        raise ValueError("索引文件中未找到权重映射信息")
    
    # 获取所有分片文件
    shard_files = set(weight_map.values())
    print(f"发现 {len(shard_files)} 个分片文件")
    
    # 合并所有张量
    merged_tensors = {}
    for shard_file in tqdm(shard_files, desc="合并分片"):
        shard_path = os.path.join(model_dir, shard_file)
        if not os.path.exists(shard_path):
            raise ValueError(f"分片文件不存在: {shard_path}")
        
        with safe_open(shard_path, framework="pt") as f:
            # 获取该分片中的所有键
            keys = [key for key in f.keys()]
            
            # 加载每个张量
            for key in keys:
                if key in merged_tensors:
                    print(f"警告: 发现重复的张量键: {key}，将使用最后一个")
                merged_tensors[key] = f.get_tensor(key)
    
    # 保存合并后的文件
    os.makedirs(output_dir, exist_ok=True)
    model_name = os.path.basename(os.path.normpath(model_dir))
    output_path = os.path.join(output_dir, f"{model_name}.safetensors")
    
    print(f"正在保存合并后的模型到 {output_path}...")
    save_file(merged_tensors, output_path)
    print(f"合并完成！保存至：{output_path}")

def merge_safetensors_shards(base_model_name, shards_dir, output_dir):
    # 检查是否存在索引文件
    index_file = os.path.join(shards_dir, "model.safetensors.index.json")
    if os.path.exists(index_file):
        print(f"发现索引文件，将使用索引文件进行合并")
        return merge_from_index_json(shards_dir, output_dir)
    
    # 匹配分片文件的正则表达式（支持01-of-06和1-of-6两种格式）
    pattern = re.compile(rf"{re.escape(base_model_name)}-?(\d+)-of-(\d+)\.safetensors", re.IGNORECASE)
    
    # 收集所有分片文件
    shard_files = []
    total_shards = 0
    for filename in os.listdir(shards_dir):
        match = pattern.match(filename)
        if match:
            shard_num = int(match.group(1))
            total_shards = int(match.group(2))
            shard_files.append((shard_num, os.path.join(shards_dir, filename)))
    
    # 验证分片数量
    if not shard_files:
        raise ValueError(f"在目录 {shards_dir} 中未找到 {base_model_name} 的分片文件")
    
    found_shards = len(shard_files)
    if found_shards != total_shards:
        raise ValueError(f"分片数量不匹配，应有 {total_shards} 个分片，实际找到 {found_shards} 个")
    
    # 按分片编号排序
    shard_files.sort(key=lambda x: x[0])
    
    # 合并所有张量
    merged_tensors = {}
    for shard_num, filepath in tqdm(shard_files, desc="合并分片"):
        with safe_open(filepath, framework="pt") as f:
            for key in f.keys():
                if key in merged_tensors:
                    raise ValueError(f"发现重复的张量键: {key}，请检查分片文件")
                merged_tensors[key] = f.get_tensor(key)
    
    # 保存合并后的文件
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{base_model_name}.safetensors")
    save_file(merged_tensors, output_path)
    print(f"合并完成！保存至：{output_path}")

def auto_detect_model_format(model_dir):
    # 检查是否存在索引文件
    if os.path.exists(os.path.join(model_dir, "model.safetensors.index.json")):
        return "index", None
    
    # 检查是否存在分片文件
    shard_pattern = re.compile(r"(.*?)-?(\d+)-of-(\d+)\.safetensors")
    base_names = set()
    
    for filename in os.listdir(model_dir):
        match = shard_pattern.match(filename)
        if match:
            if match.group(1).startswith("model"):
                base_names.add("model")
            else:
                base_names.add(match.group(1))
    
    if base_names:
        return "shards", list(base_names)[0] if len(base_names) == 1 else None
    
    return "unknown", None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合并Safetensors模型文件")
    parser.add_argument("--model_dir", type=str, required=True,
                      help="包含模型文件的目录")
    parser.add_argument("--output_dir", type=str, default="models/merged",
                      help="合并文件输出目录")
    parser.add_argument("--base_name", type=str, default=None,
                      help="模型基础名称，例如：phi-4（如果不提供，将尝试自动检测）")
    parser.add_argument("--auto_detect", action="store_true",
                      help="自动检测模型格式并合并")
    
    args = parser.parse_args()
    
    try:
        if args.auto_detect:
            format_type, detected_base_name = auto_detect_model_format(args.model_dir)
            
            if format_type == "index":
                print("检测到索引文件格式，使用索引文件合并")
                merge_from_index_json(args.model_dir, args.output_dir)
            elif format_type == "shards":
                base_name = args.base_name or detected_base_name
                if not base_name:
                    raise ValueError("无法自动检测基础名称，请使用--base_name参数指定")
                print(f"检测到分片文件格式，基础名称: {base_name}")
                merge_safetensors_shards(base_name, args.model_dir, args.output_dir)
            else:
                raise ValueError("无法检测模型格式，请检查文件或手动指定参数")
        else:
            # 如果提供了base_name，使用传统方法
            if args.base_name:
                merge_safetensors_shards(args.base_name, args.model_dir, args.output_dir)
            else:
                # 否则尝试使用索引文件
                merge_from_index_json(args.model_dir, args.output_dir)
    except Exception as e:
        print(f"合并失败: {str(e)}")