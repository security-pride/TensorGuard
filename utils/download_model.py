from huggingface_hub import HfApi, snapshot_download
import os
import argparse
from tqdm import tqdm
from merge_model import auto_detect_model_format

# 修复模型列表定义，使用正确的列表语法
models = [
    # "fdtn-ai/Foundation-Sec-8B",
    # "osllmai-community/Llama-3.2-1B",
    # "meta-llama/Llama-3.2-1B-Instruct-QLORA_INT4_EO8",
    # "BuandLa/ETLCH_base_on_llama3.2-1b_taiwan",
    # "facebook/layerskip-llama3.2-1B",
    # "meta-llama/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8",
    # "cognitivecomputations/Dolphin3.0-Llama3.2-1B",
    # "DuoGuard/DuoGuard-1B-Llama-3.2-transfer",
    # "axel-datos/Llama-3.2-1B_gsm8k_full-finetuning",
    # "SEOKDONG/llama3.2_1B_korean_v0.2_sft_by_aidx",
    # "meta-llama/Llama-3.2-3B-Instruct-SpinQuant_INT4_EO8",
    # "lianghsun/Llama-3.2-Taiwan-3B",
    # "CriteriaPO/llama3.2-3b-sft-10",
    # "twinkle-ai/Llama-3.2-3B-F1-Instruct"
    # "google/gemma-3-4b-it-qat-q4_0-unquantized",
    # "mlabonne/gemma-3-4b-it-abliterated",
    # "dnotitia/Smoothie-Qwen2.5-3B-Instruct",
    # "authormist/authormist-originality",
    # "Commencis/Commencis-LLM",
    # "MaziyarPanahi/Mistral-7B-Instruct-v0.1-GPTQ",
    # "microsoft/Phi-4-reasoning-plus",
    # "microsoft/Phi-4-reasoning",
    # "ykarout/phi-4-deepseek-r1-distilled-v8"
    # "Gensyn/Qwen2.5-7B-Instruct",
    # "unsloth/Qwen2.5-7B-Instruct",
    # "context-labs/Meta-Llama-3.1-8B-Instruct-FP16",
    # "meta-llama/Llama-3.1-8B-Instruct"
    # "Qwen/Qwen2.5-3B"
    "MaziyarPanahi/Mistral-7B-Instruct-v0.1-GPTQ"
]

# Hugging Face API token
token = '''huggingface_api_token'''

def download_model(model_id, output_dir, token=None, revision=None):
    try:
        print(f"正在下载模型: {model_id}")
        model_id_path = model_id.replace('/', '_')
        local_dir = snapshot_download(
            repo_id=model_id,
            local_dir=os.path.join(output_dir, model_id_path),
            token=token,
            revision=revision
        )
        print(f"模型 {model_id} 下载完成，保存在: {local_dir}")
        
        # 检测模型格式
        format_type, base_name = auto_detect_model_format(local_dir)
        print(f"模型格式: {format_type}, 基础名称: {base_name}")
        
        return local_dir, format_type, base_name
    except Exception as e:
        print(f"下载模型 {model_id} 时出错: {str(e)}")
        return None, None, None

def download_all_models(models, output_dir, token=None):
    results = []
    for model_id in tqdm(models, desc="下载模型"):
        os.makedirs(output_dir, exist_ok=True)
        local_dir, format_type, base_name = download_model(model_id, output_dir, token)
        if local_dir:
            results.append({
                "model_id": model_id,
                "local_dir": local_dir,
                "format_type": format_type,
                "base_name": base_name
            })
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从Hugging Face下载模型")
    parser.add_argument("--output_dir", type=str, default="models",
                      help="下载模型的输出目录")
    parser.add_argument("--token", type=str, default=token,
                      help="Hugging Face API token")
    parser.add_argument("--model_ids", nargs="+",
                      help="要下载的特定模型ID (如果不提供，将下载预定义列表中的所有模型)")
    
    args = parser.parse_args()
    
    # 确定要下载的模型列表
    models_to_download = args.model_ids if args.model_ids else models
    
    # 下载所有模型
    print(f"开始下载 {len(models_to_download)} 个模型...")
    download_results = download_all_models(models_to_download, args.output_dir, args.token)
    
    print("\n下载结果摘要:")
    print(f"成功下载: {len(download_results)} 个模型")
    for result in download_results:
        print(f"- {result['model_id']}, 格式: {result['format_type']}")