import torch
import numpy as np
from safetensors import safe_open
from scipy import stats
from tqdm import tqdm

class ModelGradientMonitor:
    def __init__(self, model_path, device="cuda:1"):
        self.device = device
        self.model_params = self.load_model(model_path)
        self.input_shape = self.detect_input_shape()

        path_parts = model_path.split("/")
        filename = path_parts[-1]
        # 情况1: 直接是模型文件 (如 llama-3.2-1b.safetensors)
        if filename.endswith(".safetensors") and not filename.startswith("model"):
            self.model_name = filename.replace(".safetensors", "")
        # 情况2: 分片模型文件 (如 model-00001-of-000002.safetensors)
        elif filename.startswith("model-") and "of-" in filename:
            self.model_name = path_parts[-2]  # 使用上一级目录名
        # 情况3: 简单的model.safetensors
        elif filename == "model.safetensors":
            self.model_name = path_parts[-2]  # 使用上一级目录名
        # 情况4: 带有model后缀的文件 (如 unsloth-Qwen2.5-3B-Instruct-bnb-4bit-model.safetensors)
        elif filename.endswith("-model.safetensors"):
            self.model_name = filename.replace("-model.safetensors", "")
        # 情况5: 分片模型但没有model前缀 (如 phi-4-01-of-06.safetensors)
        elif "-of-" in filename and filename.endswith(".safetensors"):
            # 提取基础名称 (去掉-XX-of-XX部分)
            base_name = "-".join(filename.split("-")[:-3])
            self.model_name = base_name
        # 默认情况
        else:
            self.model_name = path_parts[-2] + "_" + filename.replace(".safetensors", "")

    def load_model(self, path):
        """加载模型参数并启用梯度"""
        params = {}
        with safe_open(path, framework="pt", device=self.device) as f:
            for key in f.keys():
                if "layers." in key:
                    layer_num = int(key.split("layers.")[1].split(".")[0])
                    if layer_num > 15:
                        continue
                
                tensor = f.get_tensor(key)
                if tensor.dtype != torch.float32:
                    tensor = tensor.to(torch.float32)
                tensor.requires_grad = True  # 启用梯度
                params[key] = tensor
        return params

    def detect_input_shape(self):
        """自动检测输入维度"""
        keywords = ["embed_tokens", "transformer"]
        for name in self.model_params:
            if any(keyword in name for keyword in keywords):
                return (1, self.model_params[name].shape[1])  # (batch_size, hidden_size)
        
        # 如果找不到标准名称，取第一个权重矩阵的输入维度
        first_weight = next(iter(self.model_params.values()))
        return (1, first_weight.shape[1])

    def generate_perturbation(self, target_layer, noise_type="adversarial"):
        """生成扰动信号（使用自动检测的输入维度）"""
        torch.manual_seed(0)
        input_dim = self.model_params[target_layer].shape[1]
        x = torch.randn((1, input_dim), device=self.device)
        
        if noise_type == "adversarial":
            return self.fgsm_perturbation(x)
        elif noise_type == "structured":
            return self.structured_noise(x)
        elif noise_type == "low_frequency":
            return self.low_frequency_noise(x)
        elif noise_type == "high_frequency":
            return self.high_frequency_noise(x)
        else:
            return x + torch.randn_like(x) * 0.1

    def fgsm_perturbation(self, x, epsilon=0.03):
        """生成对抗性扰动"""
        x_leaf = x.clone().detach().requires_grad_(True)
        
        weight = self.find_appropriate_weight(x_leaf.shape[-1])
        
        outputs = x_leaf @ weight.to(torch.float32).T
        pseudo_target = torch.randint(0, outputs.shape[1], (x_leaf.shape[0],), device=self.device)
        loss = torch.nn.functional.cross_entropy(outputs, pseudo_target)
        loss.backward()
        
        perturbed_x = x_leaf + epsilon * x_leaf.grad.sign()
        return perturbed_x.detach().requires_grad_(True)

    def find_appropriate_weight(self, input_dim):
        """寻找与输入维度匹配的权重矩阵"""
        for name, param in self.model_params.items():
            if param.ndim == 2 and param.shape[1] == input_dim:
                param.requires_grad = True
                return param
        
        available_dims = {name: p.shape for name, p in self.model_params.items() if p.ndim == 2}
        raise ValueError(
            f"找不到与输入维度 {input_dim} 匹配的权重矩阵。可用二维参数：\n"
            f"{available_dims}"
        )

    def structured_noise(self, x):
        """生成结构化噪声"""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        freq = torch.fft.rfft(x)
        mask = torch.zeros_like(freq)
        mask[..., :freq.shape[-1]//4] = 1 #保留频率前四分之一的部分
        return x + torch.fft.irfft(freq * mask, n=x.size(-1)).real

    def low_frequency_noise(self, x):
        """生成低频噪声"""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        freq = torch.fft.rfft(x)
        mask = torch.zeros_like(freq)
        mask[..., :freq.shape[-1]//8] = 1  # 只保留最低频率部分
        noise = torch.fft.irfft(freq * mask, n=x.size(-1)).real
        return x + 0.1 * noise

    def high_frequency_noise(self, x):
        """生成高频噪声"""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        freq = torch.fft.rfft(x)
        mask = torch.zeros_like(freq)
        mask[..., freq.shape[-1]//2:] = 1  # 只保留高频部分
        noise = torch.fft.irfft(freq * mask, n=x.size(-1)).real
        return x + 0.1 * noise

    def extract_gradients(self, x, layer_name):
        """提取梯度并验证非空"""
        try:
            # 清除旧梯度
            for param in self.model_params.values():
                if param.grad is not None:
                    param.grad = None

            weight = self.model_params[layer_name]
            outputs = x @ weight.to(torch.float32).T
            loss = outputs.norm()
            loss.backward(retain_graph=True)

            # 仅返回目标层梯度
            if weight.grad is None:
                return None
                
            # 转换梯度为numpy进行特征提取
            grad_np = weight.grad.detach().cpu().numpy()
            
            # 提取丰富的特征
            return self.extract_rich_features(grad_np, layer_name)
            
        except Exception as e:
            print(f"提取梯度时发生错误：{str(e)}")
            return {}  # 返回空字典避免后续索引错误

    def extract_rich_features(self, grad_np, layer_name):
        """提取梯度的丰富特征集合"""
        features = {}
        # 1. 基础统计特征
        features["mean"] = float(np.mean(grad_np))
        features["std"] = float(np.std(grad_np))
        features["norm"] = float(np.linalg.norm(grad_np))
        
        # # 2. 分布特征
        if grad_np.size > 1000000:
            np.random.seed(0)  # 固定随机种子以确保结果可重现
            sample_size = min(500000, grad_np.size)  # 最多采样50万个点
            flat_indices = np.random.choice(grad_np.size, sample_size, replace=False)
            sampled_data = grad_np.flatten()[flat_indices]
            features["skewness"] = float(stats.skew(sampled_data))
            features["kurtosis"] = float(stats.kurtosis(sampled_data))
        else:
            features["skewness"] = float(stats.skew(grad_np.flatten()))
            features["kurtosis"] = float(stats.kurtosis(grad_np.flatten()))
        
        # # 4. 频域特征 (对一维展平梯度进行FFT)
        # flattened = grad_np.flatten()
        # print(flattened.shape)
        # if len(flattened) > 1:  # 确保有足够的数据进行FFT
        #     fft_vals = np.abs(np.fft.rfft(flattened))
        #     features["fft_mean"] = float(np.mean(fft_vals))
        #     features["fft_std"] = float(np.std(fft_vals))
        #     features["fft_max"] = float(np.max(fft_vals))
            
        #     # 频域能量分布 (低、中、高频)
        #     third = len(fft_vals) // 3
        #     features["low_freq_energy"] = float(np.sum(fft_vals[:third]) / np.sum(fft_vals))
        #     features["mid_freq_energy"] = float(np.sum(fft_vals[third:2*third]) / np.sum(fft_vals))
        #     features["high_freq_energy"] = float(np.sum(fft_vals[2*third:]) / np.sum(fft_vals))
        
        # # 5. 矩阵特征 (针对2D梯度)
        # if grad_np.ndim == 2:
        #     # SVD分解特征
        #     try:
        #         u, s, vh = np.linalg.svd(grad_np, full_matrices=False)
        #         features["singular_values_top3"] = [float(s[i]) if i < len(s) else 0.0 for i in range(3)]
        #         features["singular_values_ratio"] = float(s[0] / (np.sum(s) + 1e-10))  # 第一奇异值占比
                
        #         # 矩阵条件数 (如果可计算)
        #         if len(s) > 1 and s[-1] > 1e-10:
        #             features["condition_number"] = float(s[0] / s[-1])
        #     except Exception as e:
        #         print(f"计算SVD时出错: {e}")
        
        # # 6. 行/列统计特征
        # if grad_np.ndim == 2:
        #     row_means = np.mean(grad_np, axis=1)
        #     col_means = np.mean(grad_np, axis=0)
        #     features["row_mean_std"] = float(np.std(row_means))
        #     features["col_mean_std"] = float(np.std(col_means))
            
        # 7. 层表示特征
        features["layer_name"] = layer_name
        features["layer_size"] = grad_np.size
        features["layer_shape"] = list(grad_np.shape)
        
        # 8. 结构化标记 (如果层名包含特定关键词)
        layer_type = "unknown"
        if "attention" in layer_name.lower() or "attn" in layer_name.lower():
            layer_type = "attention"
        elif "ffn" in layer_name.lower() or "mlp" in layer_name.lower():
            layer_type = "ffn"
        elif "embed" in layer_name.lower():
            layer_type = "embedding"
        elif "norm" in layer_name.lower():
            layer_type = "norm"
        features["layer_type"] = layer_type
        
        return features

    def visualize_gradients(self, features, layer_name):
        """打印修正后的梯度统计信息"""
        print("=== 梯度响应统计 ===")
        
        # 从特征中筛选基础统计量进行显示
        basic_stats = {}
        for feature in features:
            for key in ["mean", "std", "norm", "skewness", "kurtosis"]:
                if key in feature:
                    if key not in basic_stats:
                        basic_stats[key] = []
                    basic_stats[key].append(feature[key])
        
        # 打印基础统计量
        print(f"示例层 '{layer_name}' 统计:")
        for stat_name, values in basic_stats.items():
            avg = np.mean(values)
            std = np.std(values)
            print(f"{stat_name}: {avg:.3f} ± {std:.3f}")
        
        # # 频域特征
        # freq_stats = {}
        # for feature in features:
        #     for key in ["low_freq_energy", "mid_freq_energy", "high_freq_energy"]:
        #         if key in feature:
        #             if key not in freq_stats:
        #                 freq_stats[key] = []
        #             freq_stats[key].append(feature[key])
        
        # if freq_stats:
        #     print("\n频域特征:")
        #     for stat_name, values in freq_stats.items():
        #         avg = np.mean(values)
        #         print(f"{stat_name}: {avg:.3f}")

    def analyze_gradients(self, num_samples=100):
        """梯度响应分析"""
        noise_types = ["adversarial", "structured", "gaussian", "low_frequency", "high_frequency"]
        random_state = np.random.RandomState(0)
        
        # 收集所有层的特征
        all_layer_features = {}
        
        # 仅遍历二维权重层
        target_layers = [name for name, param in self.model_params.items() if param.ndim == 2]
        
        # 限制分析的层数以减少计算负担
        if len(target_layers) > 10:
            # 选择代表性层 (如嵌入层、注意力层、FFN层等)
            selected_layers = []
            layer_types = {"embed": [], "attention": [], "ffn": [], "others": []}
            
            for layer in target_layers:
                if "embed" in layer.lower():
                    layer_types["embed"].append(layer)
                elif "attention" in layer.lower() or "attn" in layer.lower():
                    layer_types["attention"].append(layer)
                elif "ffn" in layer.lower() or "mlp" in layer.lower():
                    layer_types["ffn"].append(layer)
                else:
                    layer_types["others"].append(layer)
            
            # 从每种类型中选择代表层
            for layer_type, layers in layer_types.items():
                if layers:
                    # 均匀采样
                    indices = np.linspace(0, len(layers)-1, min(3, len(layers)), dtype=int)
                    selected_layers.extend([layers[i] for i in indices])
            
            # 如果选择的层太少，添加一些随机层
            if len(selected_layers) < 10:
                remaining = list(set(target_layers) - set(selected_layers))
                if remaining:
                    random_layers = random_state.choice(remaining, min(10-len(selected_layers), len(remaining)), replace=False)
                    selected_layers.extend(random_layers)
            
            target_layers = selected_layers
        
        # 分析选定的层
        for layer_name in target_layers:
            print(f"\n=== 分析层: {layer_name} ===")
            gradient_features = []
            
            # 获取当前层的权重矩阵
            weight = self.model_params[layer_name]
            original_requires_grad = weight.requires_grad
            weight.requires_grad = True  # 确保启用梯度
            
            for _ in tqdm(range(num_samples), desc=f"采样 {layer_name}"):
                # 生成针对当前层的扰动
                noise_type = random_state.choice(noise_types)
                x = self.generate_perturbation(target_layer=layer_name, noise_type=noise_type)
                
                # 提取当前层的梯度
                grad_info = self.extract_gradients(x, layer_name)
                if grad_info:
                    gradient_features.append(grad_info)
            
            # 恢复原始梯度设置
            weight.requires_grad = original_requires_grad
            
            # 统计与可视化
            if gradient_features:
                self.visualize_gradients(gradient_features, layer_name)
                all_layer_features[layer_name] = gradient_features
        
        return all_layer_features

    def extract_model_fingerprint(self, all_layer_features):
        """从所有层的特征中提取模型指纹"""
        model_features = {}
        
        # 1. 全局特征聚合
        all_means = []
        all_stds = []
        all_norms = []
        all_skewness = []
        all_kurtosis = []
        
        # 提取不同层类型的特征
        layer_types = {
            "attention": {"means": [], "stds": [], "norms": []},
            "ffn": {"means": [], "stds": [], "norms": []},
            "embedding": {"means": [], "stds": [], "norms": []},
            "norm": {"means": [], "stds": [], "norms": []}
        }
        
        # 收集所有层的特征统计
        for layer_name, features in all_layer_features.items():
            layer_means = [f.get("mean", 0) for f in features if "mean" in f]
            layer_stds = [f.get("std", 0) for f in features if "std" in f]
            layer_norms = [f.get("norm", 0) for f in features if "norm" in f]
            layer_skew = [f.get("skewness", 0) for f in features if "skewness" in f]
            layer_kurt = [f.get("kurtosis", 0) for f in features if "kurtosis" in f]
            
            if layer_means and layer_stds and layer_norms:
                all_means.append(np.mean(layer_means))
                all_stds.append(np.mean(layer_stds))
                all_norms.append(np.mean(layer_norms))
                
                if layer_skew:
                    all_skewness.append(np.mean(layer_skew))
                if layer_kurt:
                    all_kurtosis.append(np.mean(layer_kurt))
                
                # 分类层特征
                layer_type = "unknown"
                for feature in features:
                    if "layer_type" in feature:
                        layer_type = feature["layer_type"]
                        break
                
                if layer_type in layer_types:
                    layer_types[layer_type]["means"].append(np.mean(layer_means))
                    layer_types[layer_type]["stds"].append(np.mean(layer_stds))
                    layer_types[layer_type]["norms"].append(np.mean(layer_norms))
        
        # 全局统计特征
        model_features["global_mean"] = np.mean(all_means) if all_means else 0
        model_features["global_std"] = np.mean(all_stds) if all_stds else 0
        model_features["global_norm"] = np.mean(all_norms) if all_norms else 0
        model_features["global_skewness"] = np.mean(all_skewness) if all_skewness else 0
        model_features["global_kurtosis"] = np.mean(all_kurtosis) if all_kurtosis else 0
        
        # 各层类型统计特征
        for layer_type, stats in layer_types.items():
            if stats["means"]:
                model_features[f"{layer_type}_mean"] = np.mean(stats["means"])
                model_features[f"{layer_type}_std"] = np.mean(stats["stds"])
                model_features[f"{layer_type}_norm"] = np.mean(stats["norms"])
        
        # 频域特征聚合
        # freq_features = ["low_freq_energy", "mid_freq_energy", "high_freq_energy"]
        # for freq_feat in freq_features:
        #     values = []
        #     for _, features in all_layer_features.items():
        #         for feature in features:
        #             if freq_feat in feature:
        #                 values.append(feature[freq_feat])
        #     if values:
        #         model_features[f"global_{freq_feat}"] = np.mean(values)
        
        # 模型架构特征
        total_params = 0
        for param in self.model_params.values():
            total_params += param.numel()
        model_features["total_params"] = total_params
        model_features["num_layers"] = len(all_layer_features)
        model_features["model_name"] = self.model_name
        
        return model_features

    def save_model_features(self, model_features, filename):
        """保存模型特征到文件"""
        import json
        with open(filename, 'w') as f:
            json.dump(model_features, f, indent=2)
        print(f"模型特征已保存到 {filename}")

def debug_model_analysis(model_paths, num_samples=30):
    import gc

    for model_path in model_paths:
        print(f"\n====== 分析模型: {model_path} ======")

        # 清理内存
        gc.collect()
        torch.cuda.empty_cache()

        monitor = ModelGradientMonitor(model_path=model_path)
        
        # 执行梯度分析
        layer_features = monitor.analyze_gradients(num_samples=num_samples)
        
        # 提取模型指纹
        model_features = monitor.extract_model_fingerprint(layer_features)
        
        # 保存模型特征
        monitor.save_model_features(model_features, f"features/{monitor.model_name}_features.json")

if __name__ == "__main__":
    model_paths = [
        "path/to/model.safetensors"
    ]
    
    debug_model_analysis(model_paths, num_samples=15)

    # pass