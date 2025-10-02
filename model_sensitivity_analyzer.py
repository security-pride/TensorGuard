import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from model_gradient_monitor import ModelGradientMonitor

class ModelSensitivityAnalyzer:
    def __init__(self, model_path, device="cuda:0"):
        """初始化敏感度分析器"""
        self.model_path = model_path
        self.device = device
        self.monitor = ModelGradientMonitor(model_path=model_path, device=device)
        self.model_name = self.monitor.model_name
    
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
        for name, param in self.monitor.model_params.items():
            if param.ndim == 2 and param.shape[1] == input_dim:
                param.requires_grad = True
                return param
        
        available_dims = {name: p.shape for name, p in self.monitor.model_params.items() if p.ndim == 2}
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

    # 添加针对注意力层的扰动生成方法
    def generate_attention_perturbation(self, target_layer, perturbation_type="token_focus"):
        """
        生成针对注意力层的特殊扰动
        
        参数:
            target_layer: 目标层名称
            perturbation_type: 扰动类型，可选值:
                - "token_focus": 模拟token间强关注模式
                - "attention_pattern": 模拟特定注意力模式
                - "head_specific": 针对特定注意力头的扰动
        """
        torch.manual_seed(0)
        input_dim = self.monitor.model_params[target_layer].shape[1]
        
        if perturbation_type == "token_focus":
            # 创建模拟token间强关注的输入
            # 生成一个基础向量
            x_base = torch.randn((1, input_dim), device=self.device)
            
            # 创建一个掩码，随机选择一部分维度进行强化
            mask = torch.zeros_like(x_base)
            focus_indices = torch.randperm(input_dim)[:input_dim//4]  # 随机选择25%的维度
            mask[:, focus_indices] = 3.0  # 这些维度的值放大3倍
            
            # 应用掩码生成扰动
            x_perturbed = x_base * (1 + mask)
            
        elif perturbation_type == "attention_pattern":
            # 模拟特定的注意力模式
            x = torch.zeros((1, input_dim), device=self.device)
            
            # 创建周期性模式，模拟注意力的波动
            positions = torch.arange(0, input_dim, device=self.device).float()
            pattern = torch.sin(positions * (6.28 / (input_dim / 8)))  # 创建8个周期的正弦波
            
            # 应用模式
            x[0, :] = pattern * 2.0  # 放大模式
            x_perturbed = x + torch.randn_like(x) * 0.2  # 添加少量噪声
            
        elif perturbation_type == "head_specific":
            # 针对多头注意力的特定头部
            # 假设输入维度可以被注意力头数量整除
            num_heads = 8  # 假设有8个注意力头
            head_dim = input_dim // num_heads
            
            # 创建基础输入
            x = torch.randn((1, input_dim), device=self.device) * 0.1
            
            # 随机选择1-3个头进行强化
            num_heads_to_enhance = torch.randint(1, 4, (1,)).item()
            heads_to_enhance = torch.randperm(num_heads)[:num_heads_to_enhance]
            
            # 对选定的头部区域进行强化
            for head_idx in heads_to_enhance:
                start_idx = head_idx * head_dim
                end_idx = (head_idx + 1) * head_dim
                x[0, start_idx:end_idx] = torch.randn(head_dim, device=self.device) * 2.0
                
            x_perturbed = x
        elif perturbation_type == "adversarial":
            x = torch.randn((1, input_dim), device=self.device)
            x_perturbed = self.fgsm_perturbation(x)
        elif perturbation_type == "structured":
            # 生成结构化噪声
            x = torch.randn((1, input_dim), device=self.device)
            x_perturbed = self.structured_noise(x)
        elif perturbation_type == "gaussian":
            # 生成高斯噪声
            x = torch.randn((1, input_dim), device=self.device)
            x_perturbed = x + torch.randn_like(x) * 0.1
        elif perturbation_type == "low_frequency":
            # 生成低频噪声
            x = torch.randn((1, input_dim), device=self.device)
            x_perturbed = self.low_frequency_noise(x)
        elif perturbation_type == "high_frequency":
            # 生成高频噪声
            x = torch.randn((1, input_dim), device=self.device)
            x_perturbed = self.high_frequency_noise(x)
        else:
            # 默认返回随机扰动
            x_perturbed = torch.randn((1, input_dim), device=self.device)
            
        return x_perturbed.requires_grad_(True)

    def identify_sensitive_layers(self, num_samples=30, threshold_factor=1.5):
        """
        通过扰动响应分析识别模型中的敏感层和非敏感层
        
        参数:
            num_samples: 每层采样次数
            threshold_factor: 判定敏感层的阈值因子（相对于平均响应强度）
        
        返回:
            sensitive_layers: 敏感层列表
            non_sensitive_layers: 非敏感层列表
            sensitivity_scores: 各层的敏感度得分
        """
        # 仅分析二维权重层
        target_layers = [name for name, param in self.monitor.model_params.items() if param.ndim == 2]
        
        # 收集各层的梯度响应强度
        layer_responses = {}
        noise_types = ["adversarial", "structured", "gaussian", "low_frequency", "high_frequency"]
        random_state = np.random.RandomState(0)
        
        print("开始分析各层敏感度...")
        for layer_name in tqdm(target_layers):
            responses = []
            
            # 获取当前层的权重矩阵
            weight = self.monitor.model_params[layer_name]
            original_requires_grad = weight.requires_grad
            weight.requires_grad = True
            
            for _ in range(num_samples):
                # 生成不同类型的扰动
                noise_type = random_state.choice(noise_types)
                x = self.monitor.generate_perturbation(target_layer=layer_name, noise_type=noise_type)
                
                # 清除旧梯度
                for param in self.monitor.model_params.values():
                    if param.grad is not None:
                        param.grad = None
                        
                # 计算梯度
                with torch.amp.autocast('cuda', enabled=True):
                    outputs = x @ weight.to(torch.float32).T
                    loss = outputs.norm()
                loss.backward(retain_graph=False)
                
                if weight.grad is not None:
                    # 计算梯度响应强度（使用Frobenius范数）
                    grad_norm = torch.norm(weight.grad).item()
                    responses.append(grad_norm)
                    weight.grad = None
            
            # 恢复原始梯度设置
            weight.requires_grad = original_requires_grad
            
            if responses:
                # 使用平均响应强度作为敏感度指标
                layer_responses[layer_name] = np.mean(responses)
            
            torch.cuda.empty_cache()
        
        # 计算平均响应强度
        mean_response = np.mean(list(layer_responses.values()))
        
        # 根据阈值区分敏感层和非敏感层
        sensitive_layers = []
        non_sensitive_layers = []
        sensitivity_scores = {}
        
        for layer_name, response in layer_responses.items():
            # 计算相对敏感度得分
            sensitivity = response / mean_response
            sensitivity_scores[layer_name] = sensitivity
            
            if sensitivity > threshold_factor:
                sensitive_layers.append(layer_name)
            else:
                non_sensitive_layers.append(layer_name)
        
        # 按敏感度排序
        sensitive_layers.sort(key=lambda x: sensitivity_scores[x], reverse=True)
        non_sensitive_layers.sort(key=lambda x: sensitivity_scores[x], reverse=True)
        
        print(f"\n发现 {len(sensitive_layers)} 个敏感层，{len(non_sensitive_layers)} 个非敏感层")
        print("\n前5个最敏感的层:")
        for i, layer in enumerate(sensitive_layers[:5]):
            print(f"{i+1}. {layer}: 敏感度 = {sensitivity_scores[layer]:.3f}")
        
        return sensitive_layers, non_sensitive_layers, sensitivity_scores
    
    def visualize_sensitivity(self, sensitivity_scores, top_n=20, save_path=None):
        """可视化层敏感度分布"""
        # 按敏感度排序
        sorted_layers = sorted(sensitivity_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 取前N个最敏感的层
        top_layers = sorted_layers[:top_n]
        layer_names = [item[0] for item in top_layers]
        scores = [item[1] for item in top_layers]
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(layer_names)), scores, color='skyblue')
        plt.yticks(range(len(layer_names)), layer_names)
        plt.xlabel('Sensitivity Score')
        plt.title(f'{self.model_name} Sensitivity Analysis (top {top_n} layers)')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # 为每个条形添加数值标签
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, 
                    f'{scores[i]:.2f}', va='center', fontsize=30)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"敏感度可视化已保存到 {save_path}")
        
        plt.show()
    
    def save_sensitivity_results(self, sensitive_layers, non_sensitive_layers, sensitivity_scores, output_dir="features"):
        """保存敏感度分析结果"""
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 准备结果数据
        results = {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "sensitive_layers": sensitive_layers,
            "non_sensitive_layers": non_sensitive_layers,
            "sensitivity_scores": {k: float(v) for k, v in sensitivity_scores.items()}
        }
        
        # 保存为JSON文件
        output_file = os.path.join(output_dir, f"{self.model_name}_sensitivity.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"敏感度分析结果已保存到 {output_file}")
        
        # 生成可视化
        vis_path = os.path.join(output_dir, f"{self.model_name}_sensitivity_vis.png")
        self.visualize_sensitivity(sensitivity_scores, save_path=vis_path)
        
        return output_file
    
    def test_attention_perturbation_sensitivity(self, num_samples=20, threshold_factor=1.5):
        """
        使用注意力扰动方案测试所有层的敏感度，验证该方案是否对注意力层更敏感
        
        参数:
            num_samples: 每层采样次数
            threshold_factor: 判定敏感层的阈值因子（相对于平均响应强度）
        
        返回:
            attn_sensitive_layers: 对注意力扰动敏感的层列表
            attn_sensitivity_scores: 各层对注意力扰动的敏感度得分
            is_attn_layer: 标记每个层是否为注意力相关层的字典
        """
        # 仅分析二维权重层
        target_layers = [name for name, param in self.monitor.model_params.items() if param.ndim == 2]
        
        # 收集各层的梯度响应强度
        layer_responses = {}
        attn_perturbation_types = ["token_focus", "attention_pattern", "head_specific", "adversarial", "structured", "gaussian", "low_frequency", "high_frequency"]
        random_state = np.random.RandomState(0)
        
        # 标记注意力相关层
        is_attn_layer = {}
        for layer_name in target_layers:
            is_attn_layer[layer_name] = any(kw in layer_name.lower() for kw in 
                                           ["attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj"])
        
        print("开始使用注意力扰动方案分析各层敏感度...")
        for layer_name in tqdm(target_layers):
            responses = []
            
            # 获取当前层的权重矩阵
            weight = self.monitor.model_params[layer_name]
            original_requires_grad = weight.requires_grad
            weight.requires_grad = True
            
            for _ in range(num_samples):
                # 随机选择一种注意力扰动类型
                # perturbation_type = random_state.choice(attn_perturbation_types)
                perturbation_type = "high_frequency" # ["adversarial", "structured", "gaussian", "low_frequency", "high_frequency"]
                
                # 生成注意力扰动
                x = self.generate_attention_perturbation(target_layer=layer_name, perturbation_type=perturbation_type)
                
                # 清除旧梯度
                for param in self.monitor.model_params.values():
                    if param.grad is not None:
                        param.grad = None
                        
                # 计算梯度
                with torch.amp.autocast('cuda', enabled=True):
                    outputs = x @ weight.to(torch.float32).T
                    loss = outputs.norm()
                loss.backward(retain_graph=False)
                
                if weight.grad is not None:
                    # 计算梯度响应强度（使用Frobenius范数）
                    grad_norm = torch.norm(weight.grad).item()
                    responses.append(grad_norm)
                    weight.grad = None
            
            # 恢复原始梯度设置
            weight.requires_grad = original_requires_grad
            
            if responses:
                # 使用平均响应强度作为敏感度指标
                layer_responses[layer_name] = np.mean(responses)
            
            torch.cuda.empty_cache()
        
        # 计算平均响应强度
        mean_response = np.mean(list(layer_responses.values()))
        
        # 根据阈值区分敏感层和非敏感层
        attn_sensitive_layers = []
        attn_sensitivity_scores = {}
        
        for layer_name, response in layer_responses.items():
            # 计算相对敏感度得分
            sensitivity = response / mean_response
            attn_sensitivity_scores[layer_name] = sensitivity
            
            if sensitivity > threshold_factor:
                attn_sensitive_layers.append(layer_name)
        
        # 按敏感度排序
        attn_sensitive_layers.sort(key=lambda x: attn_sensitivity_scores[x], reverse=True)
        
        # 分析注意力层和非注意力层的敏感度差异
        attn_layers = [layer for layer in target_layers if is_attn_layer[layer]]
        non_attn_layers = [layer for layer in target_layers if not is_attn_layer[layer]]
        
        attn_layer_scores = [attn_sensitivity_scores[layer] for layer in attn_layers]
        non_attn_layer_scores = [attn_sensitivity_scores[layer] for layer in non_attn_layers]
        
        avg_attn_sensitivity = np.mean(attn_layer_scores) if attn_layer_scores else 0
        avg_non_attn_sensitivity = np.mean(non_attn_layer_scores) if non_attn_layer_scores else 0
        
        print(f"\n发现 {len(attn_sensitive_layers)} 个对注意力扰动敏感的层")
        print(f"注意力相关层平均敏感度: {avg_attn_sensitivity:.3f}")
        print(f"非注意力层平均敏感度: {avg_non_attn_sensitivity:.3f}")
        print(f"敏感度比值 (注意力/非注意力): {avg_attn_sensitivity/avg_non_attn_sensitivity:.3f}")
        
        print("\n前5个对注意力扰动最敏感的层:")
        for i, layer in enumerate(attn_sensitive_layers[:5]):
            layer_type = "注意力层" if is_attn_layer[layer] else "非注意力层"
            print(f"{i+1}. {layer} ({layer_type}): 敏感度 = {attn_sensitivity_scores[layer]:.3f}")
        
        path_1 = "sensitivity/" + perturbation_type + ".pdf"
        if perturbation_type == "high_frequency":
            path_1 = "sensitivity/high_freq.pdf"
        if perturbation_type == "low_frequency":
            path_1 = "sensitivity/low_freq.pdf"
        # path_1 = "sensitivity/random.pdf"

        # 可视化注意力扰动敏感度分布
        self.visualize_attention_sensitivity(attn_sensitivity_scores, is_attn_layer, save_path=path_1)
        
        return attn_sensitive_layers, attn_sensitivity_scores, is_attn_layer
    
    def visualize_attention_sensitivity(self, sensitivity_scores, is_attn_layer, top_n=20, save_path=None):
        """可视化注意力扰动敏感度分布，区分注意力层和非注意力层"""
        # 按敏感度排序
        sorted_layers = sorted(sensitivity_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 取前N个最敏感的层
        top_layers = sorted_layers[:top_n]
        layer_names = []
        scores = []
        for name, score in top_layers:
            layer_names.append(name)
            scores.append(score)
        
        # 区分注意力层和非注意力层的颜色
        colors = ['#ff7f0e' if is_attn_layer[layer] else '#1f77b4' for layer in layer_names]
        
        plt.figure(figsize=(30, 25), dpi=300)
        bars = plt.barh(range(len(layer_names)), scores, color=colors)

        plt.yticks(range(len(layer_names)), [''] * len(layer_names))
        
        # 在柱状图内部添加层名称
        for i, bar in enumerate(bars):
            # 计算文本位置 - 放在柱状图内部靠左侧
            text_x = min(0.5, bar.get_width() * 0.1)  # 确保文本不会太靠右
            text_y = bar.get_y() + bar.get_height()/2
            
            # 缩短过长的层名称
            short_name = layer_names[i]
            if short_name.startswith('model.'):
                short_name = short_name[len('model.'):]  # 更快的替换方式
            if short_name.endswith('.weight'):
                short_name = short_name[:-len('.weight')]
            if len(short_name) > 50:  # 可以根据需要调整截断长度
                short_name = short_name[:22] + '...'
            
            # 添加层名称文本，白色文本以便在柱状图上清晰可见
            plt.text(0, text_y, short_name, 
                    va='center', ha='left', color='white', fontweight='bold', fontsize=50)
        plt.xlabel('Attention Perturbation Sensitivity Score', fontsize=55)
        # plt.title(f'{self.model_name} Sensitivity Analysis with High_frequency Noise(top {top_n} layers)')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tick_params(axis='x', labelsize=50)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#ff7f0e', label='Attention Layers'),
            Patch(facecolor='#1f77b4', label='None Attention Layers')
        ]
        plt.legend(handles=legend_elements, fontsize=50)
        
        # 为每个条形添加数值标签
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, 
                    f'{scores[i]:.2f}', va='center', fontsize=55)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"sensitivity/high_freq.pdf"
        
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path)
        print(f"注意力扰动敏感度可视化已保存到 {save_path}")
        
        plt.show()

def analyze_model_sensitivity(model_path, num_samples=30, threshold_factor=1.5, output_dir="features"):
    """分析单个模型的层敏感度"""
    print(f"\n====== 分析模型敏感层: {model_path} ======")
    analyzer = ModelSensitivityAnalyzer(model_path=model_path)
    
    # 执行敏感度分析
    sensitive, non_sensitive, scores = analyzer.identify_sensitive_layers(
        num_samples=num_samples, 
        threshold_factor=threshold_factor
    )
    
    # 保存结果
    output_file = analyzer.save_sensitivity_results(
        sensitive, non_sensitive, scores, output_dir=output_dir
    )
    
    return output_file

def visualize_sensitivity_charts(all_results):
    # 比较结果可视化
    if len(all_results) > 1:
        fig_height = max(15, len(all_results) * 5)
        plt.figure(figsize=(15, fig_height))
        
        # 为每个模型创建一个子图
        for i, result in enumerate(all_results):
            plt.subplot(len(all_results), 1, i+1)
            
            model_name = result["model_name"]
            top_layers = result["sensitive_layers"]
            scores = [result["sensitivity_scores"][layer] for layer in top_layers]
            
            # 简化层名称显示
            layer_names = [layer for layer in top_layers]
            
            plt.barh(range(len(layer_names)), scores, color=f'C{i}')
            plt.yticks(range(len(layer_names)), layer_names)
            plt.title(f'{model_name} Sensitivity Distribution')
            plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        compare_path = os.path.join("sensitivity", "model_sensitivity_comparison.png")
        plt.savefig(compare_path)
        print(f"模型敏感度比较已保存到 {compare_path}")
        plt.show()
    elif len(all_results) == 1:
        print("\n只有一个模型成功分析，无法生成比较图")
    else:
        print("\n没有模型成功分析，无法生成可视化结果")

def compare_model_sensitivities(model_paths, num_samples=20, output_dir="features"):
    """比较多个模型的敏感层分布"""
    all_results = []
    skipped_models = []
    
    for model_path in model_paths:
        print(f"\n====== 分析模型: {model_path} ======")
        try:
            # 尝试创建分析器并分析模型
            analyzer = ModelSensitivityAnalyzer(model_path=model_path)
            
            # 执行敏感度分析
            sensitive, non_sensitive, scores = analyzer.identify_sensitive_layers(num_samples=num_samples)
            
            # 保存结果
            output_file = analyzer.save_sensitivity_results(
                sensitive, non_sensitive, scores, output_dir=output_dir
            )
            
            all_results.append({
                "model_name": analyzer.model_name,
                "sensitive_layers": sensitive[:10],  # 只取前10个最敏感的层
                "sensitivity_scores": {k: scores[k] for k in sensitive[:10]}
            })
        
        except torch.cuda.OutOfMemoryError as e:
            # 捕获显存不足错误
            print(f"\n警告: 处理模型 {model_path} 时显存不足，跳过该模型")
            print(f"错误详情: {str(e)}")
            skipped_models.append(model_path)
        except Exception as e:
            # 捕获其他可能的错误
            print(f"\n错误: 处理模型 {model_path} 时发生异常: {str(e)}")
            skipped_models.append(model_path)
        finally:
            # 无论是否成功，都确保清理资源
            try:
                del analyzer
            except:
                pass
            # 强制清理显存
            torch.cuda.empty_cache()
            print(f"已清理显存，准备处理下一个模型")

    # 打印跳过的模型列表
    if skipped_models:
        print("\n以下模型由于显存不足或其他错误被跳过:")
        for i, model in enumerate(skipped_models):
            print(f"{i+1}. {model}")
    
    visualize_sensitivity_charts(all_results)

    return all_results, skipped_models

def test_attention_perturbation(model_path, num_samples=20, output_dir="sensitivity"):
    """测试注意力扰动方案对不同层的敏感度"""
    print(f"\n====== 测试注意力扰动方案: {model_path} ======")
    analyzer = ModelSensitivityAnalyzer(model_path=model_path)
    
    # 执行注意力扰动敏感度分析
    sensitive_layers, sensitivity_scores, is_attn_layer = analyzer.test_attention_perturbation_sensitivity(
        num_samples=num_samples
    )
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{analyzer.model_name}_attn_sensitivity.json")
    
    results = {
        "model_name": analyzer.model_name,
        "model_path": analyzer.model_path,
        "sensitive_layers": sensitive_layers,
        "sensitivity_scores": {k: float(v) for k, v in sensitivity_scores.items()},
        "is_attention_layer": is_attn_layer
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"注意力扰动敏感度分析结果已保存到 {output_file}")
    
    return output_file

if __name__ == "__main__":
    # Single Model
    # model_path = "path/to/model.safetensors"
    # analyze_model_sensitivity(model_path, num_samples=20, output_dir="sensitivity")
    
    # Multi Models
    # model_paths = ["path/to/model_1.safetensors", "path/to/model_2.safetensors"]
    # compare_model_sensitivities(model_paths, num_samples=30, output_dir="sensitivity")

    # Test the attention perturbation scheme
    model_path = "path/to/model.safetensors"
    test_attention_perturbation(model_path, num_samples=30)