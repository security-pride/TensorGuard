import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
from adjustText import adjust_text
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors

def load_model_features(filename):
    """从JSON文件加载模型特征"""
    with open(filename, 'r') as f:
        return json.load(f)

def cluster_with_user_centers(model_features_list, center_models, n_clusters=None, use_gmm=False):
    # 提取数值特征
    feature_names = []
    feature_values = []
    model_names = []
    
    for model_feat in model_features_list:
        model_name = model_feat.get("model_name", "unknown")
        model_names.append(model_name)
        
        # 第一个模型决定特征名称
        if not feature_names:
            feature_names = [k for k, v in model_feat.items() 
                           if isinstance(v, (int, float)) and k != "total_params"]
        
        # 收集特征值
        values = [model_feat.get(name, 0) for name in feature_names]
        feature_values.append(values)
    
    # 转换为numpy数组
    X = np.array(feature_values)
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA降维可视化
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # 如果未指定聚类数量，则使用中心模型的数量
    if n_clusters is None:
        n_clusters = len(center_models)
    
    # 查找中心模型的索引
    center_indices = []
    for center_name in center_models:
        try:
            idx = model_names.index(center_name)
            center_indices.append(idx)
        except ValueError:
            print(f"警告: 未找到中心模型 '{center_name}'，将使用随机初始化")
    
    if use_gmm:
        from sklearn.mixture import GaussianMixture
        
        # 如果找不到任何指定的中心模型，使用随机初始化
        if not center_indices:
            print("未找到任何指定的中心模型，GMM将使用随机初始化")
            gmm = GaussianMixture(n_components=n_clusters, random_state=1)
            gmm.fit(X_scaled)
            clusters = gmm.predict(X_scaled)
        else:  
            # 使用指定的中心模型作为初始聚类中心进行K-means
            initial_centers = X_scaled[center_indices]
            
            # 如果指定的中心模型数量少于n_clusters，则随机添加其他中心
            if len(initial_centers) < n_clusters:
                print(f"指定的中心模型数量({len(initial_centers)})少于聚类数量({n_clusters})，将随机添加其他中心")
                temp_kmeans = KMeans(n_clusters=n_clusters-len(initial_centers), random_state=1)
                temp_kmeans.fit(X_scaled)
                initial_centers = np.vstack([initial_centers, temp_kmeans.cluster_centers_])
            
            # 使用指定的初始中心进行K-means聚类
            kmeans = KMeans(n_clusters=n_clusters, init=initial_centers, n_init=1)
            kmeans_labels = kmeans.fit_predict(X_scaled)
            
            # 使用K-means的结果初始化GMM
            gmm = GaussianMixture(n_components=n_clusters, random_state=1)
            
            # 手动设置GMM的初始均值为K-means的中心
            gmm.means_init = kmeans.cluster_centers_
            
            gmm.fit(X_scaled)
            clusters = gmm.predict(X_scaled)
            
            print("已使用基于指定中心的K-means结果初始化GMM")
        
        # 评估GMM聚类质量
        if len(np.unique(clusters)) > 1:
            silhouette_avg = silhouette_score(X_scaled, clusters)
            print(f"GMM聚类质量评分 (Silhouette): {silhouette_avg:.3f}")
        
        # 保存GMM模型参数供后续使用
        model_params = {
            "means": gmm.means_,
            "covariances": gmm.covariances_,
            "weights": gmm.weights_
        }
    else:
        # 如果找不到任何指定的中心模型，使用随机初始化
        if not center_indices:
            print("未找到任何指定的中心模型，将使用随机初始化")
            kmeans = KMeans(n_clusters=n_clusters, random_state=1)
            clusters = kmeans.fit_predict(X_scaled)
        else:
            # 使用指定的中心模型作为初始聚类中心
            initial_centers = X_scaled[center_indices]
            
            # 如果指定的中心模型数量少于n_clusters，则随机添加其他中心
            if len(initial_centers) < n_clusters:
                print(f"指定的中心模型数量({len(initial_centers)})少于聚类数量({n_clusters})，将随机添加其他中心")
                # 创建一个KMeans实例来获取随机中心
                temp_kmeans = KMeans(n_clusters=n_clusters-len(initial_centers), random_state=1)
                temp_kmeans.fit(X_scaled)
                # 合并指定的中心和随机中心
                initial_centers = np.vstack([initial_centers, temp_kmeans.cluster_centers_])
            
            # 使用指定的初始中心进行K-means聚类
            kmeans = KMeans(n_clusters=n_clusters, init=initial_centers, n_init=1)
            clusters = kmeans.fit_predict(X_scaled)
        
        # 评估聚类质量
        if len(np.unique(clusters)) > 1:
            silhouette_avg = silhouette_score(X_scaled, clusters)
            print(f"K-means聚类质量评分 (Silhouette): {silhouette_avg:.3f}")
    
    # 可视化结果
    plt.figure(figsize=(12, 10), dpi=100)
    
    # 创建自定义颜色映射
    distinct_colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())[:n_clusters]
    custom_cmap = ListedColormap(distinct_colors[:n_clusters])
    
    # 绘制聚类结果
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap=custom_cmap, 
                         s=150, edgecolor='black', linewidth=1)
    
    # 标记中心模型
    for idx in center_indices:
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], 
                   marker='*', s=300, edgecolor='black', linewidth=2, 
                   facecolor='none', label='Base Model' if idx == center_indices[0] else "")
    
    # 添加模型名称标签
    texts = []
    for i, model in enumerate(model_names):
        # 缩短过长的模型名称
        if len(model) > 20:
            model = model[:18] + '..'
        text = plt.text(X_pca[i, 0], X_pca[i, 1], model, fontsize=13, 
                       ha='center', va='bottom', weight='bold')
        texts.append(text)
    
    # 自动调整文本位置避免重叠
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red', lw=0.5))
    
    # 添加图例
    legend1 = plt.legend(*scatter.legend_elements(),
                        loc="upper right", title="Cluster")
    plt.gca().add_artist(legend1)
    plt.legend(loc="lower right")
        
    algorithm_name = "GMM" if use_gmm else "K-means"
    # plt.title(f'Cluster Result ({algorithm_name})', fontsize=14, fontweight='bold')
    plt.xlabel('Main Component 1', fontsize=18)
    plt.ylabel('Main Component 2', fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图像
    os.makedirs('imgs', exist_ok=True)
    plt.tight_layout()
    # plt.savefig('imgs/user_defined_centers_clustering.pdf', bbox_inches='tight')
    plt.savefig('imgs/cluster_result.png', bbox_inches='tight')
    plt.show()
    
    # 返回结果
    results = {
        "model_names": model_names,
        "clusters": clusters.tolist(),
        "center_models": center_models,
        "center_indices": center_indices,
        "pca_coords": X_pca.tolist(),
        "algorithm": "gmm" if use_gmm else "kmeans",
        "cluster_centers": kmeans.cluster_centers_.tolist()
    }
    
    return results

def predict_unknown_model_cluster(unknown_feature_file, cluster_results=None, feature_files=None, n_clusters=None, algorithm_weights=None):
    """
    预测未知模型属于哪个模型簇，并生成检测报告
    
    参数:
        unknown_feature_file: 未知模型的特征文件路径
        cluster_results: 已有的聚类结果，如果为None则会重新执行聚类
        feature_files: 用于聚类的特征文件列表或目录
        n_clusters: 聚类数量
        algorithm_weights: 聚类算法权重
        
    返回:
        包含预测结果和详细信息的字典
    """
    
    
    # 加载未知模型特征
    try:
        unknown_model = load_model_features(unknown_feature_file)
        unknown_model_name = unknown_model.get("model_name", "未知模型")
        print(f"已加载未知模型: {unknown_model_name}")
    except Exception as e:
        print(f"加载未知模型特征文件时出错: {e}")
        return None
    
    # 如果没有提供聚类结果，则执行聚类
    if cluster_results is None:
        if feature_files is None:
            print("错误: 未提供聚类结果或特征文件")
            return None
        print("执行聚类分析...")
        cluster_results = cluster_from_files_with_centers(feature_files, center_models=[], n_clusters=n_clusters, use_gmm=False)
    
    # 提取聚类结果中的关键信息
    model_names = cluster_results["model_names"]
    combined_clusters = cluster_results["clusters"]
    pca_coords = cluster_results["pca_coords"]
    center_models = cluster_results["center_models"]
    center_indices = cluster_results["center_indices"]
    
    # 获取特征名称和值
    reference_model = load_model_features(os.path.join("./features", model_names[0] + "_features.json"))
    feature_names = [k for k, v in reference_model.items() 
                   if isinstance(v, (int, float)) and k != "total_params"]
    
    # 提取未知模型的特征值
    unknown_values = [unknown_model.get(name, 0) for name in feature_names]
    
    # 创建所有模型的特征矩阵
    all_model_features = []
    for model_name in model_names:
        try:
            model_file = os.path.join("./features", model_name + "_features.json")
            if not os.path.exists(model_file):
                # 尝试直接使用model_name作为文件路径
                model_file = model_name
            model = load_model_features(model_file)
            values = [model.get(name, 0) for name in feature_names]
            all_model_features.append(values)
        except Exception as e:
            print(f"加载模型 {model_name} 时出错: {e}")
            all_model_features.append([0] * len(feature_names))  # 使用默认值
    
    # 标准化特征
    X = np.array(all_model_features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 标准化未知模型特征
    unknown_scaled = scaler.transform(np.array([unknown_values]))
    
    # 计算未知模型与各个聚类中心的距离
    cluster_centers = {}
    for i, cluster_id in enumerate(combined_clusters):
        if cluster_id not in cluster_centers:
            cluster_centers[cluster_id] = []
        cluster_centers[cluster_id].append(X_scaled[i])
    
    # 计算每个聚类的中心点
    for cluster_id in cluster_centers:
        cluster_centers[cluster_id] = np.mean(cluster_centers[cluster_id], axis=0)
    
    # 计算未知模型到各个聚类中心的距离
    distances = {}
    for cluster_id, center in cluster_centers.items():
        dist = euclidean_distances(unknown_scaled, center.reshape(1, -1))[0][0]
        distances[cluster_id] = dist
    
    # 找出最近的聚类
    closest_cluster = min(distances, key=distances.get)
    
    # 计算未知模型与中心模型的距离（只考虑中心模型）
    center_model_distances = {}
    for idx in center_indices:
        model_name = model_names[idx]
        dist = euclidean_distances(unknown_scaled, X_scaled[idx].reshape(1, -1))[0][0]
        center_model_distances[model_name] = dist
    
    # 找出最相似的中心模型
    if center_model_distances:
        most_similar_model = min(center_model_distances, key=center_model_distances.get)
        most_similar_distance = center_model_distances[most_similar_model]
        model_distances = center_model_distances
    else:
        # 如果没有中心模型，则考虑所有模型
        print("警告: 未找到任何中心模型，将考虑所有模型")
        model_distances = {}
        for i, model_name in enumerate(model_names):
            dist = euclidean_distances(unknown_scaled, X_scaled[i].reshape(1, -1))[0][0]
            model_distances[model_name] = dist
        most_similar_model = min(model_distances, key=model_distances.get)
        most_similar_distance = model_distances[most_similar_model]
    
    # 使用PCA将未知模型投影到二维空间
    pca = PCA(n_components=2)
    pca.fit(X_scaled)
    unknown_pca = pca.transform(unknown_scaled)
    
    # 可视化结果
    plt.figure(figsize=(12, 10), dpi=100)
    
    # 创建自定义颜色映射
    n_clusters = max(combined_clusters) + 1
    distinct_colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())[:n_clusters]
    custom_cmap = ListedColormap(distinct_colors[:n_clusters])
    
    # 绘制已知模型
    scatter = plt.scatter(np.array(pca_coords)[:, 0], np.array(pca_coords)[:, 1], 
                         c=combined_clusters, cmap=custom_cmap, 
                         s=150, edgecolor='black', linewidth=1, alpha=0.7)
    
    # 绘制未知模型（使用星形标记）
    plt.scatter(unknown_pca[0, 0], unknown_pca[0, 1], 
               marker='*', s=300, c='red', edgecolor='black', linewidth=2,
               label=f'Unknown Model: {unknown_model_name}')
    
    # 标记中心模型
    for idx in center_indices:
        plt.scatter(pca_coords[idx][0], pca_coords[idx][1], 
                   marker='*', s=300, edgecolor='black', linewidth=2, 
                   facecolor='none', label='Center Model' if idx == center_indices[0] else "")
    
    # 添加模型名称标签
    from adjustText import adjust_text
    texts = []
    for i, model in enumerate(model_names):
        if len(model) > 20:
            model = model[:18] + '..'
        text = plt.text(pca_coords[i][0], pca_coords[i][1], model, fontsize=9, 
                       ha='center', va='bottom')
        texts.append(text)
    
    # 为未知模型添加标签
    text = plt.text(unknown_pca[0, 0], unknown_pca[0, 1], unknown_model_name, 
                   fontsize=12, ha='center', va='bottom', weight='bold', color='red')
    texts.append(text)
    
    # 自动调整文本位置避免重叠
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red', lw=0.5))
    
    # 添加图例
    legend1 = plt.legend(*scatter.legend_elements(),
                        loc="upper right", title="Cluster")
    plt.gca().add_artist(legend1)
    plt.legend(loc="upper left")
    
    plt.title(f'Unknown Model Clustering Analysis: {unknown_model_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Main Component 1', fontsize=12)
    plt.ylabel('Main Component 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(f'imgs/unknown_model_prediction_{unknown_model_name}.png', bbox_inches='tight')
    plt.show()
    
    # 生成检测报告
    print("\n" + "="*50)
    if most_similar_distance > 7:
        print("未知模型与任何已知模型的距离超过了阈值，可能是一个量化模型或者新的模型。")
    else:
        print(f"预测聚类: 第{closest_cluster}组")
        print(f"最相似中心模型: {most_similar_model} (距离: {most_similar_distance:.4f})")
    
    print("\n与各中心模型的距离:")
    for model_name, dist in sorted(center_model_distances.items(), key=lambda x: x[1]):
        print(f"  {model_name}: {dist:.4f}")
    
    # 返回结果
    result = {
        "unknown_model": unknown_model_name,
        "predicted_cluster": closest_cluster,
        "most_similar_model": most_similar_model,
        "cluster_distances": distances,
        "model_distances": model_distances,
        "unknown_pca_coords": unknown_pca.tolist()
    }
    
    return result

def cluster_from_files_with_centers(feature_files, center_models, n_clusters=None, use_gmm=False):
    """
    从特征文件中加载模型特征，并使用指定的中心模型进行聚类
    
    参数:
        feature_files: 特征文件路径列表或目录
        center_models: 用作聚类中心的模型名称列表
        n_clusters: 聚类数量，如果为None则使用center_models的长度
        
    返回:
        聚类结果
    """
    all_model_features = []
    
    # 如果传入的是目录，则读取该目录下所有的json文件
    if len(feature_files) == 1 and os.path.isdir(feature_files[0]):
        directory = feature_files[0]
        feature_files = [
            os.path.join(directory, f) for f in os.listdir(directory)
            if f.endswith('_features.json')
        ]
        print(f"从目录 {directory} 中找到 {len(feature_files)} 个特征文件")
    
    # 加载所有特征文件
    for file_path in feature_files:
        if not os.path.exists(file_path):
            print(f"警告: 文件 {file_path} 不存在，已跳过")
            continue
            
        try:
            print(f"加载特征文件: {file_path}")
            model_features = load_model_features(file_path)
            all_model_features.append(model_features)
        except Exception as e:
            print(f"加载文件 {file_path} 时出错: {e}")
    
    # 如果没有成功加载任何特征文件，则返回
    if not all_model_features:
        print("错误: 未能加载任何有效的特征文件")
        return None
    
    # 执行聚类
    algorithm_name = "GMM" if use_gmm else "K-means"
    print(f"\n====== 开始基于用户指定中心的{algorithm_name}聚类分析 ======")
    print(f"指定的中心模型: {', '.join(center_models)}")
    cluster_results = cluster_with_user_centers(all_model_features, center_models, n_clusters, use_gmm)
    
    return cluster_results

def print_cluster_report(cluster_results):
    """打印聚类结果报告"""
    model_names = cluster_results["model_names"]
    clusters = cluster_results["clusters"]
    center_models = cluster_results["center_models"]
    
    # 按聚类分组模型
    cluster_groups = {}
    for i, cluster_id in enumerate(clusters):
        if cluster_id not in cluster_groups:
            cluster_groups[cluster_id] = []
        cluster_groups[cluster_id].append(model_names[i])
    
    # 打印报告
    print("\n" + "="*50)
    print("聚类结果报告")
    print("="*50)
    print(f"总模型数量: {len(model_names)}")
    print(f"聚类数量: {len(cluster_groups)}")
    print(f"指定的中心模型: {', '.join(center_models)}")
    
    print("\n各聚类的模型:")
    for cluster_id, models in sorted(cluster_groups.items()):
        print(f"\n聚类 {cluster_id} (包含 {len(models)} 个模型):")
        for model in models:
            # 标记中心模型
            if model in center_models:
                print(f"  * {model} [中心模型]")
            else:
                print(f"  - {model}")
    
    print("\n" + "="*50)

# 示例用法
if __name__ == "__main__":
    feature_directory = "./feature"
    
    # Initialize the cluster centroids
    center_models = ["gemma-3-4b-it", "Llama-3.1-8B", "llama-3.2-1b", "Llama-3.2-3B", "Mistral-7B-v0.1", "phi-4", "Qwen2.5-3B", "Qwen2.5-7B-Instruct"]

    results = cluster_from_files_with_centers([feature_directory], center_models)
    
    if results:
        print_cluster_report(results)
    
    # Replace the unknown model fingerprint for family classification
    unknown_model_file = "unknown-model_features.json" 
    prediction = predict_unknown_model_cluster(unknown_model_file, cluster_results=results)