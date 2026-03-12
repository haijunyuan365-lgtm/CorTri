import pickle
import numpy as np
import torch

def inspect_dataset(data, name="dataset", indent=0):
    """递归打印数据集的结构和关键信息"""
    spacing = "  " * indent
    if isinstance(data, dict):
        print(f"{spacing}📂 字典: '{name}' | 包含 {len(data)} 个键")
        for key, value in data.items():
            inspect_dataset(value, name=str(key), indent=indent + 1)
    elif isinstance(data, list):
        print(f"{spacing}📜 列表: '{name}' | 长度: {len(data)}")
        if len(data) > 0:
            print(f"{spacing}  -> 示例 (第一个元素):")
            inspect_dataset(data[0], name=f"{name}[0]", indent=indent + 1)
    elif isinstance(data, np.ndarray):
        print(f"{spacing}📊 NumPy Array: '{name}' | Shape: {data.shape} | Dtype: {data.dtype}")
    elif torch.is_tensor(data):
        print(f"{spacing}🔥 PyTorch Tensor: '{name}' | Shape: {data.shape} | Dtype: {data.dtype}")
    else:
        val_str = str(data)
        if len(val_str) > 50:
            val_str = val_str[:47] + "..."
        print(f"{spacing}🔹 基本数据 (Type: {type(data).__name__}): '{name}' | 值: {val_str}")

def main():
    # 替换为你的文件路径
    file_path = '/root/autodl-fs/ch-simsv2u.pkl' 
    print(f"正在加载文件: {file_path} ...\n")
    
    try:
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f) 
            
        print("加载成功！数据集结构如下：\n" + "-"*40)
        inspect_dataset(dataset)
        print("-" * 40)
        
    except Exception as e:
        print(f"读取文件时出错: {e}")

# 标准的执行入口
if __name__ == "__main__":
    main()