# download_data.py
import os
import mmsdk
from mmsdk import mmdatasdk as md

# 设置保存路径为当前目录下的 data 文件夹
DATA_PATH = './data'

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

print("开始下载 CMU-MOSEI 数据集 (可能需要较长时间，请保持网络通畅)...")

# 1. 下载 Highlevel 特征 (包含 Visual, Audio, Text)
# 注意：CMU 服务器有时候不稳定，如果报错请重试
try:
    md.mmdataset(md.cmu_mosei.highlevel, DATA_PATH)
except Exception as e:
    print(f"Highlevel 特征下载提示: {e}")

# 2. 下载 Labels (标签)
try:
    md.mmdataset(md.cmu_mosei.labels, DATA_PATH)
except Exception as e:
    print(f"Labels 下载提示: {e}")

print("下载流程结束，请检查 data 文件夹中的 .csd 文件。")