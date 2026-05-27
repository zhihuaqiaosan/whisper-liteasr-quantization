"""
LiteASR 实验 - THCHS-30 噪声测试集批量分析
"""

import os
import whisper
import numpy as np
from sklearn.decomposition import PCA
import torch

def get_encoder_output(model, audio_path):
    """获取 Whisper 编码器输出"""
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).unsqueeze(0)
    with torch.no_grad():
        return model.encoder(mel).squeeze(0).cpu().numpy()

def main():
    print("="*60)
    print("LiteASR 实验：THCHS-30 噪声测试集低秩分析")
    print("="*60)
    
    # 加载模型
    print("\n加载 Whisper 模型...")
    model = whisper.load_model("tiny")
    
    # 数据路径（下载解压后的位置）
    data_dir = "./thchs30_data/test-noise"
    
    if not os.path.exists(data_dir):
        print(f"❌ 找不到 {data_dir}")
        print("请确保 test-noise.tgz 已解压到 thchs30_data 文件夹")
        return
    
    # 收集所有 wav 文件
    audio_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.wav'):
                audio_files.append(os.path.join(root, f))
    
    print(f"找到 {len(audio_files)} 个音频文件")
    
    # 提取激活值（限制数量）
    max_files = min(200, len(audio_files))
    outputs = []
    
    for i, f in enumerate(audio_files[:max_files]):
        out = get_encoder_output(model, f)
        outputs.append(out)
        if (i+1) % 20 == 0:
            print(f"  已处理 {i+1}/{max_files}")
    
    activations = np.concatenate(outputs, axis=0)
    print(f"\n激活值形状: {activations.shape}")
    
    # PCA 分析
    pca = PCA(n_components=0.95)
    pca.fit(activations)
    
    original_dim = activations.shape[1]
    compressed_dim = pca.n_components_
    compression_ratio = (1 - compressed_dim / original_dim) * 100
    
    print(f"\n原始维度: {original_dim}")
    print(f"压缩后维度: {compressed_dim}")
    print(f"压缩率: {compression_ratio:.1f}%")
    print(f"信息保留率: {pca.explained_variance_ratio_.cumsum()[-1]:.1%}")

if __name__ == "__main__":
    main()