# SoundStream 配置文件
import torch
import os

# 模型参数
C_enc = 32  # 编码器通道数
C_dec = 32  # 解码器通道数
D = 512     # 量化器维度
n_q = 8     # 量化器层数
codebook_size = 1024  # 码本大小

# 判别器参数
num_D = 3   # Wave判别器数量
C = 32      # STFT判别器通道数
F = 512    # STFT判别器频率维度

# 训练参数
batch_size = 1
lr = 1e-4
epochs = 10000
sample_rate = 16000
device = "cuda:1" if torch.cuda.is_available() else "cpu"

# 损失函数权重
lambda_adv = 1.0      # 对抗损失权重
lambda_feat = 100.0     # 特征匹配损失权重  
lambda_rec = 1.0     # 重构损失权重 (降低权重以平衡数值差异)

# wandb配置
project_name = "soundstream-training"
experiment_name = "baseline"
log_interval = 10   # 每多少个batch记录一次
save_interval = 500 # 每多少个batch保存一次模型
generate_interval = 50 # 每多少个batch生成一次音频

# 模型保存路径
checkpoint_dir = "/data2/wl/SoundStreamcheckpoints"
final_model_path = "/data2/wl/SoundStream/final_model.pt"

# 确保保存目录存在
os.makedirs(checkpoint_dir, exist_ok=True)

# 创建wandb配置字典
def get_wandb_config():
    """返回用于wandb.init的配置字典"""
    return {
        "batch_size": batch_size,
        "learning_rate": lr,
        "epochs": epochs,
        "lambda_adv": lambda_adv,
        "lambda_feat": lambda_feat,
        "lambda_rec": lambda_rec,
        "C_enc": C_enc,
        "C_dec": C_dec,
        "D": D,
        "n_q": n_q,
        "codebook_size": codebook_size,
        "num_D": num_D,
    }
