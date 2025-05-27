from discriminator import WaveDiscriminator, STFT_Discriminator
from generator import Generator
from loss_func import G_adversarial_loss, D_adversarial_loss, G_feature_loss, G_rec_loss, G_criterion, D_criterion

import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
import datasets
import config
import wandb
import logging
import numpy as np
import torchaudio

# 配置logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def collate_fn(batch):
    """
    Collate function for dataloader
    """
    arrays = [item['audio']['array'] for item in batch]
    # length of each array
    lengths = torch.tensor([len(array) for array in arrays])
    return torch.nn.utils.rnn.pad_sequence(arrays, batch_first=True), lengths

# Load dataset
ds = load_dataset("mythicinfinity/libritts", "dev", split="dev.clean[:100]")
ds = ds.train_test_split(test_size=0.1)
ds = ds.with_format("torch")

# Create dataloader
train_loader = DataLoader(ds["train"], batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(ds["test"], batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

# Create model
G = Generator(config.C_enc, config.C_dec, config.D, config.n_q, config.codebook_size).to(config.device)
D_wave = WaveDiscriminator(config.num_D).to(config.device)
D_STFT = STFT_Discriminator(config.C, config.F).to(config.device)

# Create optimizer
G_optimizer = torch.optim.Adam(G.parameters(), lr=config.lr)
D_optimizer = torch.optim.Adam(list(D_wave.parameters()) + list(D_STFT.parameters()), lr=config.lr)

# Initialize wandb
wandb.init(
    project=config.project_name,
    name=config.experiment_name,
    config=config.get_wandb_config()
)

# Watch models for gradient tracking
wandb.watch(G, log="all", log_freq=config.log_interval)
wandb.watch(D_wave, log="all", log_freq=config.log_interval)
wandb.watch(D_STFT, log="all", log_freq=config.log_interval)

total_steps = 0
epoch_loss_G = 0.0
epoch_loss_D = 0.0

for epoch in range(config.epochs):
    # train loop
    epoch_loss_G = 0.0
    epoch_loss_D = 0.0
    num_batches = 0
    
    for batch_idx, (x, lengths) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")):
        x = x.to(config.device)
        x = x.unsqueeze(1) # (batch_size, 1, seq_len)
        lengths = lengths.to(config.device)
        x_hat = G(x)

        # 裁切 x_hat , x 使得长度相同
        max_length = min(x.shape[2], x_hat.shape[2])
        x = x[:, :, :max_length]
        x_hat = x_hat[:, :, :max_length]

        logger.debug(f"x: {x.shape}")
        logger.debug(f"lengths: {lengths.shape}")
        logger.debug(f"x_hat: {x_hat.shape}")

        stft_x = torch.view_as_real(torch.stft(x.squeeze(1), n_fft=1024, hop_length=256, window=torch.hann_window(window_length=1024, device=config.device), return_complex=True)).permute(0, 3, 2, 1) # (batch, 2, time, freq)
        stft_x_hat = torch.view_as_real(torch.stft(x_hat.squeeze(1), n_fft=1024, hop_length=256, window=torch.hann_window(window_length=1024, device=config.device), return_complex=True)).permute(0, 3, 2, 1) # (batch, 2, time, freq)

        logger.debug(f"stft_x: {stft_x.shape}")
        logger.debug(f"stft_x_hat: {stft_x_hat.shape}")

        lengths_stft = D_STFT.cal_lengths(1+lengths)
        lengths_wave = D_wave.cal_lengths(lengths)

        # Train generator
        stft_outputs_x = D_STFT(stft_x)
        stft_outputs_x_hat = D_STFT(stft_x_hat)
        wave_outputs_x = D_wave(x)
        wave_outputs_x_hat = D_wave(x_hat)

        logger.debug(f"stft_outputs_x: {stft_outputs_x[-1].shape}")
        logger.debug(f"stft_outputs_x_hat: {stft_outputs_x_hat[-1].shape}")
        logger.debug(f"wave_outputs_x: {wave_outputs_x[0][-1].shape}")
        logger.debug(f"wave_outputs_x_hat: {wave_outputs_x_hat[0][-1].shape}")

        loss_G, loss_components = G_criterion(config, x, x_hat, stft_outputs_x, stft_outputs_x_hat, lengths_stft, wave_outputs_x, wave_outputs_x_hat, lengths_wave, return_components=True)

        G_optimizer.zero_grad()
        loss_G.backward()
        G_optimizer.step()

        # Train discriminator
        stft_outputs_x = D_STFT(stft_x)
        wave_outputs_x = D_wave(x)

        stft_outputs_x_hat_det = D_STFT(stft_x_hat.detach())
        wave_outputs_x_hat_det = D_wave(x_hat.detach())

        loss_D = D_criterion(stft_outputs_x, stft_outputs_x_hat_det, lengths_stft, wave_outputs_x, wave_outputs_x_hat_det, lengths_wave)

        D_optimizer.zero_grad()
        loss_D.backward()
        D_optimizer.step()
        
        # 累积损失
        epoch_loss_G += loss_G.item()
        epoch_loss_D += loss_D.item()
        num_batches += 1
        total_steps += 1
        
        # 生成音频并保存x和x_hat
        if total_steps % config.generate_interval == 0 and total_steps > 0:
            # 选择batch中的第一个样本
            sample_idx = 0
            x_sample = x[sample_idx:sample_idx+1].detach()  # [1, 1, seq_len]
            x_hat_sample = x_hat[sample_idx:sample_idx+1].detach()  # [1, 1, seq_len]
            original_length = lengths[sample_idx].item()
            
            # 裁切到原始长度（去除padding）
            x_sample = x_sample[:, :, :original_length]
            x_hat_sample = x_hat_sample[:, :, :original_length]
            
            # 转换为numpy并调整形状用于保存
            x_audio = x_sample.squeeze().cpu().numpy()  # [seq_len]
            x_hat_audio = x_hat_sample.squeeze().cpu().numpy()  # [seq_len]
            
            # 确保音频在[-1, 1]范围内
            x_audio = np.clip(x_audio, -1.0, 1.0)
            x_hat_audio = np.clip(x_hat_audio, -1.0, 1.0)
            
            # 创建保存目录
            audio_save_dir = os.path.join(config.checkpoint_dir, "generated_audio")
            os.makedirs(audio_save_dir, exist_ok=True)
            
            # 保存原始音频和重建音频
            original_path = os.path.join(audio_save_dir, f"original_epoch_{epoch}_step_{total_steps}.wav")
            reconstructed_path = os.path.join(audio_save_dir, f"reconstructed_epoch_{epoch}_step_{total_steps}.wav")
            
            # 使用torchaudio保存音频文件
            torchaudio.save(
                original_path,
                torch.from_numpy(x_audio).unsqueeze(0),  # [1, seq_len]
                config.sample_rate,
                encoding="PCM_S",
                bits_per_sample=16
            )
            
            torchaudio.save(
                reconstructed_path,
                torch.from_numpy(x_hat_audio).unsqueeze(0),  # [1, seq_len]
                config.sample_rate,
                encoding="PCM_S",
                bits_per_sample=16
            )
            
            logger.info(f"保存音频文件: {original_path}, {reconstructed_path}")
            
            # 记录到wandb
            wandb.log({
                "audio/original": wandb.Audio(x_audio, sample_rate=config.sample_rate),
                "audio/reconstructed": wandb.Audio(x_hat_audio, sample_rate=config.sample_rate),
                "audio/original_length": original_length,
            }, step=total_steps)

        # 记录训练指标
        if total_steps % config.log_interval == 0:
            wandb.log({
                "train/generator_loss": loss_G.item(),
                "train/discriminator_loss": loss_D.item(),
                "train/adversarial_loss": loss_components["adversarial_loss"].item(),
                "train/feature_loss": loss_components["feature_loss"].item(),
                "train/reconstruction_loss": loss_components["reconstruction_loss"].item(),
                "train/epoch": epoch + 1,
                "train/step": total_steps,
                "train/learning_rate": config.lr,
            }, step=total_steps)
        
        # 保存模型检查点
        if total_steps % config.save_interval == 0 and total_steps > 0:
            checkpoint = {
                'epoch': epoch,
                'step': total_steps,
                'generator_state_dict': G.state_dict(),
                'wave_discriminator_state_dict': D_wave.state_dict(),
                'stft_discriminator_state_dict': D_STFT.state_dict(),
                'g_optimizer_state_dict': G_optimizer.state_dict(),
                'd_optimizer_state_dict': D_optimizer.state_dict(),
                'loss_G': loss_G.item(),
                'loss_D': loss_D.item(),
            }
            checkpoint_path = os.path.join(config.checkpoint_dir, f"checkpoint_epoch_{epoch}_step_{total_steps}.pt")
            torch.save(checkpoint, checkpoint_path)
            # wandb.save(checkpoint_path)
    
    # 记录每个epoch的平均损失
    avg_loss_G = epoch_loss_G / num_batches
    avg_loss_D = epoch_loss_D / num_batches
    
    wandb.log({
        "epoch/avg_generator_loss": avg_loss_G,
        "epoch/avg_discriminator_loss": avg_loss_D,
        "epoch/number": epoch + 1,
    }, step=total_steps)
    
    logger.info(f"Epoch {epoch+1}/{config.epochs} - Avg G Loss: {avg_loss_G:.4f}, Avg D Loss: {avg_loss_D:.4f}")

# 训练结束，保存最终模型
final_checkpoint = {
    'epoch': config.epochs,
    'step': total_steps,
    'generator_state_dict': G.state_dict(),
    'wave_discriminator_state_dict': D_wave.state_dict(),
    'stft_discriminator_state_dict': D_STFT.state_dict(),
    'g_optimizer_state_dict': G_optimizer.state_dict(),
    'd_optimizer_state_dict': D_optimizer.state_dict(),
}
final_checkpoint_path = os.path.join(config.checkpoint_dir, config.final_model_path)
torch.save(final_checkpoint, final_checkpoint_path)
wandb.save(final_checkpoint_path)

# 结束wandb运行
wandb.finish()


        

        

        
        






    




