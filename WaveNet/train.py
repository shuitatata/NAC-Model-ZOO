import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import wandb
from tqdm import tqdm
from config import config
import soundfile as sf

from model import WaveNet
from data_utils import VCTKSpeakerDataset, mu_law_encode, mu_law_decode

# --- Training Function ---
def train(model, dataloader, optimizer, criterion, device, config, epoch, global_step):
    """
    训练一个epoch
    
    Args:
        model: WaveNet模型
        dataloader: 数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 训练设备
        config: 配置字典
        epoch: 当前epoch数 (0-based)
        global_step: 全局步数
    
    Returns:
        updated_global_step: 更新后的全局步数
        avg_loss: 本epoch的平均损失
    """
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['epochs']}")
    for batch_idx, (segments_encoded, segments_onehot) in enumerate(pbar):
        # segments_encoded: (B, segment_length) - µ-law encoded integer indices for targets
        # segments_onehot: (B, segment_length, quantization_channels) - one-hot encoded for inputs
        segments_encoded = segments_encoded.to(device)
        segments_onehot = segments_onehot.to(device)

        # Prepare input for the model: one-hot encoded data, all but the last sample
        # Model input X: [s_0, s_1, ..., s_{L-2}] (one-hot encoded)
        input_onehot = segments_onehot[:, :-1, :]  # (B, segment_length - 1, quantization_channels)
        
        # Prepare target: encoded indices, all samples except the first one
        # Target Y: [s_1, s_2, ..., s_{L-1}] (integer indices)
        target_indices = segments_encoded[:, 1:]  # (B, segment_length - 1)
        
        optimizer.zero_grad()
        
        # Model forward pass
        # output_logits: (B, segment_length - 1, quantization_channels)
        output_logits = model(input_onehot)
        
        # Reshape for CrossEntropyLoss:
        output_logits_flat = output_logits.reshape(-1, config["quantization_channels"])
        target_indices_flat = target_indices.reshape(-1)
        
        loss = criterion(output_logits_flat, target_indices_flat)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"Batch": f"{batch_idx+1}/{num_batches}", "Loss": f"{loss.item():.4f}"})
        
        global_step += 1
        if global_step % config["log_interval"] == 0:
            wandb.log({"train_loss": loss.item()}, step=global_step)
    
    avg_loss = total_loss / num_batches
    return global_step, avg_loss

# --- Audio Generation Function ---
def generate_audio(model, length, quantization_channels, device, prime_samples_count, temperature=1.0):
    model.eval()
    # context: (1, prime_samples_count) which will be (1,1) after change
    context = torch.full((1, prime_samples_count), quantization_channels // 2, dtype=torch.long).to(device)
    generated_samples = []

    for _ in range(length):
        input_one_hot = torch.nn.functional.one_hot(context, num_classes=quantization_channels).float().to(device)
        # model expects (B, L, C)
        # output_logits will be (1, 1, quantization_channels) because input length is 1
        output_logits = model(input_one_hot) # (1, 1, Q)
        last_sample_logits = output_logits[:, -1, :] / temperature
        probabilities = torch.softmax(last_sample_logits, dim=-1)
        next_sample = torch.multinomial(probabilities, num_samples=1).squeeze()
        generated_samples.append(next_sample.item())
        context = torch.roll(context, shifts=-1, dims=1)
        context[0, -1] = next_sample
        
    return np.array(generated_samples)

def create_dummy_data(config):
    # Create a dummy directory for the script to run
    dummy_speaker_dir = "./dummy_vctk_speaker_data"
    os.makedirs(dummy_speaker_dir, exist_ok=True)
    config["audio_dir"] = dummy_speaker_dir

    # Create dummy audio files
    dummy_audio = np.zeros(config["sample_rate"] * 5, dtype=np.float32)
    sf.write(os.path.join(dummy_speaker_dir, "dummy_audio_001.flac"), dummy_audio, config["sample_rate"])
    sf.write(os.path.join(dummy_speaker_dir, "dummy_audio_002.flac"), dummy_audio, config["sample_rate"])
    model_receptive_fields_for_dummy = sum([2**i for i in range(config["wavenet_layer_size"])]*config["wavenet_stack_size"])
    config["segment_length"] = model_receptive_fields_for_dummy + 2048 # Ensure segment length is adequate
    config["epochs"] = 2

# --- Main Execution ---
if __name__ == "__main__":
    # Check if audio directory exists, create dummy data if not
    if not os.path.exists(config["audio_dir"]):
        print(f"Warning: audio_dir '{config['audio_dir']}' does not exist. Creating a dummy directory for the script to run.")
        print("Please replace this with your actual VCTK speaker directory.")
        create_dummy_data(config)

    # Initialize wandb
    wandb.init(project=config["wandb_project_name"], config=config)
    
    # Setup device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    print("Loading dataset...")
    dataset = VCTKSpeakerDataset(
        audio_dir=config["audio_dir"],
        segment_length=config["segment_length"],
        quantization_channels=config["quantization_channels"],
        target_sample_rate=config["sample_rate"]
    )
    if len(dataset) == 0:
        print("Dataset is empty. Check audio_dir and segment_length relative to audio duration.")
        exit(1)
    
    if config["max_dataset_samples"] is not None:
        dataset = Subset(dataset, range(config["max_dataset_samples"]))

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
    )
    print("Dataset loaded.")

    # Initialize model
    model = WaveNet(
        layers=config["wavenet_layer_size"],
        blocks=config["wavenet_stack_size"],
        classes=config["quantization_channels"], 
        res_channels=config["wavenet_res_channels"],
        skip_channels=config["wavenet_skip_channels"]
    ).to(device)
    
    wandb.watch(model, log="all")

    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss().to(device)

    # Training loop
    print("Starting training...")
    global_step = 0
    for epoch in range(config["epochs"]):
        global_step, avg_loss = train(model, dataloader, optimizer, criterion, device, config, epoch, global_step)
        
        # Log average loss for this epoch
        wandb.log({"epoch_avg_loss": avg_loss, "epoch": epoch + 1}, step=global_step)
        print(f"Epoch {epoch+1}/{config['epochs']} completed. Average loss: {avg_loss:.4f}")
        
        # Generate audio samples
        if (epoch + 1) % config["generation_interval"] == 0:
            model.eval()
            print("Generating audio sample...")
            with torch.no_grad():
                generated_audio_encoded = generate_audio(
                    model, 
                    config["generation_length"], 
                    config["quantization_channels"], 
                    device,
                    prime_samples_count=1, # Start with a single seed sample
                    temperature=config["generation_temperature"]
                ) 
            
            decoded_audio = mu_law_decode(generated_audio_encoded, config["quantization_channels"])
            wandb.log({
                f"generated_audio_epoch_{epoch+1}": wandb.Audio(
                    decoded_audio, 
                    caption=f"Epoch {epoch+1}", 
                    sample_rate=config["sample_rate"]
                )
            }, step=global_step)
            print("Audio sample generated and logged to wandb.")

        # Save model checkpoint
        if (epoch + 1) % config["save_interval"] == 0:
            os.makedirs(config["model_dir"], exist_ok=True)
            checkpoint_path = f"{config['model_dir']}/wavenet_speaker_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            wandb.save(checkpoint_path)
            print(f"Model saved to {checkpoint_path}")

    print("Training finished.")
    wandb.finish()



