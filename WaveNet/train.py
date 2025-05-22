import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, Subset
import numpy as np
import os
import wandb
from tqdm import tqdm

from model import WaveNet
from data_utils import VCTKSpeakerDataset, mu_law_encode, mu_law_decode, one_hot_encode

# --- Training Function ---
def train_wavenet(config):
    wandb.init(project=config["wandb_project_name"], config=config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data ---
    print("Loading dataset...")
    dataset = VCTKSpeakerDataset(
        audio_dir=config["audio_dir"],
        segment_length=config["segment_length"],
        quantization_channels=config["quantization_channels"],
        target_sample_rate=config["sample_rate"]
    )
    if len(dataset) == 0:
        print("Dataset is empty. Check audio_dir and segment_length relative to audio duration.")
        return
    
    if config["max_dataset_samples"] is not None:
        subset = Subset(dataset, range(config["max_dataset_samples"]))
    else:
        subset = dataset

    dataloader = DataLoader(
        subset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        
    )
    print("Dataset loaded.")

    # --- Model ---
    model = WaveNet(
        layers=config["wavenet_layer_size"],
        blocks=config["wavenet_stack_size"],
        classes=config["quantization_channels"], 
        res_channels=config["wavenet_res_channels"],
        skip_channels=config["wavenet_skip_channels"]
    ).to(device)
    
    wandb.watch(model, log="all")

    # --- Optimizer and Loss ---
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss().to(device)

    # --- Training Loop ---
    print("Starting training...")
    global_step = 0
    for epoch in range(config["epochs"]):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for batch_idx, segments_raw in enumerate(pbar):
            # segments_raw: (B, segment_length) - µ-law encoded integer indices
            segments_raw = segments_raw.to(device)

            # Prepare input for the model: one-hot encode all but the last sample
            # Model input X: [s_0, s_1, ..., s_{L-2}]
            input_sequence = segments_raw[:, :-1] # (B, segment_length - 1)
            input_one_hot = one_hot_encode(input_sequence, config["quantization_channels"]).to(device)
            
            # Prepare target: all samples except the first one
            # Target Y: [s_1, s_2, ..., s_{L-1}]
            target_indices = segments_raw[:, 1:] # (B, segment_length - 1)
            
            optimizer.zero_grad()
            
            # Model forward pass
            # output_logits: (B, segment_length - 1, quantization_channels)
            # Assuming model output length is same as input length due to causal conv design in model.py
            output_logits = model(input_one_hot)
            
            # Reshape for CrossEntropyLoss:
            output_logits_flat = output_logits.reshape(-1, config["quantization_channels"])
            target_indices_flat = target_indices.reshape(-1)
            
            loss = criterion(output_logits_flat, target_indices_flat)
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({"Batch": f"{batch_idx+1}/{len(dataloader)}", "Loss": f"{loss.item():.4f}"})
            
            global_step += 1
            if global_step % config["log_interval"] == 0:
                # print(f"Epoch: {epoch+1}/{config['epochs']} | Batch: {batch_idx+1}/{len(dataloader)} | Loss: {loss.item():.4f}")
                wandb.log({"train_loss": loss.item()}, step=global_step)
        
        if (epoch + 1) % config["generation_interval"] == 0:
            model.eval()
            print("Generating audio sample...")
            with torch.no_grad():
                generated_audio_encoded = generate_audio(model, 
                    config["generation_length"], 
                    config["quantization_channels"], 
                    device,
                    # prime_samples_count=receptive_field + 1, # Changed: prime with one sample
                    prime_samples_count=1, # Start with a single seed sample
                    temperature=config["generation_temperature"]) 
            
            decoded_audio = mu_law_decode(generated_audio_encoded, config["quantization_channels"])
            wandb.log({
                f"generated_audio_epoch_{epoch+1}": wandb.Audio(
                    decoded_audio, 
                    caption=f"Epoch {epoch+1}", 
                    sample_rate=config["sample_rate"]
                )
            }, step=global_step)
            print("Audio sample generated and logged to wandb.")

        if (epoch + 1) % config["save_interval"] == 0:
            checkpoint_path = f"{config['model_dir']}/wavenet_speaker_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            wandb.save(checkpoint_path)
            print(f"Model saved to {checkpoint_path}")

    print("Training finished.")
    wandb.finish()

# --- Audio Generation Function ---
def generate_audio(model, length, quantization_channels, device, prime_samples_count, temperature=1.0):
    model.eval()
    # context: (1, prime_samples_count) which will be (1,1) after change
    context = torch.full((1, prime_samples_count), quantization_channels // 2, dtype=torch.long).to(device)
    generated_samples = []

    for _ in range(length):
        input_one_hot = one_hot_encode(context, quantization_channels).to(device)
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


# --- Main Execution ---
if __name__ == "__main__":
    config = {
        "audio_dir": "./vctk/VCTK-Corpus/wav48/p225",
        "model_dir": "/data2/wl/wavenet/ckpts",
        "wandb_project_name": "wavenet-audio-generation",
        "wandb_entity": "shuitata",
        "sample_rate": 16000, 
        "quantization_channels": 256,
        "segment_length": 8000, 
        "max_dataset_samples": 100000,
        "wavenet_layer_size": 10,
        "wavenet_stack_size": 4,
        "wavenet_res_channels": 256,
        "wavenet_skip_channels": 256,
        "batch_size": 16,
        "epochs": 50,
        "learning_rate": 1e-4, 
        "num_workers": 8, 
        "log_interval": 20, 
        "save_interval": 5, 
        "generation_interval": 10, 
        "generation_length": 16000 * 2, 
        "generation_temperature": 0.8,
    }
    
    if not os.path.exists(config["audio_dir"]):
        print(f"Warning: audio_dir '{config['audio_dir']}' does not exist. Creating a dummy directory for the script to run.")
        print("Please replace this with your actual VCTK speaker directory.")

        # Create a dummy directory for the script to run
        dummy_speaker_dir = "./dummy_vctk_speaker_data"
        os.makedirs(dummy_speaker_dir, exist_ok=True)

        # Create dummy audio files
        import soundfile as sf # Import here for a one-time use
        dummy_audio = np.zeros(config["sample_rate"] * 5, dtype=np.float32)
        sf.write(os.path.join(dummy_speaker_dir, "dummy_audio_001.flac"), dummy_audio, config["sample_rate"])
        sf.write(os.path.join(dummy_speaker_dir, "dummy_audio_002.flac"), dummy_audio, config["sample_rate"])
        config["audio_dir"] = dummy_speaker_dir
        model_receptive_fields_for_dummy = sum([2**i for i in range(config["wavenet_layer_size"])]*config["wavenet_stack_size"])
        config["segment_length"] = model_receptive_fields_for_dummy + 2048 # Ensure segment length is adequate
        config["epochs"] = 2

    train_wavenet(config)



