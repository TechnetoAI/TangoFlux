import os
import torch
import yaml
from datasets import load_dataset
from diffusers import AutoencoderOobleck
from tangoflux.utils import read_wav_file, pad_wav
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def precompute_vae(config_path):
    
    config = load_config(config_path)
    latent_dir = config["paths"]["vae_dir"]
    
    accelerator = Accelerator()

    # Load VAE model
    vae = AutoencoderOobleck.from_pretrained(
        "stabilityai/stable-audio-open-1.0", subfolder="vae"
    )
    vae.eval()
    vae.to("cuda:0")

    # Load dataset
    data_files = {
        "train": config["paths"]["train_file"],
        "validation": config["paths"]["val_file"],
        "test": config["paths"]["test_file"] or config["paths"]["val_file"]
    }
    raw_datasets = load_dataset("json", data_files=data_files)

    # Define batch size
    batch_size = config["training"].get("batch_size", 16)

    # Precompute VAE latents
    for split in raw_datasets.keys():
        dataset = raw_datasets[split]
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        latents = []
        for batch in tqdm(dataloader):
            audio_paths = batch["location"]
            audio_list = []

            for audio_path in audio_paths:
                wav = read_wav_file(audio_path, config["training"]["max_audio_duration"])
                if wav.shape[0] == 1:
                    wav = wav.repeat(2, 1)
                audio_list.append(wav)

            # Pad and stack audio inputs
            # audio_input = pad_wav(audio_list).to(accelerator.device)
            audio_input = torch.stack(audio_list, dim=0).cuda()
            

            with torch.no_grad():
                latent = vae.encode(audio_input).latent_dist.sample().cpu()
                
                print(latent.shape)
                
                
            for (idx, audio_path) in enumerate(audio_paths):
                filename = os.path.basename(audio_path).replace('.wav', '_latent.npz')
                # torch.save(latent[idx], os.path.join(latent_dir, filename))
                
                np.savez_compressed(os.path.join(latent_dir, filename), tensor=latent[idx].numpy())

                

if __name__ == "__main__":
    precompute_vae("configs/tangoflux_config.yaml")