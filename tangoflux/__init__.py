from diffusers import AutoencoderOobleck
import torch
from transformers import T5EncoderModel, T5TokenizerFast
from diffusers import FluxTransformer2DModel
from torch import nn
from typing import List
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import compute_density_for_timestep_sampling
import copy
import torch.nn.functional as F
import numpy as np
from tangoflux.model import TangoFlux
from huggingface_hub import snapshot_download
from tqdm import tqdm
from typing import Optional, Union, List
from datasets import load_dataset, Audio
from math import pi
import json
import inspect
import yaml
from safetensors.torch import load_file
from tangoflux.utils import read_wav_file


class TangoFluxInference:

    def __init__(
        self,
        name="declare-lab/TangoFlux",
        device="cuda" if torch.cuda.is_available() else "cpu",
        remote=True
    ):

        self.vae = AutoencoderOobleck()


        if remote:
            paths = snapshot_download(repo_id=name)
        else:
            paths = name
    
        vae_weights = load_file("{}/model.safetensors".format(paths))
        self.vae.load_state_dict(vae_weights)
        weights = load_file("{}/model_1.safetensors".format(paths))

        with open("{}/config.json".format(paths), "r") as f:
            config = json.load(f)
        self.model = TangoFlux(config)
        self.model.load_state_dict(weights, strict=False)
        # _IncompatibleKeys(missing_keys=['text_encoder.encoder.embed_tokens.weight'], unexpected_keys=[]) this behaviour is expected
        self.vae.to(device)
        self.model.to(device)
        
        
    def generate_longform(self, prompt, steps=25, duration=10, guidance_scale=4.5, audio_prefix_path="", audio_prefix_duration=10):
        wav = read_wav_file(
            audio_prefix_path, audio_prefix_duration
        )
        
        if (
            wav.shape[0] == 1
        ):  ## If this audio is mono, we repeat the channel so it become "fake stereo"
            wav = wav.repeat(2, 1)

        audio_input = torch.stack([wav], dim=0)
        audio_input = audio_input.to(self.vae.device)
        audio_latent = self.vae.encode(
            audio_input
        ).latent_dist.sample()
        
        
        first_prefix_len = audio_latent.shape[-1]
        
        audio_latent = audio_latent.transpose(
            1, 2
        )
        
        generated_audio = []
        target_audio_len = duration * self.vae.config.sampling_rate
        current_audio_len = 0
        
        
        pbar = tqdm(total=target_audio_len)
            
        while current_audio_len < target_audio_len:
            with torch.no_grad():
                latents = self.model.inference_flow(
                    prompt,
                    duration=30,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    prefix=audio_latent,
                    disable_progress=True
                )
                
                to_decode_latents = latents[:, :-first_prefix_len, :]
                # to_decode_latents = latents
                
                audio_latent = latents[:, -first_prefix_len:, :]
                
                wave = self.vae.decode(to_decode_latents.transpose(2, 1)).sample.cpu()[0]
                generated_audio.append(wave)
                
                current_audio_len += wave.shape[-1]
                
                pbar.update(wave.shape[-1])
            
            
        # waveform_end = int(duration * self.vae.config.sampling_rate)
        
        # wave = wave[:, :waveform_end]
        
        wave = torch.cat(generated_audio, dim=-1)
    

        return wave




def generate(self, prompt, steps=25, duration=10, guidance_scale=4.5):

        with torch.no_grad():
            latents = self.model.inference_flow(
                prompt,
                duration=duration,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
            )

            wave = self.vae.decode(latents.transpose(2, 1)).sample.cpu()[0]
        waveform_end = int(duration * self.vae.config.sampling_rate)
        wave = wave[:, :waveform_end]
        return wave
