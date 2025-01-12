from transformers import T5EncoderModel, T5TokenizerFast
import torch
from diffusers import FluxTransformer2DModel
from torch import nn
import random
from typing import List
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import compute_density_for_timestep_sampling
import copy
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from typing import Optional, Union, List
from datasets import load_dataset, Audio
from math import pi
import inspect
import yaml


class StableAudioPositionalEmbedding(nn.Module):
    """Used for continuous time
    Adapted from Stable Audio Open.
    """

    def __init__(self, dim: int):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        times = times[..., None]
        freqs = times * self.weights[None] * 2 * pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((times, fouriered), dim=-1)
        return fouriered


class DurationEmbedder(nn.Module):
    """
    A simple linear projection model to map numbers to a latent space.

    Code is adapted from:
    https://github.com/Stability-AI/stable-audio-tools
    """

    def __init__(
        self,
        number_embedding_dim,
        min_value,
        max_value,
        internal_dim: Optional[int] = 256,
    ):
        super().__init__()
        self.time_positional_embedding = nn.Sequential(
            StableAudioPositionalEmbedding(internal_dim),
            nn.Linear(in_features=internal_dim + 1, out_features=number_embedding_dim),
        )

        self.number_embedding_dim = number_embedding_dim
        self.min_value = min_value
        self.max_value = max_value
        self.dtype = torch.float32

    def forward(self, floats: torch.Tensor):
        floats = floats.clamp(self.min_value, self.max_value)
        normalized_floats = (floats - self.min_value) / (self.max_value - self.min_value)

        embedder_dtype = next(self.time_positional_embedding.parameters()).dtype
        normalized_floats = normalized_floats.to(embedder_dtype)

        embedding = self.time_positional_embedding(normalized_floats)
        float_embeds = embedding.view(-1, 1, self.number_embedding_dim)
        return float_embeds


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler ({scheduler.__class__}) does not support custom timesteps."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler ({scheduler.__class__}) does not support custom sigmas."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class TangoFlux(nn.Module):
    def __init__(self, config, initialize_reference_model=False):
        super().__init__()

        self.num_layers = config.get("num_layers", 6)
        self.num_single_layers = config.get("num_single_layers", 18)
        self.in_channels = config.get("in_channels", 64)
        self.attention_head_dim = config.get("attention_head_dim", 128)
        self.joint_attention_dim = config.get("joint_attention_dim", 1024)
        self.num_attention_heads = config.get("num_attention_heads", 8)
        self.audio_seq_len = config.get("audio_seq_len", 645)
        self.max_duration = config.get("max_duration", 30)
        self.uncondition = config.get("uncondition", False)
        self.text_encoder_name = config.get("text_encoder_name", "google/flan-t5-large")

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
        self.noise_scheduler_copy = copy.deepcopy(self.noise_scheduler)
        self.max_text_seq_len = 64
        self.text_encoder = T5EncoderModel.from_pretrained(self.text_encoder_name)
        self.tokenizer = T5TokenizerFast.from_pretrained(self.text_encoder_name)
        self.text_embedding_dim = self.text_encoder.config.d_model

        self.fc = nn.Sequential(
            nn.Linear(self.text_embedding_dim, self.joint_attention_dim),
            nn.ReLU()
        )
        self.duration_emebdder = DurationEmbedder(
            self.text_embedding_dim, min_value=0, max_value=self.max_duration
        )

        self.transformer = FluxTransformer2DModel(
            in_channels=self.in_channels,
            num_layers=self.num_layers,
            num_single_layers=self.num_single_layers,
            attention_head_dim=self.attention_head_dim,
            num_attention_heads=self.num_attention_heads,
            joint_attention_dim=self.joint_attention_dim,
            pooled_projection_dim=self.text_embedding_dim,
            guidance_embeds=False,
        )

        self.beta_dpo = 2000  # used for DPO training

    def get_sigmas(self, timesteps, n_dim=3, dtype=torch.float32):
        device = self.text_encoder.device
        sigmas = self.noise_scheduler_copy.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler_copy.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def encode_text_classifier_free(self, prompt: List[str], num_samples_per_prompt=1):
        device = self.text_encoder.device
        batch = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids, attention_mask = batch.input_ids.to(device), batch.attention_mask.to(device)

        with torch.no_grad():
            prompt_embeds = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]

        prompt_embeds = prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
        attention_mask = attention_mask.repeat_interleave(num_samples_per_prompt, 0)

        uncond_tokens = [""]
        max_length = prompt_embeds.shape[1]
        uncond_batch = self.tokenizer(
            uncond_tokens,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        uncond_input_ids = uncond_batch.input_ids.to(device)
        uncond_attention_mask = uncond_batch.attention_mask.to(device)

        with torch.no_grad():
            negative_prompt_embeds = self.text_encoder(
                input_ids=uncond_input_ids, attention_mask=uncond_attention_mask
            )[0]

        negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_samples_per_prompt, 0)
        uncond_attention_mask = uncond_attention_mask.repeat_interleave(num_samples_per_prompt, 0)

        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        prompt_mask = torch.cat([uncond_attention_mask, attention_mask])
        boolean_prompt_mask = (prompt_mask == 1).to(device)
        return prompt_embeds, boolean_prompt_mask

    @torch.no_grad()
    def encode_text(self, prompt):
        device = self.text_encoder.device
        batch = self.tokenizer(
            prompt,
            max_length=self.max_text_seq_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids, attention_mask = batch.input_ids.to(device), batch.attention_mask.to(device)
        encoder_hidden_states = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        boolean_encoder_mask = (attention_mask == 1).to(device)
        return encoder_hidden_states, boolean_encoder_mask

    def encode_duration(self, duration: torch.Tensor):
        return self.duration_emebdder(duration)

    @torch.no_grad()
    def inference_flow(
        self,
        prompt,
        num_inference_steps=50,
        timesteps=None,
        guidance_scale=3,
        duration=10,
        disable_progress=False,
        num_samples_per_prompt=1,
        prefix=None, 
    ):
        """
        If `prefix` is not None, it should be a latents tensor of shape [batch_size, prefix_len, 64].
        We'll copy that prefix into the first prefix_len frames of 'latents'.
        The rest remains random, so we effectively get continuation from the prefix.
        """
        bsz = num_samples_per_prompt
        device = self.transformer.device
        scheduler = self.noise_scheduler

        if not isinstance(prompt, list):
            prompt = [prompt]
        if not isinstance(duration, torch.Tensor):
            duration = torch.tensor([duration], device=device)

        classifier_free_guidance = guidance_scale > 1.0
        duration_hidden_states = self.encode_duration(duration)

        # If classifier-free, double batch size and do negative embeddings
        if classifier_free_guidance:
            bsz = 2 * num_samples_per_prompt
            encoder_hidden_states, boolean_encoder_mask = self.encode_text_classifier_free(
                prompt, num_samples_per_prompt=num_samples_per_prompt
            )
            # repeat the duration for bsz
            duration_hidden_states = duration_hidden_states.repeat(bsz, 1, 1)
        else:
            encoder_hidden_states, boolean_encoder_mask = self.encode_text(prompt)
            duration_hidden_states = duration_hidden_states.repeat(bsz, 1, 1)

        # Pool text for pooled_projection
        mask_expanded = boolean_encoder_mask.unsqueeze(-1).expand_as(encoder_hidden_states)
        masked_data = torch.where(
            mask_expanded,
            encoder_hidden_states,
            torch.tensor(float("nan"), device=device),
        )
        pooled = torch.nanmean(masked_data, dim=1)
        pooled_projection = self.fc(pooled)

        # Concatenate text + duration => final shape ~ [bsz, seq_len, d]
        encoder_hidden_states = torch.cat([encoder_hidden_states, duration_hidden_states], dim=1)

        # Set up timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler, num_inference_steps, device, timesteps, sigmas
        )
        timesteps = timesteps.to(device)

        latents = torch.randn(num_samples_per_prompt, self.audio_seq_len, 64)
        weight_dtype = latents.dtype

        if prefix is not None:
            prefix_len = prefix.shape[1]
            latents[:, :prefix_len, :] = prefix

        progress_bar = tqdm(range(num_inference_steps), disable=disable_progress)
        txt_ids = torch.zeros(bsz, encoder_hidden_states.shape[1], 3).to(device)
        audio_ids = (
            torch.arange(self.audio_seq_len)
            .unsqueeze(0)
            .unsqueeze(-1)
            .repeat(bsz, 1, 3)
            .to(device)
        )

        timesteps = timesteps.to(device)
        latents = latents.to(device)
        encoder_hidden_states = encoder_hidden_states.to(device)

        for i, t in enumerate(timesteps):
            latents_input = torch.cat([latents] * 2) if classifier_free_guidance else latents
            noise_pred = self.transformer(
                hidden_states=latents_input,
                timestep=torch.tensor([t / 1000], device=device),
                guidance=None,
                pooled_projections=pooled_projection,
                encoder_hidden_states=encoder_hidden_states,
                txt_ids=txt_ids,
                img_ids=audio_ids,
                return_dict=False,
            )[0]

            if classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = scheduler.step(noise_pred, t, latents).prev_sample
            progress_bar.update(1)

        # No "cat" of prefix in the token dimension => no 1332 dimension
        return latents

    def forward(self, latents, prompt, duration=torch.tensor([10]), sft=True):
        device = latents.device
        audio_seq_length = self.audio_seq_len
        bsz = latents.shape[0]

        encoder_hidden_states, boolean_encoder_mask = self.encode_text(prompt)
        duration_hidden_states = self.encode_duration(duration)

        # pool text
        mask_expanded = boolean_encoder_mask.unsqueeze(-1).expand_as(encoder_hidden_states)
        masked_data = torch.where(mask_expanded, encoder_hidden_states, torch.tensor(float("nan"), device=device))
        pooled = torch.nanmean(masked_data, dim=1)
        pooled_projection = self.fc(pooled)

        # cat text+duration => shape [bs, seq_len, d]
        encoder_hidden_states = torch.cat([encoder_hidden_states, duration_hidden_states], dim=1)

        txt_ids = torch.zeros(bsz, encoder_hidden_states.shape[1], 3, device=device)
        audio_ids = (
            torch.arange(audio_seq_length).unsqueeze(0).unsqueeze(-1).repeat(bsz, 1, 3).to(device)
        )

        if sft:
            if self.uncondition:
                mask_indices = [k for k in range(len(prompt)) if random.random() < 0.1]
                if len(mask_indices) > 0:
                    encoder_hidden_states[mask_indices] = 0

            noise = torch.randn_like(latents)

            u = compute_density_for_timestep_sampling(
                weighting_scheme="logit_normal",
                batch_size=bsz,
                logit_mean=0,
                logit_std=1,
                mode_scale=None,
            )
            indices = (u * self.noise_scheduler_copy.config.num_train_timesteps).long()
            timesteps = self.noise_scheduler_copy.timesteps[indices].to(device=latents.device)
            sigmas = self.get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)

            noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise

            model_pred = self.transformer(
                hidden_states=noisy_model_input,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projection,
                img_ids=audio_ids,
                txt_ids=txt_ids,
                guidance=None,
                timestep=timesteps / 1000,
                return_dict=False,
            )[0]

            target = noise - latents
            loss = torch.mean(((model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1), dim=1)
            loss = loss.mean()
            raw_model_loss, raw_ref_loss, implicit_acc = 0, 0, 0

        else:
            # DPO scenario (unchanged)
            encoder_hidden_states = encoder_hidden_states.repeat(2, 1, 1)
            pooled_projection = pooled_projection.repeat(2, 1)
            noise = torch.randn_like(latents).chunk(2)[0].repeat(2, 1, 1)
            u = compute_density_for_timestep_sampling(
                weighting_scheme="logit_normal",
                batch_size=bsz // 2,
                logit_mean=0,
                logit_std=1,
                mode_scale=None,
            )
            indices = (u * self.noise_scheduler_copy.config.num_train_timesteps).long()
            timesteps = self.noise_scheduler_copy.timesteps[indices].to(device=latents.device)
            timesteps = timesteps.repeat(2)
            sigmas = self.get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
            noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise

            model_pred = self.transformer(
                hidden_states=noisy_model_input,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projection,
                img_ids=audio_ids,
                txt_ids=txt_ids,
                guidance=None,
                timestep=timesteps / 1000,
                return_dict=False,
            )[0]
            target = noise - latents

            model_losses = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            model_losses = model_losses.mean(dim=list(range(1, len(model_losses.shape))))
            model_losses_w, model_losses_l = model_losses.chunk(2)
            model_diff = model_losses_w - model_losses_l
            raw_model_loss = 0.5 * (model_losses_w.mean() + model_losses_l.mean())

            with torch.no_grad():
                ref_preds = self.ref_transformer(
                    hidden_states=noisy_model_input,
                    encoder_hidden_states=encoder_hidden_states,
                    pooled_projections=pooled_projection,
                    img_ids=audio_ids,
                    txt_ids=txt_ids,
                    guidance=None,
                    timestep=timesteps / 1000,
                    return_dict=False,
                )[0]
                ref_loss = F.mse_loss(ref_preds.float(), target.float(), reduction="none")
                ref_loss = ref_loss.mean(dim=list(range(1, len(ref_loss.shape))))
                ref_losses_w, ref_losses_l = ref_loss.chunk(2)
                ref_diff = ref_losses_w - ref_losses_l
                raw_ref_loss = ref_loss.mean()

            scale_term = -0.5 * self.beta_dpo
            inside_term = scale_term * (model_diff - ref_diff)
            implicit_acc = (scale_term * (model_diff - ref_diff) > 0).sum().float() / inside_term.size(0)
            loss = -1 * F.logsigmoid(inside_term).mean() + model_losses_w.mean()

        return loss, raw_model_loss, raw_ref_loss, implicit_acc

