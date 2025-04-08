import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from data.AddBiomechanicsDataset import InputDataKeys, OutputDataKeys
import nimblephysics as nimble
import logging
import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn

class SparseAutoencoder(nn.Module):
    """Sparse autoencoder for kinematics reconstruction"""
    
    num_dofs: int
    num_joints: int
    history_len: int
    root_history_len: int
    n_latents: int
    activation: str = "relu"
    tied: bool = False
    normalize: bool = True
    k_sparsity: int = 100  # Number of active neurons in latent space
    
    @nn.compact
    def __call__(self, x: Dict[str, jnp.ndarray], training: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # 1. Concatenate and flatten inputs
        inputs = jnp.concatenate([
            x[InputDataKeys.POS],
            x[InputDataKeys.VEL],
            x[InputDataKeys.ACC],
            x[InputDataKeys.ROOT_LINEAR_VEL_IN_ROOT_FRAME],
            x[InputDataKeys.ROOT_ANGULAR_VEL_IN_ROOT_FRAME],
            x[InputDataKeys.ROOT_LINEAR_ACC_IN_ROOT_FRAME],
            x[InputDataKeys.ROOT_ANGULAR_ACC_IN_ROOT_FRAME],
            x[InputDataKeys.JOINT_CENTERS_IN_ROOT_FRAME],
            x[InputDataKeys.ROOT_POS_HISTORY_IN_ROOT_FRAME],
            x[InputDataKeys.ROOT_EULER_HISTORY_IN_ROOT_FRAME]
        ], axis=-1).reshape((x[InputDataKeys.POS].shape[0], -1))
        
        # 2. Preprocessing (normalization if enabled)
        if self.normalize:
            mu = jnp.mean(inputs, axis=-1, keepdims=True)
            std = jnp.std(inputs, axis=-1, keepdims=True)
            inputs = (inputs - mu) / (std + 1e-5)
        
        # 3. Encoder
        # pre_bias should have same shape as input features
        pre_bias = self.param('pre_bias', nn.initializers.zeros, (inputs.shape[-1],), jnp.float32)
        latent_bias = self.param('latent_bias', nn.initializers.zeros, (self.n_latents,), jnp.float32)
        
        # Encoder weights
        encoder = nn.Dense(self.n_latents, use_bias=False, name='encoder')
        latents_pre_act = encoder(inputs - pre_bias) + latent_bias
        
        # 4. Sparse activation
        if self.activation == "relu":
            latents = jax.nn.relu(latents_pre_act)
        elif self.activation == "topk":
            # Implement top-k sparsity
            topk_values, topk_indices = jax.lax.top_k(latents_pre_act, k=self.k_sparsity)
            latents = jnp.zeros_like(latents_pre_act)
            latents = latents.at[jnp.arange(latents.shape[0])[:, None], topk_indices].set(topk_values)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
        
        # 5. Decoder
        if self.tied:
            # Use transposed encoder weights
            decoder_weights = encoder.variables['params']['kernel'].T
            recons = jnp.dot(latents, decoder_weights) + pre_bias
        else:
            decoder = nn.Dense(inputs.shape[-1], use_bias=False, name='decoder')
            recons = decoder(latents) + pre_bias
        
        # 6. Denormalize if needed
        if self.normalize:
            recons = recons * std + mu
        
        return latents_pre_act, latents, recons
    
    def encode(self, x: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Encode input to latent space"""
        _, latents, _ = self(x)
        return latents
    
    def decode(self, latents: jnp.ndarray) -> jnp.ndarray:
        """Decode from latent space"""
        if self.tied:
            decoder_weights = self.variables['params']['encoder']['kernel'].T
            recons = jnp.dot(latents, decoder_weights) + self.variables['params']['pre_bias']
        else:
            decoder = nn.Dense(self.input_size, use_bias=False, name='decoder')
            recons = decoder(latents) + self.variables['params']['pre_bias']
        return recons

def create_sae_model(
    num_dofs: int,
    num_joints: int,
    history_len: int,
    root_history_len: int,
    n_latents: int = 512,
    activation: str = "relu",
    tied: bool = False,
    normalize: bool = True,
    k_sparsity: int = 100
) -> SparseAutoencoder:
    """Factory function to create a sparse autoencoder model"""
    
    # Calculate input size
    input_size = (num_dofs * 3 + 12 + num_joints * 3 + root_history_len * 6) * history_len
    
    return SparseAutoencoder(
        num_dofs=num_dofs,
        num_joints=num_joints,
        history_len=history_len,
        root_history_len=root_history_len,
        n_latents=n_latents,
        activation=activation,
        tied=tied,
        normalize=normalize,
        k_sparsity=k_sparsity
    )


