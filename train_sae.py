import jax
import jax.numpy as jnp
import optax
import wandb
from tqdm import tqdm
import numpy as np
from typing import Dict, Tuple
import os

from models.SaeBaseline import create_sae_model
from data.AddBiomechanicsDataset import AddBiomechanicsDataset, InputDataKeys, OutputDataKeys
from models.loss import autoencoder_loss, normalized_mean_squared_error, normalized_L1_loss

@jax.jit
def train_step(
    params: Dict,
    opt_state: optax.OptState,
    batch: Dict[str, jnp.ndarray],
    model: create_sae_model,
    optimizer: optax.GradientTransformation,
    l1_weight: float,
) -> Tuple[Dict, optax.OptState, Dict[str, float]]:
    """Single training step"""
    
    def loss_fn(params):
        # Forward pass
        latents_pre_act, latents, recons = model.apply(params, batch)
        
        # Compute loss
        loss = autoencoder_loss(
            reconstruction=recons,
            original_input=batch[InputDataKeys.POS],  # Using position as main reconstruction target
            latent_activations=latents,
            l1_weight=l1_weight
        )
        return loss, (latents_pre_act, latents, recons)
    
    # Compute gradients and update parameters
    (loss, (latents_pre_act, latents, recons)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    # Compute metrics
    metrics = {
        'loss': loss,
        'reconstruction_error': normalized_mean_squared_error(recons, batch[InputDataKeys.POS]),
        'sparsity': normalized_L1_loss(latents, batch[InputDataKeys.POS]),
        'latent_activation_mean': jnp.mean(jnp.abs(latents)),
        'latent_activation_std': jnp.std(latents)
    }
    
    return params, opt_state, metrics

def train(
    data_path: str,
    geometry_folder: str,
    window_size: int = 50,
    batch_size: int = 32,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    l1_weight: float = 0.1,
    n_latents: int = 512,
    activation: str = "relu",
    tied: bool = False,
    normalize: bool = True,
    k_sparsity: int = 100,
    wandb_project: str = "biomechanics-sae",
    wandb_entity: str = None,
):
    """Train the sparse autoencoder"""
    
    # Initialize wandb
    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        config={
            "window_size": window_size,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "l1_weight": l1_weight,
            "n_latents": n_latents,
            "activation": activation,
            "tied": tied,
            "normalize": normalize,
            "k_sparsity": k_sparsity,
        }
    )
    
    # Create dataset
    dataset = AddBiomechanicsDataset(
        data_path=data_path,
        window_size=window_size,
        geometry_folder=geometry_folder,
        stride=1
    )
    
    # Create model
    model = create_sae_model(
        num_dofs=dataset.num_dofs,
        num_joints=dataset.num_joints,
        history_len=window_size,
        root_history_len=window_size,
        n_latents=n_latents,
        activation=activation,
        tied=tied,
        normalize=normalize,
        k_sparsity=k_sparsity
    )
    
    # Initialize parameters
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, next(iter(dataset)))
    
    # Create optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    # Training loop
    for epoch in tqdm(range(num_epochs)):
        epoch_metrics = []
        
        # Create batches
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            
            # Convert batch to jax arrays
            batch = jax.tree_map(jnp.array, batch)
            
            # Training step
            params, opt_state, metrics = train_step(
                params=params,
                opt_state=opt_state,
                batch=batch,
                model=model,
                optimizer=optimizer,
                l1_weight=l1_weight
            )
            
            epoch_metrics.append(metrics)
        
        # Compute and log epoch metrics
        avg_metrics = {
            k: np.mean([m[k] for m in epoch_metrics])
            for k in epoch_metrics[0].keys()
        }
        wandb.log(avg_metrics, step=epoch)
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"checkpoints/sae_epoch_{epoch+1}.pkl"
            os.makedirs("checkpoints", exist_ok=True)
            with open(checkpoint_path, "wb") as f:
                jnp.save(f, params)
    
    # Save final model
    final_path = "checkpoints/sae_final.pkl"
    with open(final_path, "wb") as f:
        jnp.save(f, params)
    
    wandb.finish()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to biomechanics data")
    parser.add_argument("--geometry_folder", type=str, required=True, help="Path to geometry folder")
    parser.add_argument("--window_size", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--l1_weight", type=float, default=0.1)
    parser.add_argument("--n_latents", type=int, default=512)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--tied", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--k_sparsity", type=int, default=100)
    parser.add_argument("--wandb_project", type=str, default="biomechanics-sae")
    parser.add_argument("--wandb_entity", type=str, default=None)
    
    args = parser.parse_args()
    
    train(**vars(args)) 