import jax.numpy as jnp


def autoencoder_loss(
    reconstruction: jnp.ndarray,
    original_input: jnp.ndarray,
    latent_activations: jnp.ndarray,
    l1_weight: float,
) -> jnp.ndarray:
    """
    :param reconstruction: output of Autoencoder.decode (shape: [batch, n_inputs])
    :param original_input: input of Autoencoder.encode (shape: [batch, n_inputs])
    :param latent_activations: output of Autoencoder.encode (shape: [batch, n_latents])
    :param l1_weight: weight of L1 loss
    :return: loss (shape: [1])
    """
    return (
        normalized_mean_squared_error(reconstruction, original_input)
        + normalized_L1_loss(latent_activations, original_input) * l1_weight
    )


def normalized_mean_squared_error(
    reconstruction: jnp.ndarray,
    original_input: jnp.ndarray,
) -> jnp.ndarray:
    """
    :param reconstruction: output of Autoencoder.decode (shape: [batch, n_inputs])
    :param original_input: input of Autoencoder.encode (shape: [batch, n_inputs])
    :return: normalized mean squared error (shape: [1])
    """
    return jnp.mean(
        jnp.mean((reconstruction - original_input) ** 2, axis=1) / 
        (jnp.mean(original_input ** 2, axis=1) + 1e-8)
    )


def normalized_L1_loss(
    latent_activations: jnp.ndarray,
    original_input: jnp.ndarray,
) -> jnp.ndarray:
    """
    :param latent_activations: output of Autoencoder.encode (shape: [batch, n_latents])
    :param original_input: input of Autoencoder.encode (shape: [batch, n_inputs])
    :return: normalized L1 loss (shape: [1])
    """
    return jnp.mean(
        jnp.sum(jnp.abs(latent_activations), axis=1) / 
        (jnp.linalg.norm(original_input, axis=1) + 1e-8)
    )