import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from edelta.ttn.vqvae import VQVAE
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset

# Assuming you have your VQVAE model defined and initialized as `vqvae`
# And you have your data loader `dataloader`


# Define the prior model (a simple Gaussian for this example)
class SimpleGaussianPrior(nn.Module):
    def __init__(self, latent_dim):
        super(SimpleGaussianPrior, self).__init__()
        self.latent_dim = latent_dim
        self.loc = nn.Parameter(torch.zeros(latent_dim))
        self.scale = nn.Parameter(torch.ones(latent_dim))

    def forward(self, z):
        # We're assuming z is the latent code from VQ-VAE
        # Here we calculate the KL divergence between z's distribution and our Gaussian prior
        distribution = Normal(self.loc, self.scale.exp())
        kl_divergence = torch.distributions.kl_divergence(
            Normal(
                z, torch.ones_like(z)
            ),  # Assuming z is normally distributed around its mean
            distribution,
        ).sum(
            -1
        )  # Sum over the last dimension to get KL divergence for each batch element
        return kl_divergence.mean()  # Mean over the batch


if __name__ == "__main__":
    data = torch.randn(2048, 8)
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=64, num_workers=18)
    # Initialize the prior model
    prior_model = SimpleGaussianPrior(
        latent_dim=64
    )  # Assuming 64-dimensional latent space

    vqvae = VQVAE()
    trainer = L.Trainer(
        max_epochs=50, log_every_n_steps=10, accelerator="auto", devices=1
    )
    trainer.fit(vqvae, dataloader)
    # Optimizer for both VQ-VAE and the prior
    optimizer = optim.Adam(
        list(vqvae.parameters()) + list(prior_model.parameters()), lr=0.001
    )

    num_epochs = 10
    # Training loop
    for epoch in range(num_epochs):
        batch_idx = 0
        for batch in dataloader:
            # Assuming batch is your data
            x = batch[0]

            # Forward pass through VQ-VAE
            reconstructed, vq_loss, latent_codes = vqvae(x)

            # Compute reconstruction loss
            reconstruction_loss = F.mse_loss(reconstructed, x)

            # Compute KL divergence with our prior
            kl_loss = prior_model(
                latent_codes.detach()
            )  # Detach because we don't want gradients to flow back through VQ-VAE here

            # Total loss
            loss = reconstruction_loss + vq_loss + kl_loss

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging or printing progress
            if batch_idx % 16 == 0:
                print(f"Epoch {epoch},  Loss: {loss.item()}")
            batch_idx += 1

    # After training, you can use the prior to generate new latent codes:
    def generate_new_samples(num_samples):
        prior_model.eval()
        with torch.no_grad():
            z = prior_model.loc + prior_model.scale * torch.randn(
                num_samples, prior_model.latent_dim
            )
            print(vqvae.decoder(z))
        # Now use vqvae.decode(z) to generate new data samples

    generate_new_samples(10)
