import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

torch.set_float32_matmul_precision("medium")


class VQVAELayer(nn.Module):
    def __init__(self, input_dim, num_embeddings, embedding_dim):
        super(VQVAELayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # The embedding layer acts as our codebook
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, inputs):
        # inputs shape: [batch_size, sequence_length, input_dim]
        inputs_flattened = inputs.view(-1, self.embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(inputs_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight**2, dim=1)
            - 2 * torch.matmul(inputs_flattened, self.embeddings.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self.num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize
        quantized = torch.matmul(encodings, self.embeddings.weight).view_as(inputs)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        return quantized, loss, encoding_indices.view(inputs.shape[0], -1)


class VQVAE(L.LightningModule):
    def __init__(
        self, input_dim=8, hidden_dim=64, num_embeddings=512, embedding_dim=64
    ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.vq_layer = VQVAELayer(embedding_dim, num_embeddings, embedding_dim)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss, _ = self.vq_layer(z)
        return self.decoder(z_q), vq_loss, z

    def training_step(self, batch, batch_idx):
        if isinstance(batch, list) or isinstance(batch, tuple):
            x = batch[0]  # Assuming the data is the first item if it's a tuple/list
        else:
            x = batch

        # Ensure x is a tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)  # Assuming float32 is appropriate

        # Now proceed with your model
        reconstructed, vq_loss, encoded = self(x)

        reconstruction_loss = F.mse_loss(reconstructed, x)
        loss = reconstruction_loss + vq_loss

        # print(f"train_loss {loss:.04f} {reconstruction_loss:.04f} {vq_loss:.04f}")
        # self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        # self.log('reconstruction_loss', reconstruction_loss, on_step=True, on_epoch=True)
        # self.log('vq_loss', vq_loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        reconstructed, vq_loss = self(x)
        val_loss = F.mse_loss(reconstructed, x) + vq_loss
        self.log("val_loss", val_loss, on_epoch=True)


# Example usage
if __name__ == "__main__":
    # Example data
    data = torch.randn(1000, 8)
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=64, num_workers=18)

    # class ConfigVQVAE:
    #     nput_dim = 5
    #     hidden_dim = 64
    #     num_embeddings = 512
    #     embedding_dim = 64
    #

    model = VQVAE()
    trainer = L.Trainer(
        max_epochs=10, log_every_n_steps=10, accelerator="auto", devices=1
    )
    trainer.fit(model, dataloader)

    from pytorch_lightning.loggers import TensorBoardLogger

    # Create a logger
    tb_logger = TensorBoardLogger("logs/", name="my_model")

    # When creating your Trainer, pass the logger
    trainer = L.Trainer(max_epochs=10, logger=tb_logger, log_every_n_steps=1)

    # Fit your model
    trainer.fit(model, dataloader)
