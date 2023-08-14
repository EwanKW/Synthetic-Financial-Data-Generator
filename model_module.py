# Import Libraries
import numpy as np
import pandas as pd
from numpy.linalg import inv
from tqdm import tqdm
import random
from numpy.core.multiarray import concatenate
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

# Encoder
class Encoder(nn.Module):
    def __init__(self, seq_len, feat_dim, embed_dim=32):
        super(Encoder, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.hidden_dim = 2 * embed_dim
        
        self.rnn1 = nn.RNN(
            input_size=feat_dim, 
            hidden_size=self.hidden_dim, 
            num_layers=1, 
            batch_first=True
        )
        self.rnn2 = nn.RNN(
            input_size=self.hidden_dim, 
            hidden_size=embed_dim, 
            num_layers=1, 
            batch_first=True
        )
    
    def forward(self, x):
        batch_size = x.size(0)  # Get the batch size from the input tensor
        x, _ = self.rnn1(x)
        x, hidden_n = self.rnn2(x)
        hidden_n = hidden_n.view(batch_size, self.embed_dim)
        return hidden_n
    
# Decoder
class Decoder(nn.Module):
    def __init__(self, seq_len, feat_dim, input_dim, embed_dim):
        super(Decoder, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.embed_dim = embed_dim
        self.hidden_dim = 2 * embed_dim
        self.feat_dim = feat_dim
        
        self.fc = nn.Linear(input_dim, embed_dim)
        self.rnn1 = nn.RNN(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.RNN(
            input_size=embed_dim, 
            hidden_size=self.hidden_dim, 
            num_layers=1, 
            batch_first=True
        )
        self.output_layer_mean = nn.Linear(self.hidden_dim, feat_dim)
        self.output_layer_log_var = nn.Linear(self.hidden_dim, feat_dim)  # New linear layer for log_var
    
    def forward(self, x):
        batch_size = x.size(0)
        
        x = self.fc(x)
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        x, _ = self.rnn1(x)
        
        x, _ = self.rnn2(x)
        x = x.reshape(batch_size * self.seq_len, self.hidden_dim)
        
        # Separate linear layers for mean and log_var predictions
        x_mean = self.output_layer_mean(x)
        x_mean = x_mean.reshape(batch_size, self.seq_len, self.feat_dim)
        
        x_log_scale = self.output_layer_log_var(x)
        x_log_scale = x_log_scale.reshape(batch_size, self.seq_len, self.feat_dim)
        
        return x_mean, x_log_scale  # Return both mean and log_var
    
# VAE 
# adapted from (Falcon, 2022)

class VAE(pl.LightningModule):
    def __init__(self, seq_len, feat_dim, enc_out_dim, latent_dim, beta, learning_rate, min_std):
        super().__init__()

        self.save_hyperparameters()

        # encoder, decoder
        self.beta = beta
        self.min_std = min_std
        self.encoder = Encoder(seq_len, feat_dim, enc_out_dim)
        self.decoder = Decoder(seq_len, feat_dim, latent_dim, enc_out_dim)

        # distribution parameters
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)

        # for the gaussian likelihood
        # self.log_scale = nn.Parameter(torch.Tensor([0.0]))
        
        self.learning_rate = learning_rate

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def gaussian_likelihood(self, mean, logscale, sample):
        scale = torch.exp(logscale) + self.min_std
        dist = torch.distributions.Normal(mean, scale)
        log_pxz = dist.log_prob(sample)
        return log_pxz.sum(dim=(1, 2))

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def training_step(self, batch, batch_idx):
        x, _ = batch

        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample().flatten(1)

        # decoded 
        x_hat, x_log_scale = vae.decoder(z)

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, x_log_scale, x)

        # kl
        kl = self.kl_divergence(z, mu, std)

        # elbo
        elbo = (kl - (self.beta * recon_loss))
        elbo = elbo.mean()

        self.log_dict({
            'elbo': elbo,
            'kl': kl.mean(),
            'recon_loss': recon_loss.mean(), 
            'reconstruction': recon_loss.mean(),
            'kl': kl.mean(),
        })

        return elbo
    
    def on_train_epoch_end(self):
        if self.current_epoch % 5 == 0:
            elbo = self.trainer.logged_metrics['elbo']
            kl = self.trainer.logged_metrics['kl']
            recon_loss = self.trainer.logged_metrics['recon_loss']
            print(f"\rEpoch {self.current_epoch + 1}: \tELBO: {elbo:.4f}, (KL: {kl:.4f}, Recon Loss: {recon_loss:.4f})")
            
    def save_model(self, checkpoint_path):
        torch.save(self.state_dict(), checkpoint_path)
        
    def load_model(self, checkpoint_path):
        self.load_state_dict(torch.load(checkpoint_path))
        
# Conditional VAE
class CVAE(pl.LightningModule):
    def __init__(self, seq_len, feat_dim, enc_out_dim, latent_dim, beta, learning_rate, min_std):
        super().__init__()

        self.save_hyperparameters()

        # encoder, decoder
        self.beta = beta
        self.min_std = min_std
        self.encoder = Encoder(seq_len, feat_dim, enc_out_dim)
        self.decoder = Decoder(seq_len, feat_dim, latent_dim + 1, enc_out_dim)

        # distribution parameters
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)

        self.learning_rate = learning_rate

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def gaussian_likelihood(self, mean, logscale, sample):
        scale = torch.exp(logscale) + self.min_std
        dist = torch.distributions.Normal(mean, scale)
        log_pxz = dist.log_prob(sample)
        return log_pxz.sum(dim=(1, 2))

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def training_step(self, batch, batch_idx):
        x, y = batch

        # Encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # Sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample().flatten(1)

        # Decode the conditioned z
        y_reshaped = y.view(y.size(0), -1).float()
        conditioned_z = torch.cat((z, y_reshaped), dim=1)  # Concatenate z and y
        x_hat_mean, x_hat_log_scale = self.decoder(conditioned_z)

        # Reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat_mean, x_hat_log_scale, x)

        # KL divergence
        kl = self.kl_divergence(z, mu, std)

        # ELBO
        elbo = (kl - (self.beta * recon_loss))
        elbo = elbo.mean()

        self.log_dict({
            'elbo': elbo,
            'kl': kl.mean(),
            'recon_loss': recon_loss.mean(),
        })

        return elbo
    
    def on_train_epoch_end(self):
        if self.current_epoch % 5 == 0:
            elbo = self.trainer.logged_metrics['elbo']
            kl = self.trainer.logged_metrics['kl']
            recon_loss = self.trainer.logged_metrics['recon_loss']
            print(f"\rEpoch {self.current_epoch + 1}: \tELBO: {elbo:.4f}, (KL: {kl:.4f}, Recon Loss: {recon_loss:.4f})")
            
    def save_model(self, checkpoint_path):
        torch.save(self.state_dict(), checkpoint_path)
        
    def load_model(self, checkpoint_path):
        self.load_state_dict(torch.load(checkpoint_path))


