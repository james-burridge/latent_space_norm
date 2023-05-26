import torch.nn as nn
import torch
from torch.nn import functional as F

#Pytorch implementation of variational autoencoder used to encode
#LPC spectra

class spectra_VAE(nn.Module):
    def __init__(self, 
                 device='cuda', 
                 n_freq = 256, #Number of frequencies in LPC spectrum
                 z_dim=5, #Dimension of latent space
                 hidden_dims=[256,128,64], #Sizes of hidden layers in encoder and decoder
                 gauss_var=1.0): #Variance of decoder distribution
        super().__init__()
        self.device = device
        self.z_dim = z_dim
        self.n_freq = n_freq
        self.recon_weight=0.5/gauss_var
        
        #Setting up the encoder
        modules=[]
        in_features=n_freq
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features=in_features,
                              out_features=h_dim),
                nn.ReLU()   
                )
            )
            in_features=h_dim

        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(h_dim,self.z_dim) 
        self.fc_logvar = nn.Linear(h_dim,self.z_dim) 
        
        #Setting up the decoder
        modules=[]
        hidden_dims.reverse()
        in_features=self.z_dim
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features=in_features,
                              out_features=h_dim),
                nn.ReLU()   
                )
            )
            in_features=h_dim

        self.decoder = nn.Sequential(*modules)
        self.fc_final = nn.Linear(hidden_dims[-1],self.n_freq)
        #reverse the hidden_dims back to original order
        hidden_dims.reverse()
        
     
    def encode(self,x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self,mu,logvar):
        sig = torch.exp(0.5*logvar)
        eps = torch.randn_like(sig)
        return mu + eps*sig
    
    def decode(self,z):
        x = self.decoder(z)
        return self.fc_final(x)
    
    #Forward pass through module:
    #1. Encodes image to latent space distribution
    #2. Samples from this distribution
    #3. Decodes image from sampled latent point
    def forward(self,x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu,logvar)
        return self.decode(z), mu, logvar
    
    
    def loss_function(self, xhat, x, mu, logvar):
        #reconstruction loss
        recons_loss =self.recon_weight*F.mse_loss(xhat, x,reduction='sum')
        #kl divergence
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recons_loss, kld_loss   
    
    def sample(self, n):
        #print(self.z_dim,self.device)
        sample = torch.randn(n, self.z_dim).to(self.device)
        return self.decode(sample)

        
        
    