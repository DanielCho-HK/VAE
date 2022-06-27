from cv2 import log
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(400, 20)
        self.fc4 = nn.Linear(20, 400)
        self.fc5 = nn.Linear(400, 784)
    
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def decode(self, z):
        h = F.relu(self.fc4(z))
        return F.sigmoid(self.fc5(h))

    def forward(self, x):
        batchsz = x.size(0)
        x = x.view(batchsz, 784)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        x_recon = x_recon.view(batchsz, 1, 28, 28)
        return x_recon, mu, log_var
    










# class VAE(nn.Module):
#     def __init__(self):
#         super(VAE, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(784, 256),
#             nn.ReLU(),
#             nn.Linear(256, 64),
#             nn.ReLU(),
#             nn.Linear(64, 20),
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(10, 64),
#             nn.ReLU(),
#             nn.Linear(64, 256),
#             nn.ReLU(),
#             nn.Linear(256, 784),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         batchsz = x.size(0)
#         x = x.view(batchsz, 784)
#         h_ = self.encoder(x)
#         mu, sigma_ = h_.chunk(2, dim=1)
#         sigma = torch.exp(sigma_)
#         h = mu + sigma * torch.randn_like(sigma)
#         kld = torch.sum(sigma - (1 + sigma_) + torch.pow(mu, 2)) / batchsz
#         out = self.decoder(h)
#         out = out.view(batchsz, 1, 28, 28)
#         return out, kld





