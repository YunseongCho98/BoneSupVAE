import torch
from torch import nn

class ConvVAE(nn.Module):
    def __init__(self, z_dim=256, in_channels=1, base_channels=64, img_size=512):
        super().__init__()
        self.base_channels = base_channels
        self.z_dim = z_dim
        self.C = base_channels * 8
        self.h = img_size // (2**5)
        self.w = img_size // (2**5)

        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, 2, 1),
            nn.ReLU(True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, 4, 2, 1),
            nn.ReLU(True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*4, 4, 2, 1),
            nn.ReLU(True)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*8, 4, 2, 1),
            nn.ReLU(True)
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(base_channels*8, base_channels*8, 4, 2, 1),
            nn.ReLU(True)
        )

        feat_dim = self.C * self.h * self.w
        self.fc_mu     = nn.Linear(feat_dim, z_dim)
        self.fc_logvar = nn.Linear(feat_dim, z_dim)
        self.fc_dec    = nn.Linear(z_dim, feat_dim)

        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(self.C, self.C, 4, 2, 1),
            nn.ReLU(True)
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(self.C + base_channels*8, base_channels*4, 4, 2, 1),
            nn.ReLU(True)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*4 + base_channels*4, base_channels*2, 4, 2, 1),
            nn.ReLU(True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*2 + base_channels*2, base_channels, 4, 2, 1),
            nn.ReLU(True)
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels + base_channels, in_channels, 4, 2, 1),
            nn.Sigmoid()
        )

    def encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        flat = e5.view(e5.size(0), -1)
        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)
        return mu, logvar, (e1, e2, e3, e4)

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, skips):
        e1, e2, e3, e4 = skips
        batch_size = z.size(0)
        feat = self.fc_dec(z)
        d5 = feat.view(batch_size, self.C, self.h, self.w)
        d5 = self.dec5(d5)
        d4 = torch.cat([d5, e4], dim=1)
        d4 = self.dec4(d4)
        d3 = torch.cat([d4, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = torch.cat([d3, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = torch.cat([d2, e1], dim=1)
        out = self.dec1(d1)
        return out

    def forward(self, x):
        mu, logvar, skips = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, skips)
        return x_recon, mu, logvar