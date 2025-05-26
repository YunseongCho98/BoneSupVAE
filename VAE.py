import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import ChestXrayPairDataset
from model import ConvVAE
from utils import loss_fn


if __name__ == "__main__":
    img_size   = 512
    batch_size = 4
    num_epochs = 3
    lr         = 1e-4
    z_dim      = 256

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    train_ds = ChestXrayPairDataset(
        "./BS_dataset_split/train",
        img_size=img_size,
        transform=transform
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    model = ConvVAE(z_dim=z_dim, in_channels=1, base_channels=64, img_size=img_size).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    best_total_loss = float('inf')
    for ep in range(num_epochs):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch [{ep+1}/{num_epochs}]" , leave=False)
        for x, y, _ in loop:
            x, y = x.to(device), y.to(device)
            x_recon, mu, logvar = model(x)
            loss = loss_fn(x_recon, y, mu, logvar, beta=1.0)

            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
            loop.set_postfix(avg_loss=total_loss / ((loop.n + 1) * batch_size))

        print(f"Epoch {ep+1}/{num_epochs}, Avg Loss: {total_loss/len(train_ds):.4f}")

        # checkpoint best models
        if total_loss < best_total_loss:
            best_total_loss = total_loss
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), f"convvae_ep50.pth")
            print(f"[Checkpoint] Saved best models at epoch {ep+1}, Avg_loss {best_total_loss/len(train_ds):.4f}")
