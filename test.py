import os
import torch
import torchvision.utils as vutils
from VAE import ConvVAE
from datasets import ChestXrayPairDataset
from torchvision import transforms
from torch.utils.data import DataLoader

def main():
    img_size   = 512
    batch_size = 4
    z_dim      = 256
    base_ch    = 64

    model_path = "convvae_ep50.pth"
    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)

    device = (torch.device('mps') 
            if (torch.backends.mps.is_available() and torch.backends.mps.is_built()) 
            else torch.device('cuda') 
            if torch.cuda.is_available() 
            else torch.device('cpu'))
    print("Using device:", device)

    model = ConvVAE(z_dim=z_dim, in_channels=1, base_channels=base_ch).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    test_ds = ChestXrayPairDataset(
        root_dir="./BS_dataset_split/test",
        img_size=img_size,
        transform=transform
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    with torch.no_grad():
        for batch_idx, (x, y, fnames) in enumerate(test_loader):
            x = x.to(device)
            x_recon, mu, logvar = model(x)

            for i in range(x.size(0)):
                name, _ = os.path.splitext(fnames[i])

                # save reconstructed image
                vutils.save_image(
                    x_recon[i],
                    os.path.join(output_dir, f"recon_{name}.png")
                )
                # save ground truth
                vutils.save_image(
                    y[i].to(device),
                    os.path.join(output_dir, f"gt_{name}.png")
                )
                # save original input
                vutils.save_image(
                    x[i],
                    os.path.join(output_dir, f"orig_{name}.png")
                )

            break

    print(f"Saved reconstruction samples to {output_dir}")


if __name__ == "__main__":
    main()