import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ChestXrayPairDataset(Dataset):
    def __init__(self, root_dir, img_size, transform=None):
        self.original_dir = os.path.join(root_dir, "Original")
        self.target_dir   = os.path.join(root_dir, "BoneSuppression")

        IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
        self.filenames = sorted(
            fn for fn in os.listdir(self.original_dir)
            if fn.lower().endswith(IMG_EXTS)
        )

        self.img_size  = img_size
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fn   = self.filenames[idx]
        orig = Image.open(os.path.join(self.original_dir, fn)).convert("L")
        gt   = Image.open(os.path.join(self.target_dir,   fn)).convert("L")
        if self.transform:
            orig = self.transform(orig)
            gt   = self.transform(gt)
        # return image, target and filename
        return orig, gt, fn
