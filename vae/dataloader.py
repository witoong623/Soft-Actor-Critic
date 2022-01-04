import os
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class VAEImageFolder(Dataset):
    def __init__(self, root):
        super().__init__()

        self.root = root
        self.images_list = os.listdir(root)
        self.transform = T.ToTensor()

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.images_list[index])
        img = Image.open(img_path)

        return self.transform(img)

    def __len__(self):
        return len(self.images_list)


def get_dataloader(root, batch_size, num_workers=0):
    dataset = VAEImageFolder(root)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)

    return loader
