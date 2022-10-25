import cv2
import os
import re
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from common.utils import center_crop

# for town7 outskirt
# mean [0.4652, 0.4417, 0.3799]
# std [0.0946, 0.1767, 0.1865]

class VAEImageFolder(Dataset):
    def __init__(self, root):
        super().__init__()

        self.root = root
        self.images_list = os.listdir(root)
        self.transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.4652, 0.4417, 0.3799],
                        std=[0.0946, 0.1767, 0.1865])
        ])

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.images_list[index])
        img = Image.open(img_path)

        return self.transform(img)

    def __len__(self):
        return len(self.images_list)


def get_vae_dataloader(root, batch_size, num_workers=0, is_train=True):
    dataset = VAEImageFolder(root)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers)

    return loader

class CarlaAITDataset(Dataset):
    def __init__(self) -> None:
        self.root = '/home/witoon/thesis/datasets/carla-ait/pred_seg_images'
        self.filename_pattern = r'rgb_(\d+).png'
        self.id_at_half = 5675
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.3171, 0.3183, 0.3779), (0.1406, 0.0594, 0.0925))
        ])

        self._load_sample_list()

    def __getitem__(self, idx):
        if idx < len(self.first_image_list):
            sample_tuples = self.first_image_list[idx]
        else:
            sample_tuples = self.second_image_list[idx - len(self.first_image_list)]

        images_filepath = [os.path.join(self.root, sample[1]) for sample in sample_tuples]
        images = [cv2.imread(image_filepath) for image_filepath in images_filepath]
        transformed_images_tensor = [self.transform(img) for img in images]

        return torch.cat(transformed_images_tensor)


    def __len__(self):
        return len(self.first_image_list) + len(self.second_image_list)

    def _load_sample_list(self):
        filenames = os.listdir(self.root)
        id_filename_list = []

        for filename in filenames:
            match = re.search(self.filename_pattern, filename)
            if not match:
                raise Exception(f'filename {filename} does not contain pattern')

            image_id = int(match.group(1))
            id_filename_list.append((image_id, filename))

        id_filename_list.sort(key=lambda item: item[0])

        second_section_idx = id_filename_list.index((self.id_at_half, f'rgb_{self.id_at_half}.png'))

        self.first_image_list = list(zip(id_filename_list[:second_section_idx], id_filename_list[1:second_section_idx]))
        self.second_image_list = list(zip(id_filename_list[second_section_idx:], id_filename_list[second_section_idx+1:]))
