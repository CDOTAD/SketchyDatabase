import os
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


def make_dataset(root):
    images = []

    cnames = os.listdir(root)
    for cname in cnames:
        c_path = os.path.join(root, cname)
        if os.path.isdir(c_path):
            fnames = os.listdir(c_path)
            for fname in fnames:
                path = os.path.join(c_path, fname)
                images.append(path)

    return images


class ImageDataset(data.Dataset):
    def __init__(self, image_root):
        self.tranform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.image_root = image_root
        self.image_paths = sorted(make_dataset(self.image_root))

        self.len = len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]

        image = Image.open(image_path).convert('RGB')
        image = self.tranform(image)

        image_path = image_path.split('/')
        cname = image_path[-2]
        fname = image_path[-1]

        name = cname + '/' + fname

        return {'I': image, 'N': name}

    def __len__(self):
        return self.len

