from data.triplet_input import TripleDataset
from data.image_input import ImageDataset
import torch.utils.data


class TripleDataLoader(object):
    def __init__(self, opt):
        self.dataset = TripleDataset(opt.photo_root, opt.sketch_root)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            shuffle=True,
            batch_size=opt.batch_size,
            num_workers=4,
            drop_last=True
        )

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data


class ImageDataLoader(object):
    def __init__(self, opt):
        self.dataset = ImageDataset(opt.image_root)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            shuffle=False,
            batch_size=opt.batch_size,
            num_workers=4,
            drop_last=False
        )

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data