import os
from PIL import Image
import torch as t
import torchvision as tv
from torch import nn
import pickle
from utils.visualize import Visualizer
from data import ImageDataLoader
import numpy as np


class Config(object):
    def __init__(self):
        return


class Extractor(object):

    def __init__(self, e_model, batch_size=128, cat_info=True,
                 vis=False, dataloader=False):
        self.batch_size = batch_size
        self.cat_info = cat_info

        self.model = e_model

        if dataloader:
            self.dataloader = dataloader
        else:
            self.transform = tv.transforms.Compose([
                tv.transforms.Resize(224),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        self.vis = vis
        if self.vis:
            self.viser = Visualizer('caffe2torch_test')

    # extract the inputs' feature via self.model
    # the model's output only contains the inputs' feature
    @t.no_grad()
    def extract(self, data_root, out_root=None):
        if self.dataloader:
            return self._extract_with_dataloader(data_root=data_root, cat_info=self.cat_info, out_root=out_root)
        else:
            return self._extract_without_dataloader(data_root=data_root, cat_info=self.cat_info, out_root=out_root)

    # extract the inputs' feature via self.model
    # the model's output contains both the inputs' feature and category info
    @t.no_grad()
    def _extract_without_dataloader(self, data_root, cat_info, out_root):
        feature = []
        name = []

        self.model.eval()

        cnames = sorted(os.listdir(data_root))

        for cname in cnames:
            c_path = os.path.join(data_root, cname)
            if os.path.isdir(c_path):
                fnames = sorted(os.listdir(c_path))
                for fname in fnames:
                    path = os.path.join(c_path, fname)

                    image = Image.open(path)
                    image = self.transform(image)
                    image = image[None]
                    image = image.cuda()

                    if self.vis:
                        self.viser.images(image.cpu().numpy() * 0.5 + 0.5, win='extractor')
                    out = self.model(image)
                    if cat_info:
                        i_feature = out[1]
                    else:
                        i_feature = out

                    feature.append(i_feature.cpu().squeeze().numpy())
                    name.append(cname + '/' + fname)

        data = {'name': name, 'feature': feature}
        if out_root:
            out = open(out_root, 'wb')
            pickle.dump(data, out)

            out.close()

        return data

    # extract the inputs' feature via self.model
    # the model's output contains both the inputs' feature and category info
    # the input is loaded by dataloader
    @t.no_grad()
    def _extract_with_dataloader(self, data_root, cat_info, out_root):
        names = []

        self.model.eval()

        opt = Config()
        opt.image_root = data_root
        opt.batch_size = 128

        dataloader = ImageDataLoader(opt)
        dataset = dataloader.load_data()

        for i, data in enumerate(dataset):
            image = data['I'].cuda()
            name = data['N']

            out = self.model(image)
            if cat_info:
                i_feature = out[1]
            else:
                i_feature = out
            if i == 0:
                feature = i_feature.cpu().squeeze().numpy()

            else:
                feature = np.append(feature, i_feature.cpu().squeeze().numpy(), axis=0)

            names += name

        data = {'name': names, 'feature': feature}
        if out_root:
            out = open(out_root, 'wb')
            pickle.dump(data, out)

            out.close()

        return data

    # reload model with model file
    # the reloaded model contains fully connection layer
    def reload_state_dict_with_fc(self, state_file):
        temp_model = tv.models.resnet34(pretrained=False)
        temp_model.fc = nn.Linear(512, 125)
        temp_model.load_state_dict(t.load(state_file))

        pretrained_dict = temp_model.state_dict()

        model_dict = self.model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

    # reload model with model file
    # the reloaded model doesn't contain fully connection layer
    def reload_state_dic(self, state_file):
        self.model.load_state_dict(t.load(state_file))

    # reload model with model object directly
    def reload_model(self, model):
        self.model = model
