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

    def __init__(self, pretrained=False, vis=True, e_model=None):
        if e_model:
            self.model = e_model

        else:
            model = tv.models.resnet34(pretrained=pretrained)
            del model.fc
            model.fc = lambda x: x
            model.cuda()

            self.model = model

        self.transform = tv.transforms.Compose([
            tv.transforms.Resize(224),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # self.device = device
        self.vis = vis
        if self.vis:
            self.viser = Visualizer('caffe2torch_test')

    # extract the inputs' feature via self.model
    # the model's output only contains the inputs' feature
    @t.no_grad()
    def extract(self, data_root, out_root=None):
        feature = []
        name = []

        self.model.eval()

        cnames = sorted(os.listdir(data_root))

        for cname in cnames:
            fnames = sorted(os.listdir(os.path.join(data_root, cname)))
            for fname in fnames:
                path = os.path.join(data_root, cname, fname)

                image = Image.open(path)
                image = self.transform(image)
                image = image[None]
                image = image.cuda()

                if self.vis:
                    self.viser.images(image.cpu().numpy()*0.5 + 0.5, win='extractor')

                i_feature = self.model(image)

                feature.append(i_feature.cpu().squeeze().numpy())
                name.append(cname + '/' + fname)

        # 'name': category_name/file_name 'feature' : (1, D) array
        data = {'name': name, 'feature': feature}
        if out_root:
            out = open(out_root, 'wb')
            pickle.dump(data, out)

            out.close()

        return data

    # extract the inputs' feature via self.model
    # the model's output contains both the inputs' feature and category info
    @t.no_grad()
    def extract_new(self, data_root, out_root=None):
        feature = []
        name = []

        self.model.eval()

        cnames = sorted(os.listdir(data_root))

        for cname in cnames:
            fnames = sorted(os.listdir(os.path.join(data_root, cname)))
            for fname in fnames:
                path = os.path.join(data_root, cname, fname)

                image = Image.open(path)
                image = self.transform(image)
                image = image[None]
                image = image.cuda()

                if self.vis:
                    self.viser.images(image.cpu().numpy() * 0.5 + 0.5, win='extractor')

                _, i_feature = self.model(image)

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
    def extract_with_dataloader(self, data_root, batch_size, out_root=None):
        names = []

        self.model.eval()

        opt = Config()
        opt.image_root = data_root
        opt.batch_size = batch_size

        data_loader = ImageDataLoader(opt)
        dataset = data_loader.load_data()

        for i, data in enumerate(dataset):
            image = data['I'].cuda()
            name = data['N']

            _, i_feature = self.model(image)

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
