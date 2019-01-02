import os
from PIL import Image
import torch as t
import torchvision as tv
from torch import nn
import pickle
from utils.visualize import Visualizer


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

    @t.no_grad()
    def extract(self, data_root, out_root=None):
        feature = []
        name = []

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

        data = {'name': name, 'feature': feature}
        if out_root:
            out = open(out_root, 'wb')
            pickle.dump(data, out)

            out.close()

        return data

    @t.no_grad()
    def extract_new(self, data_root, out_root=None):
        feature = []
        name = []

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

    def reload_state_dict_with_fc(self, state_file):
        temp_model = tv.models.resnet34(pretrained=False)
        temp_model.fc = nn.Linear(512, 125)
        temp_model.load_state_dict(t.load(state_file))

        pretrained_dict = temp_model.state_dict()

        model_dict = self.model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

    def reload_state_dic(self, state_file):
        self.model.load_state_dict(t.load(state_file))

    def reload_model(self, model):
        self.model = model
