from utils.extractor import Extractor
from models.vgg import vgg16
import torch as t
from torch import nn

train_set_root = '/data1/zzl/dataset/sketch-triplet-train'  # where to save the training set
test_set_root = '/data1/zzl/dataset/sketch-triplet-test'  # where to save the testing set

train_photo_root = '/data1/zzl/dataset/photo-train'
test_photo_root = '/data1/zzl/dataset/photo-test'

SKETCH_RESNET = '/data1/zzl/model/caffe2torch/mixed_triplet_loss/sketch/sketch_resnet_85.pth'

PHOTO_RESNET = '/data1/zzl/model/caffe2torch/mixed_triplet_loss/photo/photo_resnet_85.pth'

SKETCH_VGG = '/data1/zzl/model/caffe2torch/vgg_triplet_loss/sketch/sketch_vgg_190.pth'
PHOTO_VGG = '/data1/zzl/model/caffe2torch/vgg_triplet_loss/photo/photo_vgg_190.pth'

FINE_TUNE_RESNET = '/data1/zzl/model/caffe2torch/fine_tune/model_270.pth'

device = 'cuda:1'

vgg = vgg16(pretrained=False).to(device)
vgg.classifier[6] = nn.Linear(in_features=4096, out_features=125, bias=True)
vgg.load_state_dict(t.load(PHOTO_VGG, map_location=t.device('cpu')))
vgg.to(device)

ext = Extractor(pretrained=False, device=device)
ext.reload_model(vgg)

photo_feature = ext.extract_new(test_photo_root, 'photo-vgg-190epoch.pkl')

vgg.load_state_dict(t.load(SKETCH_VGG, map_location=t.device('cpu')))
ext.reload_model(vgg)

sketch_feature = ext.extract_new(test_set_root, 'sketch-vgg-190epoch.pkl')

#print(t.load(PHOTO_VGG, map_location=t.device('cpu')))

