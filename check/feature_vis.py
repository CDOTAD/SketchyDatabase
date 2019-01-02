from utils.feature_visualizer import FeatureVisualizer

test_set_root = '/data1/zzl/dataset/sketch-triplet-test'
test_photo_root = '/data1/zzl/dataset/photo-test'
vis = FeatureVisualizer(data_root=test_set_root, feature_root='../feature/sketch-vgg-190epoch.pkl',
                        env='caffe2torch_featurevis')
# vis.visualize(10, win='training together sketch')

vis.embedding_vis(em_data_root=test_photo_root, em_feature_root='../feature/photo-vgg-190epoch.pkl',
                  min_class=0, max_class=10, win='embedding')
