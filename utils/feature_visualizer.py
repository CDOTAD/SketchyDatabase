import visdom
import pickle
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class FeatureVisualizer(object):

    def __init__(self, data_root, feature_root, env):
        self.data_root = data_root

        self.feature_root = feature_root
        self.feature_data = pickle.load(open(self.feature_root, 'rb'))

        self.vis = visdom.Visdom(env=env)

        self.c_index = dict()
        cnames = sorted(os.listdir(data_root))
        for i, cname in enumerate(cnames):
            self.c_index[cname] = i

        cname_list, item_num = self._get_feature_info(data_root)
        self.cname_list = cname_list
        self.item_num = item_num

        # for (cname, num) in zip(cname_list, item_num):
        #     print(cname, ':', num)

    # visualize the self.data_root
    def visualize(self, max_class, win='Feature Vis'):

        calculate_num = 0

        for i in range(max_class):

            calculate_num += self.item_num[i]

            feature = self.feature_data['feature'][:calculate_num]
            name = self.feature_data['name'][:calculate_num]

            feature_class = []
            for f_name in name:
                class_name = f_name.split('/')[0]
                feature_class.append(self.c_index[class_name])
            # print(np.shape(feature))

            tsne = TSNE(n_components=2, init='pca', random_state=0)
            X_tsen = tsne.fit_transform(feature)

            # print(np.shape(X_tsen))

            # print(set(feature_class))
            plt.figure()
            plt.scatter(X_tsen[:, 0], X_tsen[:, 1], c=np.array(feature_class), marker='.', cmap=plt.cm.Spectral)
            plt.colorbar()
            plt.grid(True)
            plt.xlabel('1st')
            plt.ylabel('2nd')
            plt.title('Feature Vis')

            self.vis.matplot(plt, win=win, opts=dict(title=win))

    # embedding vis
    def embedding_vis(self, em_data_root, em_feature_root, max_class, min_class=0, win='embedding vis'):

        cal_a = 0
        cal_b = 0

        em_cname_list, em_item_num = self._get_feature_info(em_data_root)

        em_feature_data = pickle.load(open(em_feature_root, 'rb'))

        start_a = 0
        start_b = 0
        for i in range(min_class):
            start_a += self.item_num[i]
            start_b += em_item_num[i]

        for i in range(min_class, max_class):

            cal_a += self.item_num[i]
            cal_b += em_item_num[i]

            a_feature = self.feature_data['feature'][start_a:cal_a+start_a]
            a_name = self.feature_data['name'][start_a:cal_a+start_a]
            print('np.shape(a_feature)', np.shape(a_feature))

            b_feature = em_feature_data['feature'][start_b:cal_b+start_b]
            b_name = em_feature_data['name'][start_b:cal_b+start_b]
            print('np.shape(b_feature)', np.shape(b_feature))

            a_class = []
            for name in a_name:
                c_name = name.split('/')[0]
                a_class.append(self.c_index[c_name])

            b_class = []
            for name in b_name:
                c_name = name.split('/')[0]
                b_class.append(self.c_index[c_name])

            tsne = TSNE(n_components=2, init='pca', random_state=0)
            feature = np.append(a_feature, b_feature, axis=0)

            print('np.shape(feature) :', np.shape(feature))

            X_tsen = tsne.fit_transform(feature)

            X_a = X_tsen[:cal_a]
            X_b = X_tsen[cal_a:]

            print('np.shape(X_b) :', np.shape(X_b))
            print('np.shape(b_class) :', np.shape(b_class))
            print('a_class :', a_class)
            print('b_class :', b_class)

            plt.figure()
            plt.scatter(X_a[:, 0], X_a[:, 1], c=np.array(a_class), marker='.', cmap=plt.cm.Spectral)
            plt.scatter(X_b[:, 0], X_b[:, 1], c=np.array(b_class), marker='s', cmap=plt.cm.Spectral)

            plt.colorbar()
            plt.grid(True)
            plt.xlabel('1st')
            plt.ylabel('2nd')

            plt.title('Embedding Vis')
            self.vis.matplot(plt, win=win, opts=dict(title=win))

    def _get_feature_info(self, data_root):
        item_num = []
        cname_list = []

        cnames = sorted(os.listdir(data_root))
        for cname in cnames:
            num = len(os.listdir(os.path.join(data_root, cname)))

            item_num.append(num)
            cname_list.append(cname)

        return cname_list, item_num

