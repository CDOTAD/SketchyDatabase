import numpy as np
from sklearn.neighbors import NearestNeighbors
from utils.extractor import Extractor
from utils.visualize import Visualizer
import torch as t


class Tester(object):

    def __init__(self, opt):

        # self.vis = opt.vis
        self.test_bs = opt.test_bs

        self.photo_net = opt.photo_net
        self.sketch_net = opt.sketch_net

        self.photo_test = opt.photo_test
        self.sketch_test = opt.sketch_test

        self.eps = 1e-8

    @t.no_grad()
    def _extract_feature(self):
        with t.no_grad():
            self.photo_net.eval()
            self.sketch_net.eval()

            extractor = Extractor(e_model=self.photo_net, vis=False, dataloader=True)
            photo_data = extractor.extract(self.photo_test)

            extractor.reload_model(self.sketch_net)
            sketch_data = extractor.extract(self.sketch_test)

            photo_name = photo_data['name']
            photo_feature = photo_data['feature']

            sketch_name = sketch_data['name']
            sketch_feature = sketch_data['feature']

        return photo_name, photo_feature, sketch_name, sketch_feature

    @t.no_grad()
    def _extract_feature_embedding(self):
        with t.no_grad():
            self.photo_net.eval()
            self.sketch_net.eval()

            extractor = Extractor(e_model=self.photo_net, cat_info=False, vis=False, dataloader=True)
            photo_data = extractor.extract(self.photo_test, batch_size=self.test_bs)

            extractor.reload_model(self.sketch_net)
            sketch_data = extractor.extract(self.sketch_test, batch_size=self.test_bs)

            photo_name = photo_data['name']
            photo_feature = photo_data['feature']

            sketch_name = sketch_data['name']
            sketch_feature = sketch_data['feature']

        return photo_name, photo_feature, sketch_name, sketch_feature

    @t.no_grad()
    def test_category_recall(self):

        photo_name, photo_feature, sketch_name, sketch_feature = self._extract_feature()

        nbrs = NearestNeighbors(n_neighbors=np.size(photo_feature, 0),
                                algorithm='brute', metric='euclidean').fit(photo_feature)

        count_1 = 0
        count_5 = 0
        K = 5
        for ii, (query_sketch, query_name) in enumerate(zip(sketch_feature, sketch_name)):
            query_sketch = np.reshape(query_sketch, [1, np.shape(query_sketch)[0]])

            query_split = query_name.split('/')
            query_class = query_split[0]

            distance, indices = nbrs.kneighbors(query_sketch)

            for i, indice in enumerate(indices[0][:K]):

                retrieved_name = photo_name[indice]
                retrieved_class = retrieved_name.split('/')[0]

                if retrieved_class == query_class:

                    if i == 0:
                        count_1 += 1
                    count_5 += 1
                    break

        recall_1 = count_1 / (ii + 1)
        recall_5 = count_5 / (ii + 1)

        print('recall@1 :', recall_1, '    recall@5 :', recall_5)
        return {'recall@1': recall_1, 'recall@5': recall_5}

    @t.no_grad()
    def test_instance_recall(self):
        photo_name, photo_feature, sketch_name, sketch_feature = self._extract_feature()

        nbrs = NearestNeighbors(n_neighbors=np.size(photo_feature, 0),
                                algorithm='brute', metric='euclidean').fit(photo_feature)

        count_1 = 0
        count_5 = 0
        K = 5
        for ii, (query_sketch, query_name) in enumerate(zip(sketch_feature, sketch_name)):
            query_sketch = np.reshape(query_sketch, [1, np.shape(query_sketch)[0]])

            query_split = query_name.split('/')
            query_class = query_split[0]
            query_img = query_split[1]

            distance, indices = nbrs.kneighbors(query_sketch)

            for i, indice in enumerate(indices[0][:K]):

                retrieved_name = photo_name[indice]
                retrieved_class = retrieved_name.split('/')[0]

                retrieved_name = retrieved_name.split('/')[1]
                retrieved_name = retrieved_name.split('.')[0]
                if retrieved_class == query_class:
                    if query_img.split('-')[0] == retrieved_name:
                        if i == 0:
                            count_1 += 1
                        count_5 += 1
                        break

        recall_1 = count_1 / (ii + 1)
        recall_5 = count_5 / (ii + 1)

        print('recall@1 :', recall_1, '    recall@5 :', recall_5)
        return {'recall@1': recall_1, 'recall@5': recall_5}











