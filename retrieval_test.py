import pickle
from sklearn.neighbors import NearestNeighbors
import numpy as np
from PIL import Image
import os
import torchvision.transforms as transforms
from utils.visualize import Visualizer
import tqdm

PHOTO_ROOT = '/data1/zzl/dataset/photo-test'
SKETCH_ROOT = '/data1/zzl/dataset/sketch-triplet-test'

photo_data = pickle.load(open('feature/photo-resnet50_64-265.pkl', 'rb'))
sketch_data = pickle.load(open('feature/sketch-resnet50_64-265.pkl', 'rb'))
# print(photo_data['name'][0])
photo_feature = photo_data['feature']
photo_name = photo_data['name']

sketch_feature = sketch_data['feature']
sketch_name = sketch_data['name']
# print(np.size(photo_feature, 0))

nbrs = NearestNeighbors(n_neighbors=np.size(photo_feature, 0),
                        algorithm='brute', metric='euclidean').fit(photo_feature)

s_len = np.size(sketch_feature, 0)
picked = [0] * s_len

transform = transforms.Compose([
    transforms.ToTensor()
])

vis = Visualizer('caffe2torch_test')
count = 0
for ii in range(20):
    index = np.random.randint(0, s_len)
    while picked[index]:
        index = np.random.randint(0, s_len)

    picked[index] = 1

    query_feature = sketch_feature[index]
    query_feature = np.reshape(query_feature, [1, np.shape(query_feature)[0]])
    query_name = sketch_name[index]
    # print(query_name)
    distances, indices = nbrs.kneighbors(query_feature)

    query_split = query_name.split('/')
    query_class = query_split[0]
    query_img = query_split[1]

    query_image = np.array(Image.open(os.path.join(SKETCH_ROOT, query_name)).convert('RGB'))
    for i, indice in enumerate(indices[0][:5]):
        retrievaled_name = photo_name[indice]
        retrievaled_im = np.array(Image.open(os.path.join(PHOTO_ROOT, retrievaled_name)).convert('RGB'))
        query_image = np.append(query_image, retrievaled_im, axis=1)

        retrievaled_class = retrievaled_name.split('/')[0]
        retrievaled_name = retrievaled_name.split('/')[1]
        retrievaled_name = retrievaled_name.split('.')[0]

        if retrievaled_class == query_class:
            print(ii, 'correct class', query_name, retrievaled_name)
            if query_img.find(retrievaled_name) != -1:
                print(ii, 'correct item', query_name)
                count += 1

    if ii == 0:
        result = query_image
    else:
        result = np.append(result, query_image, axis=0)

result = transform(result)
vis.images(result.numpy(), win='result')

print(count)


count = 0
count_5 = 0
K = 5

div = 0

for ii, (query_sketch, query_name) in tqdm.tqdm(enumerate(zip(sketch_feature, sketch_name))):
    query_sketch = np.reshape(query_sketch, [1, np.shape(query_sketch)[0]])

    query_split = query_name.split('/')
    query_class = query_split[0]
    query_img = query_split[1]

    distances, indices = nbrs.kneighbors(query_sketch)

    div += distances[0][1] - distances[0][0]

    # top K

    for i, indice in enumerate(indices[0][:K]):

        retrievaled_name = photo_name[indice]
        retrievaled_class = retrievaled_name.split('/')[0]

        retrievaled_name = retrievaled_name.split('/')[1]
        retrievaled_name = retrievaled_name.split('.')[0]

        if retrievaled_class == query_class:
            if query_img.find(retrievaled_name) != -1:
                if i == 0:
                    count += 1
                count_5 += 1
                break
recall = count / (ii+1)
recall_5 = count_5 / (ii+1)
print('recall@1 :', recall, '   recall@5 :', recall_5, 'div :', div/(ii+1))

