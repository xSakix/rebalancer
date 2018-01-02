'''
what we would need is a set of b&h results + rebalance results
for those find clusters and they should be diff for b&h and reb results
if its possible


currently i take results of graphs of stock data for which was rebalance
return vs b&h bigger then 50%
in those i try to identify clusters of similar data
and for such trained 'alg' i try to test it vs bb&h graph results

the point being: an algorithm trained on rebalance data should not
identify any b&h data as belonging to one of its clusters/classes
'''

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans, DBSCAN, MeanShift, AffinityPropagation, Birch, MiniBatchKMeans, SpectralClustering, \
    AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import numpy as np
import os
from sklearn.decomposition import PCA
import random


def load_data(data_dir):
    imgs = []
    names = []
    files = os.listdir(data_dir)
    # files = files[:10]
    for file in files:
        print('loading:' + file)
        img = mpimg.imread(data_dir + '/' + file)
        # img = Image.open(data_dir + '/' + file).convert('1')
        # img.save(data_dir + '/' + file)
        names.append(file)
        imgs.append(np.asarray(img))

    return names, imgs


def find_clusters(data, imgs, names):
    # pca = PCA(n_components=len(imgs))
    # data = pca.fit(data).transform(data)
    kmeans = KMeans(n_clusters=3).fit(data)
    labels = kmeans.labels_
    # kmeans = MiniBatchKMeans(n_clusters=3).fit(data)
    # labels = kmeans.labels_
    # spectral = SpectralClustering(n_clusters=3).fit(data)
    # labels = spectral.labels_
    # agglo = AgglomerativeClustering(n_clusters=3)
    # agglo.fit(data)
    # labels = agglo.labels_
    # labels = kmeans.predict(data).tolist()
    # db = DBSCAN().fit(data)
    # labels = db.labels_
    # shift = MeanShift().fit(data)
    # labels = shift.labels_
    # affinity = AffinityPropagation().fit(data)
    # labels = affinity.labels_
    # gaussian = GaussianMixture().fit(data)
    # labels = gaussian.predict(data)
    # birch = Birch().fit(data)
    # labels = birch.labels_
    # print(labels)
    predictions = list(zip(names, labels))


    return predictions,kmeans


def interpret_results(predictions, imgs):
    predictions.sort(key=lambda tup: tup[1])

    classes = []
    for (name, label) in predictions:
        if not classes.__contains__(label):
            classes.append(label)

    print('classes: ' + str(classes))

    num = 4
    counter = np.zeros(len(classes))

    index = 0
    for (name, label) in predictions:
        print('prediction =>' + str(name) + '->' + str(label))
        if (counter[int(label)] == num):
            continue
        counter[int(label)] += 1
        plt.subplot(len(classes), num, index + 1)
        plt.axis('off')
        plt.imshow(imgs[index], interpolation='nearest')
        plt.title(str(name) + '->' + str(label))
        index += 1
    plt.show()


def shufle_pred(predictions):
    indexes = list(range(0,len(predictions)))
    np.random.shuffle(indexes)
    new_pred = []
    for i in indexes:
        new_pred.append(predictions[i])

    return new_pred

names, imgs = load_data('train_stock')

data = np.array(imgs).reshape((len(imgs), -1))

predictions, kmeans = find_clusters(data, imgs, names)

interpret_results(predictions, imgs)

interpret_results(shufle_pred(predictions), imgs)
interpret_results(shufle_pred(predictions), imgs)
interpret_results(shufle_pred(predictions), imgs)

names, imgs = load_data('stock_bah_results')
data = np.array(imgs).reshape((len(imgs), -1))
labels = kmeans.predict(data)
predictions = list(zip(names, labels))
# interpret_results(predictions, imgs)
for (name, label) in predictions:
    print('prediction =>' + str(name) + '->' + str(label))
