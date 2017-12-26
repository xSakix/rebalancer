import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans, DBSCAN, MeanShift, AffinityPropagation, Birch
from sklearn.mixture import GaussianMixture
import numpy as np
import os
from sklearn.decomposition import PCA


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
    # kmeans = KMeans(init='k-means++', n_clusters=2, n_init=3).fit(data)
    # labels = kmeans.labels_
    #
    # labels = kmeans.predict(data).tolist()
    # db = DBSCAN().fit(data)
    # labels = db.labels_
    # shift = MeanShift().fit(data)
    # labels = shift.labels_
    # affinity = AffinityPropagation().fit(data)
    # labels = affinity.labels_
    # gaussian = GaussianMixture().fit(data)
    # labels = gaussian.predict(data)
    birch = Birch().fit(data)
    labels = birch.labels_
    print(labels)
    predictions = list(zip(names, labels))

    return predictions


def interpret_results(predictions, imgs):

    predictions.sort(key = lambda tup : tup[1])

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


names, imgs = load_data('train_stock')

data = np.array(imgs).reshape((len(imgs), -1))

predictions = find_clusters(data, imgs, names)

interpret_results(predictions, imgs)
