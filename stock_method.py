import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans

import numpy as np
import os

data_dir = 'stock_results'

imgs = []
names = []
files = os.listdir(data_dir)
files = files[:40]
for file in files:
    print('loading:' + file)
    img = mpimg.imread(data_dir + '/' + file)
    names.append(file)
    imgs.append(img)


data = np.array(imgs).reshape((len(imgs), -1))


kmeans = KMeans(init='k-means++', n_clusters=len(imgs), n_init=3)
kmeans.fit(data)

labels = kmeans.predict(data).tolist()

predictions = list(zip(names, labels))

num = int(len(imgs) / 4)+1
print(num)

index = 0
for index, (name, label) in enumerate(predictions):
    print('prediction =>' + str(name) + '->' + str(label))
    plt.subplot(num, 4, index + 1)
    plt.axis('off')
    plt.imshow(imgs[index], interpolation='nearest')
    plt.title(str(name) + '->' + str(label))

plt.show()