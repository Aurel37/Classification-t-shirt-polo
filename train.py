from __future__ import absolute_import

import tensorflow as tf


from scraping import url_open
import numpy as np
import matplotlib.pyplot as plt
from utils.reframe import zoomclass
from layers.model import net_model
import cv2
import re

if __name__ == "__main__":
    net_model.summary()
    label = ['polo', 'shirt', 't_shirt']
    imgs_batch = []

    train_data = []
    label_train = []

    test_data = []
    label_test = []

    for i in label:
        with open('images/' + i + '_url.txt', 'r') as f:
            res = []
            for url in f.read().splitlines():
                tab = url_open(url)
                #print(tab, end='\r')
                if re.search("jumia", url) is not None:
                    if tab is not None:
                        res.append([tab, True])
                else:
                    if tab is not None:
                        res.append([tab, False])

        imgs_batch.append(res)

    for i in range(3):
        n = len(imgs_batch[i])
        r = np.random.permutation(n)
        for j in r[0:int(3*n/4)]:
            if imgs_batch[i][j][1]:
                img1 = cv2.resize(imgs_batch[i][j][0], (224, 224))
                train_data.append(img1.reshape(224, 224, 3))
                label_train.append(np.array(i))
            else:
                img_tre = zoomclass('t_shirt', imgs_batch[i][j][0])
                for img in img_tre:
                    img1 = cv2.resize(img, (224, 224))
                    train_data.append(img1.reshape(224, 224, 3))
                    label_train.append(np.array(i))
        for j in r[int(3*n/4):]:
            if imgs_batch[i][j][1]:
                img1 = cv2.resize(imgs_batch[i][j][0], (224, 224))
                test_data.append(img1.reshape(224, 224, 3))
                label_test.append(np.array(i))
            else:
                img_tre = zoomclass('t_shirt', imgs_batch[i][j][0])
                for img in img_tre:
                    img1 = cv2.resize(img, (224, 224))
                    test_data.append(img1.reshape(224, 224, 3))
                    label_test.append(np.array(i))
    train_data = np.array(train_data)
    label_train = np.array(label_train)
    test_data = np.array(test_data)
    label_test = np.array(label_test)

    net_model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    history = net_model.fit(train_data, label_train, epochs=10, validation_data=(test_data, label_test))
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()
    test_loss, test_acc = net_model.evaluate(test_data, label_test, verbose=2)
    print(test_acc)
    net_model.save_weights('weights/weights')
