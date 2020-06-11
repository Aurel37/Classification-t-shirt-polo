from __future__ import absolute_import

import argparse
import os

from utils.reframe import zoomclass
from layers.model import net_model

import cv2

label = ['polo', 'shirt', 't_shirt']


def predict_label(img, net_model, label):
    img1 = cv2.resize(img, (80, 80))
    predict = net_model.predict(img1.reshape(1, 80, 80, 3))
    maxi = predict[0][0]
    curs = 0
    test = 0
    for i, pred in enumerate(predict[0]):
        test += pred
        if pred > maxi:
            maxi = pred
            curs = i
    return label[curs]


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--file', required=True)
    folder = vars(parse.parse_args())

    net_model.load_weights('weights/weights')

    path = [path for path in os.listdir(folder['file'] + '/')]
    imgs = [cv2.imread(folder['file'] + '/' + path) for path in os.listdir(folder['file'] + '/')]

    compt = 0
    for img in imgs:
        zooms = zoomclass('t_shirt', img)
        if zooms is not None:
            for zoom in zooms:
                lb = predict_label(zoom, net_model, label)
                print(path[compt] + ": ", lb)
                compt += 1
        else:
            lb = predict_label(zoom, net_model, label)
            print(path[compt] + ": ", lb)
            compt += 1
