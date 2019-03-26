#-*- coding:utf-8 -*-

from scipy.misc import imresize
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2

def crop(img, name):
    cropped = img[70:338, 127:406]
    cv2.imwrite(name, imresize(cropped, (200,200)))

def main():
    pathes = glob('./samples1/*')
    for path in pathes:
        img = cv2.imread(path)
        val = round(0.011* int(path.split('/')[-1].split('.')[0]), 2)
        n = str(val)
        if len(str(val)) == 3:
           n += '0'
        name = './image1/'+n+'.png'
        crop(img, name)
    print('finished!')

if __name__ == '__main__':
   main()
