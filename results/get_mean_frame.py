import os

import cv2
import numpy as np
from tqdm import tqdm

root = '/root/VD-Diff-main/experiments'

dirs = os.listdir(root + '/GOPRO-test-O/visualization/GOPRO/iter_0/')
for d in dirs:
    images = os.listdir(root + '/GOPRO-test-O/visualization/GOPRO/iter_0/' + d)
    for img in tqdm(images, dynamic_ncols=True):
        img1 = cv2.imread(root + '/GOPRO-test-O/visualization/GOPRO/iter_0/' + d + '/' + img[0:4] + '_GOPRO-test-O.png')
        img2 = cv2.imread(root + '/GOPRO-test-90/visualization/GOPRO/iter_0/' + d + '/' + img[0:4] + '_GOPRO-test-90.png')
        img3 = cv2.imread(root + '/GOPRO-test-180/visualization/GOPRO/iter_0/' + d + '/' + img[0:4] + '_GOPRO-test-180.png')
        img4 = cv2.imread(root + '/GOPRO-test-270/visualization/GOPRO/iter_0/' + d + '/' + img[0:4] + '_GOPRO-test-270.png')
        img5 = cv2.imread(root + '/GOPRO-test-V/visualization/GOPRO/iter_0/' + d + '/' + img[0:4] + '_GOPRO-test-V.png')
        img6 = cv2.imread(root + '/GOPRO-test-H/visualization/GOPRO/iter_0/' + d + '/' + img[0:4] + '_GOPRO-test-H.png')
        img7 = cv2.imread(root + '/GOPRO-test-T/visualization/GOPRO/iter_0/' + d + '/' + img[0:4] + '_GOPRO-test-T.png')
        # print(img1.shape, img2.shape, img3.shape, img4.shape, img5.shape, img6.shape, img7.shape)

        imgs = np.stack([img1, img2, img3, img4, img5, img6, img7], axis=0)
        imgs = np.mean(imgs, axis=0)
        imgs = imgs.astype(np.uint8)

        save_path = root + '/GOPRO-test-mean/' + d
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cv2.imwrite(save_path + '/' + img, imgs)
