import os
import sys

import cv2
from tqdm import tqdm

sys.path.append('/root/VD-Diff-main/')

from basicsr.metrics import calculate_psnr, calculate_ssim

restor_root = './GOPRO-16'
gt_root = '/root/GOPRO/test/gt'

psnr = 0.0
ssim = 0.0
count = 0
for video in sorted(os.listdir(restor_root)):
    video_gt_root = gt_root + '/' + video
    video_restor_root = restor_root + '/' + video
    img_gt_list = sorted(os.listdir(video_gt_root))
    # img_gt_list = img_gt_list[1:-1]
    # print(len(img_gt_list))
    img_restor_list = sorted(os.listdir(video_restor_root))
    list_id = 0
    for img in tqdm(img_restor_list):
        img_gt_root = video_gt_root + '/' + img_gt_list[list_id]
        img_restor_root = video_restor_root + '/' + img_restor_list[list_id]
        img_gt = cv2.imread(img_gt_root)[:, :, ::-1]
        img_restor = cv2.imread(img_restor_root)[:, :, ::-1]
        psnr_ = calculate_psnr(img_restor, img_gt, 0)
        ssim_ = calculate_ssim(img_restor, img_gt, 0)
        psnr += psnr_
        ssim += ssim_
        count += 1
        list_id += 1

print('PSNR:', psnr / count)
print('SSIM:', ssim / count)
