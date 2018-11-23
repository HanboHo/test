import PIL.Image as Image
import numpy as np
import os
import cv2

DIR1 = "/mnt/internal/99_Data/08_Mapillary/images"
DIR2 = "/home/chen/segmentationchallengesubmission/testing/eval_dgx2_mapillary_panoptic_nokl_02_lowerlr_01_newfusion_test_submission/prediction_test"

files1 = [file for file in os.listdir(DIR1) if os.path.isfile(
    os.path.join(DIR1, file))]
files2 = [file for file in os.listdir(DIR2) if os.path.isfile(
    os.path.join(DIR2, file))]

view_factor = 0.25

for i in files1:
    print(i)
    image = os.path.join(DIR1, i)
    label = os.path.join(DIR2, i.split('.')[0]) + '.png'
    image = cv2.imread(image)
    label = cv2.imread(label)
    image = cv2.resize(image, (0, 0), fx=view_factor, fy=view_factor)
    label = cv2.resize(label, (0, 0), fx=view_factor, fy=view_factor)
    alpha = 0.4
    beta = 1.0 - alpha
    final_img = cv2.addWeighted(image, beta, label, alpha, 0.0)
    cv2.imshow('Ori Image', final_img)
    while True:
        key = cv2.waitKey(0)
        if key == ord('q'):
            exit()
        elif key == ord('y'):
            break

print(0)
