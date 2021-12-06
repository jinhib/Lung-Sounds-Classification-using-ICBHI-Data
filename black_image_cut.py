from keras.preprocessing import image
import os
import matplotlib.pyplot as plt, img
import numpy as np
import cv2

base_dir = 'F:/Experiment/2021.09.LungSound_2017ICBHI/0.data/original_mel'
for label in ['crackle', 'wheeze', 'normal', 'mixed']:
    path_dir = base_dir + '/' + label

# data_list = os.listdir(path_dir)
# data_name = plt.imread('104_1b1_Ll_sc_Litt3200_1.png')
# plt.imshow(data_name)
# plt.axis('off'), plt.xticks([]), plt.yticks([])
# plt.tight_layout()
# plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
# plt.savefig(fname='test.jpg', bbox_inches='tight', pad_inches=0, dpi=100)
# plt.close()

img = cv2.imread('104_1b1_Ll_sc_Litt3200_1.png')

# 변환 graky
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 임계값 조절
mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)[1]

# mask
mask = 255 - mask

# morphology 적용
# borderconstant 사용
kernel = np.ones((3,3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# anti-alias the mask
# blur alpha channel
mask = cv2.GaussianBlur(mask, (0,0), sigmaX=2, sigmaY=2, borderType = cv2.BORDER_DEFAULT)

# linear stretch so that 127.5 goes to 0, but 255 stays 255
mask = (2*(mask.astype(np.float32))-255.0).clip(0,255).astype(np.uint8)

# put mask into alpha channel
result = img.copy()
result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
result[:, :, 3] = mask

# 저장
cv2.imwrite('test1.jpg', result)