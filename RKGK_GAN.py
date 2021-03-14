import gan
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import cv2

imglst = []
# imglst = np.array([mpimg.imread('./Input/%s' % f) for f in os.listdir('./Input/')])
for f in os.listdir('./InputReverseDilate/'):
    img = cv2.imread('./InputReverseDilate/%s' % f, cv2.IMREAD_GRAYSCALE)
    # img.resize((56 ,56))
    # print(img)
    cv2.imwrite('sample.png', img)

    imglst.append(img)
imglst = np.array(imglst)
print(imglst.shape)

cv2.waitKey (0)  
Gan = gan.Gan(width=56, height=56, channels=1)
Gan.train(imglst, batch=32, save_interval=50)

# show
# plt.imshow(img)  # 顯示圖片
# plt.axis('off')  # 不顯示座標軸
# plt.show()