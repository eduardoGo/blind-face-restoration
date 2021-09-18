import cv2
import numpy as np
import os

## Loading Textures
path_textures = "C:\\Users\\eduardo\\Documents\\9 periodo\\image processing\\projeto\\ab2-final\\another\\blind-face-restoration\\gfpgan\\data\\textures"
path_output_textures = "C:\\Users\\eduardo\\Documents\\9 periodo\\image processing\\projeto\\ab2-final\\another\\blind-face-restoration\\gfpgan\\data\\textures_result"
textures = []
for filename in os.listdir(path_textures):

    img = cv2.imread(os.path.join(path_textures,filename))

    tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,alpha = cv2.threshold(tmp,128,np.random.randint(low=230,high=255),cv2.THRESH_BINARY)
    b, g, r = cv2.split(img)
    rgba = [b,g,r, alpha]
    dst = cv2.merge(rgba,4)
    # Save bgra to PNG file
    cv2.imwrite(os.path.join(path_output_textures,filename.replace('jpg','png')), dst)