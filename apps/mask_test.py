import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.segmentation import clear_border
import skimage.morphology as mp
import scipy.ndimage.morphology as sm
image = Image.open('/home/ubuntu/Downloads/test_0.png')
image = image.resize((512,512))
image.save('/home/ubuntu/Downloads/test_0.png')

image = Image.open('/home/ubuntu/Downloads/test_0.png')
model_cut = np.asarray(image)
b = (model_cut == 0)
c = b.astype(int)
c[c != 1] = 255
c[c == 1] = 0

cv2.imwrite('/home/ubuntu/Downloads/test_0_mask.png', c)