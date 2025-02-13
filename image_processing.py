import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageLoader(object):
    def __init__(self, image_path):
        self.image_data = cv2.imread(image_path)
        self.image_rgb = cv2.cvtColor(self.image_data, cv2.COLOR_BGR2RGB)
        self.image_hsv = cv2.cvtColor(self.image_data, cv2.COLOR_BGR2HSV)
        self.pixels_rgb = self.image_rgb.reshape(-1,3)
        self.pixels_hsv = self.image_hsv.reshape(-1,3)
        self.pixels_sorted_index_hue = np.argsort(self.pixels_hsv[:, 0])
        self.pixels_sorted_index_saturation = np.argsort(self.pixels_hsv[:, 1])
        self.pixels_sorted_index_value = np.argsort(self.pixels_hsv[:, 2])
    def to_display_shape(self, pixels):
        return pixels.reshape(self.image_data.shape)
# Load image and convert to RGB
image = ImageLoader('img1.png')
plt.imshow(image.to_display_shape(image.pixels_rgb[image.pixels_sorted_index_saturation]))
plt.show()


