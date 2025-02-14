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
        #self.pixels_sorted_index_hue = np.argsort(self.pixels_hsv[:, 0])
        #self.pixels_sorted_index_saturation = np.argsort(self.pixels_hsv[:, 1])
        #self.pixels_sorted_index_value = np.argsort(self.pixels_hsv[:, 2])

    def to_display_shape(self, pixels):
        return pixels.reshape(self.image_data.shape)

class ImageAnalyzer(object):
    def __init__(self, image_data: ImageLoader):
        self.image_data = image_data

    def get_color_analyzing(self):
        pixels_sorted_by_hue = self.image_data.pixels_hsv[np.argsort(self.image_data.pixels_hsv[:,0])]
        hue_chunk_start = 0
        pixels_sorted = np.empty((0,3))
        for hue_chunk_end in range(len(pixels_sorted_by_hue)):
            if pixels_sorted_by_hue[hue_chunk_end, 0] != pixels_sorted_by_hue[hue_chunk_start, 0]:
                hue_chunk = pixels_sorted_by_hue[hue_chunk_start:hue_chunk_end]
                sorted_hue_chunk = hue_chunk[np.argsort(hue_chunk[:,1])]
                pixels_sorted = np.concatenate([pixels_sorted, sorted_hue_chunk], axis=0)
                hue_chunk_start = hue_chunk_end

        print(pixels_sorted)

# Load image and convert to RGB
image = ImageLoader('img1.png')
analyze = ImageAnalyzer(image)

analyze.get_color_analyzing()


# plt.imshow(image.to_display_shape(image.pixels_rgb[image.pixels_sorted_index_saturation]))
# plt.show()


