from pickletools import uint8

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
    def to_display_shape(self, pixels):
        return pixels.reshape(self.image_data.shape)


class ColorProfile(object):
    def __init__(self, color_chunk, image_data: ImageLoader):
        self.percentage =  len(color_chunk) * 100 / len(image_data.pixels_hsv)

        self.saturations = np.sort(color_chunk[:, 1])
        self.saturation_max = np.max(self.saturations)
        self.saturation_min = np.min(self.saturations)
        self.saturation_mean = np.mean(self.saturations)
        self.saturation_median = np.median(self.saturations)

        self.values = np.sort(color_chunk[:,2])
        self.value_max = np.max(self.values)
        self.value_min = np.min(self.values)
        self.value_mean = np.mean(self.values)
        self.value_median = np.median(self.values)


class ImageAnalyzer(object):
    def __init__(self, image_data: ImageLoader):
        self.image_data = image_data
        self.analyze_result = {}
        self.hue_list = []

    def color_analyzing(self):
        pixels_sorted_by_hue = self.image_data.pixels_hsv[np.argsort(self.image_data.pixels_hsv[:,0])]
        hue_chunk_start = 0
        for hue_chunk_end in range(len(pixels_sorted_by_hue)):
            if pixels_sorted_by_hue[hue_chunk_end, 0] != pixels_sorted_by_hue[hue_chunk_start, 0] :
                hue_chunk = pixels_sorted_by_hue[hue_chunk_start:hue_chunk_end]
                self.analyze_result.update({pixels_sorted_by_hue[hue_chunk_start, 0]:ColorProfile(hue_chunk, self.image_data)})
                hue_chunk_start = hue_chunk_end
        self.hue_list = list(self.analyze_result.keys())

    def get_palette(self, percentage=1):
        palette = {}
        for key in list(self.analyze_result.keys()):
            color = self.analyze_result[key]
            if color.percentage >= percentage:
                hsv_color = [key, color.saturation_median, color.value_median]
                rgb_color = cv2.cvtColor(np.array([[hsv_color]], dtype=np.uint8), cv2.COLOR_HSV2RGB)
                palette.update({color.percentage: rgb_color[0,0]})
        return palette

# Load image and convert to RGB
image = ImageLoader('img1.png')
analyze = ImageAnalyzer(image)

analyze.color_analyzing()

palette = analyze.get_palette(percentage=0.05)
percentages = np.sort(list(palette.keys()))


for p in percentages:
    print(palette[p])
    rgb = palette[p]
    print(rgb)
    block = np.empty((10,10,3), dtype=np.uint8)
    block[:,:] = rgb
    plt.imshow(block)
    plt.show()




