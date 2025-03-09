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




class ImageAnalyzer(object):
    #data type for store the information of color been extract from this image. Each single colorProfile object repreasent the information of colors with one certain hue value
    class ColorProfile(object):
        def __init__(self, color_chunk, image_data: ImageLoader):
            self.percentage = len(color_chunk) * 100 / len(image_data.pixels_hsv)

            self.saturation = np.sort(color_chunk[:, 1])
            self.saturation_max = np.max(self.saturation)
            self.saturation_min = np.min(self.saturation)
            self.saturation_mean = np.mean(self.saturation)
            self.saturation_median = np.median(self.saturation)

            self.values = np.sort(color_chunk[:, 2])
            self.value_max = np.max(self.values)
            self.value_min = np.min(self.values)
            self.value_mean = np.mean(self.values)
            self.value_median = np.median(self.values)

    def __init__(self, image_data: ImageLoader):
        self.image_data = image_data
        self.analyze_result = {}
        self.hue_list = []

    # extract all the color on the image and store them in list
    def color_analyzing(self):
        pixels_sorted_by_hue = self.image_data.pixels_hsv[np.argsort(self.image_data.pixels_hsv[:,0])]
        hue_chunk_start = 0
        for hue_chunk_end in range(len(pixels_sorted_by_hue)):
            if pixels_sorted_by_hue[hue_chunk_end, 0] != pixels_sorted_by_hue[hue_chunk_start, 0] :
                hue_chunk = pixels_sorted_by_hue[hue_chunk_start:hue_chunk_end]
                self.analyze_result.update({pixels_sorted_by_hue[hue_chunk_start, 0]:self.ColorProfile(hue_chunk, self.image_data)})
                hue_chunk_start = hue_chunk_end
        self.hue_list = list(self.analyze_result.keys())

    #get palette with some filter rules, only select the median value of each hue existed on the image
    def get_palette(self, percentage=0.01, saturation=0, brightness=0):
        palette = {}
        for key in list(self.analyze_result.keys()):
            color = self.analyze_result[key]
            if color.percentage >= percentage and color.saturation_median >= saturation and color.value_median >= brightness:
                hsv_color = [key, color.saturation_median, color.value_median]
                rgb_color = cv2.cvtColor(np.array([[hsv_color]], dtype=np.uint8), cv2.COLOR_HSV2RGB)
                palette.update({color.percentage: rgb_color[0,0]})
        return palette


# a useful hex code conversion approach able to been used indipendent from color analyzer
def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])




# for demo only
# Load image and convert to RGB
image = ImageLoader('img2.jpg')
analyze = ImageAnalyzer(image)

analyze.color_analyzing()

palette = analyze.get_palette(percentage=0.01, saturation=35, brightness=50)
percentages = np.sort(list(palette.keys()))

num_colors = len(percentages)  # Number of colors
cols = min(num_colors, 10)  # Max 5 colors per row
rows = (num_colors + cols - 1) // cols  # Compute number of rows needed

fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))  # Adjust figure size

# If only one row, axes may not be a 2D array, so we ensure it’s iterable
axes = np.array(axes).reshape(rows, cols)

for idx, p in enumerate(percentages):
    rgb = palette[p]  # Extract RGB color
    row, col = divmod(idx, cols)  # Calculate grid position

    # Create 10x10 block of the color
    block = np.full((10, 10, 3), rgb, dtype=np.uint8)

    # Plot the block in the corresponding subplot
    axes[row, col].imshow(block)
    axes[row, col].axis("off")  # Hide axis for a cleaner look
    axes[row, col].set_title(f"{p:.2f}% {rgb_to_hex(rgb)}")  # Optional: Show percentage

# Remove empty subplots if the last row isn't full
for i in range(idx + 1, rows * cols):
    fig.delaxes(axes.flatten()[i])

plt.tight_layout()
plt.show()



