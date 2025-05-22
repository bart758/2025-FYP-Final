import cv2
import numpy as np
from math import sqrt, floor, ceil, nan, pi
from skimage import color, exposure
from skimage.color import rgb2gray
from skimage.feature import blob_log
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.transform import resize
from skimage.transform import rotate
from skimage import morphology
from sklearn.cluster import KMeans
from skimage.segmentation import slic
from skimage.color import rgb2hsv
from scipy.stats import circmean, circvar, circstd
from statistics import variance, stdev
from scipy.spatial import ConvexHull
from itertools import combinations


def get_multicolor_rate(image: np.ndarray, mask: np.ndarray, n: int = 4) -> float:
    """
    Measure color variation in the masked lesion area using clustering.
    Returns normalized color variation in [0, 1] range.

    Parameters
    ----------
    image: np.ndarray
        Color image of the lesion.
    mask: np.ndarray
        Mask of the lesion image.
    n: int, optional
        Number of color groups. (Conservative: 3, Sensitive: 5), by default 4.

    Returns
    -------
    float
        Normalized maximum distance between colors in lesion.
    """

    image = resize(image, (image.shape[0] // 4, image.shape[1] // 4), anti_aliasing=True)
    mask = resize(mask, (mask.shape[0] // 4, mask.shape[1] // 4), anti_aliasing=True)

    # Remove background
    im2 = image.copy()
    try:
        im2[mask == 0] = 0
    except IndexError:
        f"Image sizes don't match up in image {image}."

    col_list = [im2[i, j] for i in range(mask.shape[0]) for j in range(mask.shape[1]) if mask[i, j] != 0]

    if len(col_list) < 2:
        return 0.0

    # KMeans clustering
    cluster = KMeans(n_clusters=n, n_init=10, random_state=0).fit(col_list)
    centers = cluster.cluster_centers_

    # Calculate max pairwise distance
    max_dist = max(np.linalg.norm(np.array(c1) - np.array(c2)) for c1, c2 in combinations(centers, 2))

    # Normalize RGB values to [0, 1]
    normalized = round(max_dist / np.sqrt(3), 4)

    return normalized

def measure_pigment_network(image):
    """The function analyzes an image to calculate the percentage of its area covered by pigment-like
regions. It does this by converting the image to LAB color space with l_channel, _, _ = cv2.split(lab_image),
enhancing the lightness channel with enhanced_l_channel = cv2.equalizeHist(l_channel), segmenting the image using
a threshold (binary + otsu threshold), and then computing the ratio of the segmented "pigment" pixels to the
total pixels. The result is returned as a percentage. """
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(lab_image)

    enhanced_l_channel = cv2.equalizeHist(l_channel)
    _, binary_mask = cv2.threshold(enhanced_l_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    total_pixels = np.prod(binary_mask.shape[:2])
    pigment_pixels = np.count_nonzero(binary_mask)
    coverage_percentage = (pigment_pixels / total_pixels) * 100

    return coverage_percentage



def original_measure_blue_veil(image):
    """
     This function takes in an image and iterates through all pixels, counting all that have blue as their 
     dominant color, while also making sure that red and green values are similar to each other:
    if b > 60 and (r - 46 < g) and (g < r + 15) b,r and g signifying blue, red and green values.
    """
    height, width, _ = image.shape
    count = 0

    for y in range(height):
        for x in range(width):
            b, g, r = image[y, x]

            if b > 60 and (r - 46 < g) and (g < r + 15):
                count += 1

    return count

def measure_blue_veil(image):
    """
    This function takes in an image and iterates through all pixels, counting all that have blue as their
    dominant color, while also making sure that red and green values are similar to each other:
    if b > 60 and (r - 46 < g) and (g < r + 15) b,r and g signifying blue, red and green values.
    """
    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
    count = np.sum(np.where((b > 60) & (r - 46 < g) & (g < r + 15), 1, 0))

    return count


def measure_vascular(image):
    """
     This function takes in an image and enhances the red channel, afterwards converting it to HSV format, 
     HSV separates color information (hue) from intensity (saturation and value), which is useful for color-based 
     segmentation. Then it isolates ranges of the color red in a mask, returning the number of red pixels in the image.
    """
    red_channel = image[:, :, 0]
    enhanced_red_channel = exposure.adjust_gamma(red_channel, gamma=1)
    enhanced_image = image.copy()
    enhanced_image[:, :, 0] = enhanced_red_channel
    hsv_img = color.rgb2hsv(enhanced_image)

    lower_red1 = np.array([0, 40/100, 00/100])
    upper_red1 = np.array([25/360, 1, 1])
    mask1 = np.logical_and(np.all(hsv_img >= lower_red1, axis=-1), np.all(hsv_img <= upper_red1, axis=-1))

    lower_red2 = np.array([330/360, 40/100, 00/100])  
    upper_red2 = np.array([1, 1, 1]) 
    mask2 = np.logical_and(np.all(hsv_img >= lower_red2, axis=-1), np.all(hsv_img <= upper_red2, axis=-1))

    mask = np.logical_or(mask1, mask2)

    return np.sum(mask)




def measure_irregular_pigmentation(image):
    """
    This function identifies irregular pigmentation by converting an image to grayscale, applying
    Otsu's thresholding, and segmenting the regions. It calculates the percentage of irregular pixels
    ompared to the total image area, providing a measure of pigmentation irregularity.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold = threshold_otsu(gray)
    binary = gray > threshold
    labeled_image = label(binary)

    min_rows, min_cols, max_rows, max_cols = [], [], [], []

    for region in regionprops(labeled_image):
        area = region.area
        perimeter = region.perimeter

        if perimeter == 0:
            continue

        circularity = 4 * np.pi * (area / (perimeter ** 2))

        if circularity < 0.6:
            min_row, min_col, max_row, max_col = region.bbox
            min_rows.append(min_row)
            min_cols.append(min_col)
            max_rows.append(max_row)
            max_cols.append(max_col)

    _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    total_pixels = np.prod(binary_mask.shape[:2])
    irregular_pixels = np.count_nonzero(binary_mask)
    coverage_percentage = (irregular_pixels / total_pixels) * 100

    return coverage_percentage



def get_com_col(cluster, centroids):

    """To analyze the output of KMeans clustering on image pixels and identify the most significant colors
    based on their frequency. It builds a histogram of color cluster frequencies, normalizes it, and collects
    colors that occur in more than 8% of the pixels. Optionally, it also creates a color bar image (rect) to
    visually represent the distribution of dominant colors—useful for debugging or visualization, though it's not returned."""

    com_col_list = []
    labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    (hist, _) = np.histogram(cluster.labels_, bins=labels)
    hist = hist.astype("float")
    hist /= hist.sum()

    rect = np.zeros((50, 300, 3), dtype=np.uint8)
    colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)], key= lambda x:x[0])
    start = 0
    for percent, color in colors:
        if percent > 0.08:
            com_col_list.append(color)
        end = start + (percent * 300)
        cv2.rectangle(
            rect,
            (int(start), 0),
            (int(end), 50),
            color.astype("uint8").tolist(),
            -1,
        )
        start = end
    return com_col_list




def slic_segmentation(image, mask, n_segments = 50, compactness = 0.1):
        
    '''The SLIC function divides the grayscale mask into n different segments based on the input, each segment treated as a superpixel instead of treating each
    single pixel individually, which reduces complexity. The function adjusts the shape and color balance of each superpixel based on the compactness input (
    the higher the value, the more compact and less flexible each superpixel shape is). Each superpixel is then given an integer label and the function returns 
    a Numpy array that acts as a map for the superpixels, each value in the array being the label of that superpixel. This allows analysis of larger regions of the 
    image separately, instead of single pixels.'''

    slic_segments = slic(image,
                    n_segments = n_segments,
                    compactness = compactness,
                    sigma = 1,
                    mask = mask,
                    start_label = 1,
                    channel_axis = 2)

    return slic_segments



def get_rgb_means(image, slic_segments):
    
    '''This feature takes an array representing a RGB image and a NumPy array with the labels for each superpixel (the output of the slic_segmentation feature).
     For each superpixel, the feature calculates the mean of the red, green, and blue pixels and adds these values to to inner array representing that superpixel.
    The values for each color range between 0 to 255, where 0 is black and 255 is max intensity (brightness) of that color.'''
      
    max_segment_id = np.unique(slic_segments)[-1]

    rgb_means = []
    for i in range(1, max_segment_id + 1):

        segment = image.copy()
        segment[slic_segments != i] = -1

        rgb_mean = np.mean(segment, axis = (0, 1), where = (segment != -1))
        
        rgb_means.append(rgb_mean) 
        
    return rgb_means



def get_hsv_means(image, slic_segments):
    
    '''This feature measures the mean hue, saturation, and value (brightness) of each region in the image (each superpixel). It creates a list of arrays 
    and inserts the HSV means into the inner array which represents the superpixel, and then returns that list.'''

    hsv_image = rgb2hsv(image)

    max_segment_id = np.unique(slic_segments)[-1]

    hsv_means = []
    for i in range(1, max_segment_id + 1):

        segment = hsv_image.copy()
        segment[slic_segments != i] = nan

        hue_mean = circmean(segment[:, :, 0], high=1, low=0, nan_policy='omit') 
        sat_mean = np.mean(segment[:, :, 1], where = (slic_segments == i))  
        val_mean = np.mean(segment[:, :, 2], where = (slic_segments == i)) 

        hsv_mean = np.asarray([hue_mean, sat_mean, val_mean])

        hsv_means.append(hsv_mean)
        
    return hsv_means





def rgb_var(image, slic_segments):
    """Takes rgb image data and slic_segments, which is the same size as the image and for each pixel has a value for which segmnet/superpixel 
    the pixel belongs to by similarity. And returns tuple (red_var, green_var, blue_var), variance."""

    if len(np.unique(slic_segments)) == 2: # if only two superpixels, variance in all colors are zero
        return 0, 0, 0

    rgb_means = get_rgb_means(image, slic_segments) # else gets rgb means of superpixels
    n = len(rgb_means) 

    red = []
    green = []
    blue = []
    for rgb_mean in rgb_means: # separates by color channel
        red.append(rgb_mean[0])
        green.append(rgb_mean[1])
        blue.append(rgb_mean[2])

    red_var = variance(red, sum(red)/n) # and gets sample variance for each channel, using means from each superpixel
    green_var = variance(green, sum(green)/n)
    blue_var = variance(blue, sum(blue)/n)

    return red_var, green_var, blue_var




def hsv_var(image, slic_segments):
    """Takes hsv image data and slic_segments, which is the same size as the image and for each pixel has a value for which segmnet/superpixel 
    the pixel belongs to by similarity. And returns tuple (hue_var, sat_var, val_var), variance."""

    if len(np.unique(slic_segments)) == 2: # if only two superpixels, variance in all colors are zero
        return 0, 0, 0

    hsv_means = get_hsv_means(image, slic_segments) # else gets hsv means of superpixels
    n = len(hsv_means) 

    hue = []
    sat = []
    val = []
    for hsv_mean in hsv_means: # separates by channel
        hue.append(hsv_mean[0])
        sat.append(hsv_mean[1])
        val.append(hsv_mean[2])

    hue_var = circvar(hue, high=1, low=0) # and gets sample variance for each channel, using means from each superpixel
    sat_var = variance(sat, sum(sat)/n)
    val_var = variance(val, sum(val)/n)

    return hue_var, sat_var, val_var



def color_dominance(image, mask, clusters = 5, include_ratios = False):
    """Input RGB image, mask and otional boolean include_rations. Returns HSV for n dominant colors in n clusters, 
    if include_ratios is True returns the ratio of pixels in each cluster, sorted descending by ratio."""
    
    cut_im = cut_im_by_mask(image, mask) # crops image to only show masked area
    hsv_im = rgb2hsv(cut_im) # rgb to hsv
    flat_im = np.reshape(hsv_im, (-1, 3)) # flattens image from 2D array, where each [i, j] has h, s, v channels to 
    # 1D array with the pixels in reading order of the original array and each [i] has h, s, v channels

    k_means = KMeans(n_clusters=clusters, n_init=10, random_state=0) # defines KMeans settings to be applied to image -> n_clusters = number of clusters to separate into, 
    #  n_init = number of times to run kmeans with different centroids as starting point, random_state = int means deterministic centroid initialization
    k_means.fit(flat_im) # separates the image into 5 clusters 

    dom_colors = np.array(k_means.cluster_centers_, dtype='float32') # gets the color of centroids, which are dominant in the image

    if include_ratios: # if we include ratios, we add the ratio of pixels in each group, sorted descending by ratio

        counts = np.unique(k_means.labels_, return_counts=True)[1] 
        ratios = counts / flat_im.shape[0] 

        r_and_c = zip(ratios, dom_colors) 
        r_and_c = sorted(r_and_c, key=lambda x: x[0],reverse=True) 

        return r_and_c
    
    return dom_colors



def get_relative_rgb_means(image, slic_segments):
    """Input RGB image and slic segments array. Returns normalized proportional rgb in lesion, difference in rgb between lesion and skin"""

    max_segment_id = np.unique(slic_segments)[-1] # get the number of segments in the image

    rgb_means = []
    for i in range(0, max_segment_id + 1): # for the i segment id in the number of segments

        segment = np.astype(image.copy(), "int8") # get a copy of the image
        segment[slic_segments != i] = -1 # and set all the non i segments to -1

        rgb_mean = np.mean(segment, axis = (0, 1), where = (segment != -1)) # get the mean r, g, b values along the i-th segment
        
        rgb_means.append(rgb_mean) # append the means of r, g, b values, rgb_means i a list of array[mean(r), mean(g), mean(b)]

    rgb_means_lesion = np.mean(rgb_means[1:],axis=0) # get mean of all the rgb values in the lesion
    rgb_means_skin = np.mean(rgb_means[0]) # get mean of rgb values in outside of lesion because of the way that slic works segment 0 is always the skin, because it starts at the corner

    F1, F2, F3 = rgb_means_lesion/sum(rgb_means_lesion) # compute the normalized proportions of r, g, b in the lesion
    F10, F11, F12 = rgb_means_lesion - rgb_means_skin # compute the difference between the rgb in skin and lesion
        
    return F1, F2, F3, F10, F11, F12 # F1, F10 - red; F2, F11 - green; F3, F12 - blue


# This function is from Feature_B, but we need it here for another function


def cut_im_by_mask(image, mask):
    """Almost the same as the previous function. This time you pass an image and a mask.
     The function masks the active columns / rows based on the mask and then crops the
     image based on that. So the returned image will be a rectangle just zoomed in on some
     part based on the mask."""

    col_sums = np.sum(mask, axis=0)
    row_sums = np.sum(mask, axis=1)

    active_cols = []
    for index, col_sum in enumerate(col_sums):
        if col_sum != 0:
            active_cols.append(index)

    active_rows = []
    for index, row_sum in enumerate(row_sums):
        if row_sum != 0:
            active_rows.append(index)

    col_min = active_cols[0]
    col_max = active_cols[-1]
    row_min = active_rows[0]
    row_max = active_rows[-1]

    cut_image = image[row_min:row_max+1, col_min:col_max+1]

    return cut_image