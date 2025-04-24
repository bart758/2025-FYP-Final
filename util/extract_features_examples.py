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


def measure_pigment_network(image):

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(lab_image)

    enhanced_l_channel = cv2.equalizeHist(l_channel)
    _, binary_mask = cv2.threshold(enhanced_l_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    total_pixels = np.prod(binary_mask.shape[:2])
    pigment_pixels = np.count_nonzero(binary_mask)
    coverage_percentage = (pigment_pixels / total_pixels) * 100

    return coverage_percentage


def measure_blue_veil(image):
    
    height, width, _ = image.shape
    count = 0

    for y in range(height):
        for x in range(width):
            b, g, r = image[y, x]

            if b > 60 and (r - 46 < g) and (g < r + 15):
                count += 1

    return count


def measure_vascular(image):
    
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


def measure_globules(image):
    
    image_gray = rgb2gray(image)
    inverted_image = 1 - image_gray

    blobs_doh = blob_log(inverted_image, min_sigma=1, max_sigma=4, num_sigma=50, threshold=.05)
    blobs_doh[:, 2] = blobs_doh[:, 2] * sqrt(2)
    blob_amount = len(blobs_doh)

    return blob_amount


def measure_streaks(image):
   
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lesion_area = cv2.contourArea(contours[0])
    border_perimeter = cv2.arcLength(contours[0], True)
    if lesion_area == 0:
        irregularity = 0
    else:
        irregularity = (border_perimeter ** 2) / (4 * np.pi * lesion_area)

    return irregularity


def measure_irregular_pigmentation(image):
    
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


def measure_regression(image):

    """To measure the number of pixels in the input image that fall within a specific light gray to white
    color range in the HSV color space. This can be useful for quantifying certain regions in an image
    based on color, such as detecting highlights, surfaces, or regression areas that are light in color."""
   
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_color = np.array([0, 0, 150])
    upper_color = np.array([180, 30, 255])
    mask = cv2.inRange(hsv_img, lower_color, upper_color)
    num_pixels = cv2.countNonZero(mask)

    return num_pixels




def get_compactness(mask):

    """ To calculate the compactness of a binary region in an image, which is a measure of how closely packed
    or circular the shape is. A value closer to 1 suggests a shape that is more circular, while higher values
    indicate more irregular or elongated shapes."""

    # mask = color.rgb2gray(mask)
    area = np.sum(mask)

    struct_el = morphology.disk(3)
    mask_eroded = morphology.binary_erosion(mask, struct_el)
    perimeter = np.sum(mask - mask_eroded)

    return perimeter**2 / (4 * np.pi * area)

def get_asymmetry(mask):

    """To quantify the asymmetry of a binary shape. The mask is rotated multiple times (in 30° increments), and for each
    rotation, the function compares the shape with its horizontally flipped version using a logical XOR operation.
    The higher the score, the more asymmetric the shape is, which can be useful in tasks like medical image analysis
    (e.g., detecting irregular lesions)."""

    # mask = color.rgb2gray(mask)
    scores = []
    for _ in range(6):
        segment = crop(mask)
        (np.sum(segment))
        scores.append(np.sum(np.logical_xor(segment, np.flip(segment))) / (np.sum(segment)))
        mask = rotate(mask, 30)
    return sum(scores) / len(scores)

def get_multicolor_rate(im, mask, n):

    """To quantify the degree of color diversity within a specified region of an image. It resizes the image and mask
    for efficiency, extracts only the pixels within the mask, clusters them into n color groups, and computes the maximum 
    Euclidean distance between the most prominent color centers. Useful in image analysis tasks like skin lesion detection,
    artistic analysis, or visual complexity assessment."""

    # mask = color.rgb2gray(mask)
    im = resize(im, (im.shape[0] // 4, im.shape[1] // 4), anti_aliasing=True)
    mask = resize(
        mask, (mask.shape[0] // 4, mask.shape[1] // 4), anti_aliasing=True
    )
    im2 = im.copy()
    im2[mask == 0] = 0

    columns = im.shape[0]
    rows = im.shape[1]
    col_list = []
    for i in range(columns):
        for j in range(rows):
            if mask[i][j] != 0:
                col_list.append(im2[i][j] * 256)

    if len(col_list) == 0:
        return ""

    cluster = KMeans(n_clusters=n, n_init=10).fit(col_list)
    com_col_list = get_com_col(cluster, cluster.cluster_centers_)

    dist_list = []
    m = len(com_col_list)

    if m <= 1:
        return ""

    for i in range(0, m - 1):
        j = i + 1
        col_1 = com_col_list[i]
        col_2 = com_col_list[j]
        dist_list.append(
            np.sqrt(
                (col_1[0] - col_2[0]) ** 2
                + (col_1[1] - col_2[1]) ** 2
                + (col_1[2] - col_2[2]) ** 2
            )
        )
    return np.max(dist_list)

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

def crop(mask):
        
    """To crop a binary mask image to a centered region that tightly surrounds the object. The crop is horizontally
    symmetric around the midpoint and vertically tight around the object, which helps in standardizing input regions
    for further analysis like asymmetry or shape metrics."""

    mid = find_midpoint_v4(mask)
    y_nonzero, x_nonzero = np.nonzero(mask)
    y_lims = [np.min(y_nonzero), np.max(y_nonzero)]
    x_lims = np.array([np.min(x_nonzero), np.max(x_nonzero)])
    x_dist = max(np.abs(x_lims - mid))
    x_lims = [mid - x_dist, mid+x_dist]
    return mask[y_lims[0]:y_lims[1], x_lims[0]:x_lims[1]]

"""Finds the horizontal midpoint of an image in terms of pixel intensity distribution."""
def find_midpoint_v4(mask):
        summed = np.sum(mask, axis=0)
        half_sum = np.sum(summed) / 2
        for i, n in enumerate(np.add.accumulate(summed)):
            if n > half_sum:
                return i
  
"""Cuts empty / excess borders. Isolates the area of interest. Removes all borders 
(rows/columns) that sum up to 0. Basically making a rectangle around the area of interes
and cutting it."""
def cut_mask(mask):
    
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

    cut_mask_ = mask[row_min:row_max+1, col_min:col_max+1]

    return cut_mask_

"""Almost the same as the previous function. This time you pass an image and a mask.
 The function masks the active columns / rows based on the mask and then crops the 
 image based on that. So the returned image will be a rectangle just zoomed in on some 
 part based on the mask."""

def cut_im_by_mask(image, mask):
    

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

"""Literally just returns the geometrical midpoint of the image in the format [row col]"""
def find_midpoint_v1(image):
    
    row_mid = image.shape[0] / 2
    col_mid = image.shape[1] / 2
    return row_mid, col_mid

"""Measures the asymmetry of an image. The function splits the image into 4 sections 
based on the midpoint. Not like a coordinate plane with 4 quadrants. The parts are not 
unique. It splits it into upper lower halves and right left halves. Then flips lower 
and right halves. Then it uses xor to compare the left with flipped right and upper 
with flipped lower. Creating arrays which have values of 1 where compared halves are 
not the same. Then it calculates the actual score by summing all the not symmetrical 
pixels and dividing their sum by 2 times the sum of all the pixels of the original mask. 
(the denominator will always be larger than the numerator )
"""

def asymmetry(mask):
    

    row_mid, col_mid = find_midpoint_v1(mask)

    upper_half = mask[:ceil(row_mid), :]
    lower_half = mask[floor(row_mid):, :]
    left_half = mask[:, :ceil(col_mid)]
    right_half = mask[:, floor(col_mid):]

    flipped_lower = np.flip(lower_half, axis=0)
    flipped_right = np.flip(right_half, axis=1)

    hori_xor_area = np.logical_xor(upper_half, flipped_lower)
    vert_xor_area = np.logical_xor(left_half, flipped_right)

    total_pxls = np.sum(mask)
    hori_asymmetry_pxls = np.sum(hori_xor_area)
    vert_asymmetry_pxls = np.sum(vert_xor_area)

    asymmetry_score = (hori_asymmetry_pxls + vert_asymmetry_pxls) / (total_pxls * 2)

    return round(asymmetry_score, 4)

"""Measures the rotational asymmetry. The integer in the input is the amount of 
rotations within 90 degrees. It will always rotate the image to 90 degrees in the end,
 but the input n determines the amount of steps in which it will do it. Basically, 
 using the two previous functions, this one calculates asymmetry for each rotation 
 then returns a dictionary"""

def rotation_asymmetry(mask, n: int):

    asymmetry_scores = {}

    for i in range(n):

        degrees = 90 * i / n

        rotated_mask = rotate(mask, degrees)
        cutted_mask = cut_mask(rotated_mask)

        asymmetry_scores[degrees] = asymmetry(cutted_mask)

    return asymmetry_scores

def mean_asymmetry(mask, rotations = 30):
    
    asymmetry_scores = rotation_asymmetry(mask, rotations)
    mean_score = sum(asymmetry_scores.values()) / len(asymmetry_scores)

    return mean_score          

def best_asymmetry(mask, rotations = 30):
    
    asymmetry_scores = rotation_asymmetry(mask, rotations)
    best_score = min(asymmetry_scores.values())

    return best_score

def worst_asymmetry(mask, rotations = 30):
    
    asymmetry_scores = rotation_asymmetry(mask, rotations)
    worst_score = max(asymmetry_scores.values())

    return worst_score  

def slic_segmentation(image, mask, n_segments = 50, compactness = 0.1):
    
    slic_segments = slic(image,
                    n_segments = n_segments,
                    compactness = compactness,
                    sigma = 1,
                    mask = mask,
                    start_label = 1,
                    channel_axis = 2)
    
    return slic_segments

def get_rgb_means(image, slic_segments):
    
    max_segment_id = np.unique(slic_segments)[-1]

    rgb_means = []
    for i in range(1, max_segment_id + 1):

        segment = image.copy()
        segment[slic_segments != i] = -1

        rgb_mean = np.mean(segment, axis = (0, 1), where = (segment != -1))
        
        rgb_means.append(rgb_mean) 
        
    return rgb_means

def get_hsv_means(image, slic_segments):
    
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

def compactness_score(mask):
    """Input mask where white is 1 and black is 0, returns compactness score, by formula 1-(4*\pi*area)/(perimeter^2). Higher means more compact"""

    A = np.sum(mask) # calculates the area of the mask - 1 where white, 0 where black, sum is number of white pixels in mask - lession

    struct_el = morphology.disk(2) # binary mask of circle with radius 2

    mask_eroded = morphology.binary_erosion(mask, struct_el) # sets pixels (i,j) to the minimum over all pixels in the neighborhood covered by the binary mask centered at (i,j), so
    # in practice sets the border pixels to 0, shrinks the mask

    perimeter = mask - mask_eroded # gets difference of masks, which is the eroded part, border/perimeter

    l = np.sum(perimeter) # sum of perimeter is the lenght of the border

    compactness = (4*pi*A)/(l**2) # calculate compactness by formula

    score = round(1-compactness, 3)

    return score

def convexity_score(mask):
    """Input a mask. Returns convexity between 0 and 1, 1 being perfectly convex - circle or elipsis, 0 completely irregular."""

    coords = np.transpose(np.nonzero(mask)) # get a vertical array of all the coordinates of nonzero values in mask

    hull = ConvexHull(coords) # Calculate a ConvexHull - the smallest convex polygon that encompases all the points
    print(hull.area)

    lesion_area = np.count_nonzero(mask)

    convex_hull_area = hull.volume + hull.area # hull.volume is the area of the hull and hull.area is the perimiter of the hull because input points are 2D

    convexity = lesion_area / convex_hull_area # compare the area of the lession with the area of the convex hull

    return convexity # should be 0 to 1, 1 being perfectly convex - circle or elipsis, 0 completely irregular 

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