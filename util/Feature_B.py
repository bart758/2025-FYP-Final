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



def measure_streaks(image):
    """
    This function analyzes an image to detect streaks by converting it to grayscale, applying adaptive thresholding,
    and finding contours. It calculates lesion irregularity using the border perimeter and lesion area, returning
    a metric for streak irregularity.
    """
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
