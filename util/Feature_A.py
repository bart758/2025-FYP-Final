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



def measure_globules(image):
    """
    This function implements the method used in feature extraction algorithms called ‘blob-counting’ which in 
    turn means that it counts regions in an image that differ significantly in properties (like intensity,
    color, texture) compared to their surrounding area. They are typically connected regions that can be detected
    using algorithms like the Laplacian of Gaussian (LoG) or Difference of Gaussian (DoG) methods.
    This function implements LoG. It returns the number of blobs, in our context, modules found in the image.
    """
    
    image_gray = rgb2gray(image)
    inverted_image = 1 - image_gray

    blobs_doh = blob_log(inverted_image, min_sigma=1, max_sigma=4, num_sigma=50, threshold=.05)
    blobs_doh[:, 2] = blobs_doh[:, 2] * sqrt(2)
    blob_amount = len(blobs_doh)

    return blob_amount




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





def find_midpoint_v4(mask):
    """Finds the horizontal midpoint of an image in terms of pixel intensity distribution."""
    summed = np.sum(mask, axis=0)
    half_sum = np.sum(summed) / 2
    for i, n in enumerate(np.add.accumulate(summed)):
        if n > half_sum:
            return i
            




def find_midpoint_v1(image):
    """Literally just returns the geometrical midpoint of the image in the format [row col]"""
    row_mid = image.shape[0] / 2
    col_mid = image.shape[1] / 2
    return row_mid, col_mid




def asymmetry(mask):
    """Measures the asymmetry of an image. The function splits the image into 4 sections 
    based on the midpoint. Not like a coordinate plane with 4 quadrants. The parts are not 
    unique. It splits it into upper lower halves and right left halves. Then flips lower 
    and right halves. Then it uses xor to compare the left with flipped right and upper 
    with flipped lower. Creating arrays which have values of 1 where compared halves are 
    not the same. Then it calculates the actual score by summing all the not symmetrical 
    pixels and dividing their sum by 2 times the sum of all the pixels of the original mask. 
    (the denominator will always be larger than the numerator )
    """
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

    return asymmetry_score






def rotation_asymmetry(mask, n: int):
    """Measures the rotational asymmetry. The integer in the input is the amount of 
    rotations within 90 degrees. It will always rotate the image to 90 degrees in the end,
    but the input n determines the amount of steps in which it will do it. Basically, 
    using the two previous functions, this one calculates asymmetry for each rotation 
    then returns a dictionary"""
    asymmetry_scores = {}

    for i in range(n):

        degrees = 90 * i / n

        rotated_mask = rotate(mask, degrees)
        cutted_mask = cut_mask(rotated_mask)

        asymmetry_scores[degrees] = asymmetry(cutted_mask)

    return asymmetry_scores



def mean_asymmetry(mask, rotations = 30):

    ''' This feature rotates the image for the amount of rotations by given the input and quantifies how symmetric the mask is at each rotation.
     It them averages the symmetry scores, returning a single float between 0 - indicating full symmetry, to 1 - indicating no symmetry at all'''
    
    asymmetry_scores = rotation_asymmetry(mask, rotations)
    mean_score = sum(asymmetry_scores.values()) / len(asymmetry_scores)

    return mean_score



def best_asymmetry(mask, rotations = 30):

    '''This feature rotates the image for the amount of rotations given by the input and measures the asymmetry score for each rotation. 
    The function then returns the asymmetry score at the rotation in which the asymmetry score is at a minimum (closest to 0). '''
    
    asymmetry_scores = rotation_asymmetry(mask, rotations)
    best_score = min(asymmetry_scores.values())

    return best_score




def worst_asymmetry(mask, rotations = 30):
    
    ''' This feature rotates the image for the amount of rotations given by the input and measures the asymmetry score for each rotation. 
    The function then returns the asymmetry score at the rotation in which the asymmetry score is at a maximum (closest to 1). '''
    asymmetry_scores = rotation_asymmetry(mask, rotations)
    worst_score = max(asymmetry_scores.values())

    return worst_score 


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




# This function is from Feature_B, but we need it here for another function

def cut_mask(mask):
    """Cuts empty / excess borders. Isolates the area of interest. Removes all borders 
    (rows/columns) that sum up to 0. Basically making a rectangle around the area of interes
    and cutting it."""
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