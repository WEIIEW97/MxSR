import numpy as np
# import cv2

import re
import matplotlib.pyplot as plt

def read_pfm(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data*scale


def compute_confidence_map(disp_left, disp_right, I_left, I_right, max_disp=64, cost_clip=100):
    """
    Compute the confidence map based on disparity consistency and matching cost.
    
    Parameters:
    - disp_left: Disparity map of the left image.
    - disp_right: Disparity map of the right image.
    - I_left, I_right: Left and right stereo images.
    - max_disp: Maximum possible disparity.
    - cost_clip: Value to clip the cost before applying exponential to avoid overflow.
    
    Returns:
    - confidence_map: The computed confidence map.
    """
    
    height, width = disp_left.shape
    confidence_map = np.ones_like(disp_left, dtype=np.float32)
    
    # Left-to-right consistency check (disparity consistency)
    for y in range(height):
        for x in range(width):
            # Check if the disparity values are valid (not NaN or infinity)
            if not np.isnan(disp_left[y, x]) and not np.isinf(disp_left[y, x]):
                # Ensure the disparity is within bounds (no out-of-bounds indexing)
                disparity_offset = int(disp_left[y, x])
                if 0 <= x - disparity_offset < width:
                    # Compute disparity difference between the left and right image
                    disparity_diff = abs(disp_left[y, x] - disp_right[y, x - disparity_offset])
                    # Higher disparity difference means lower confidence
                    confidence_map[y, x] *= np.exp(-min(disparity_diff, cost_clip))  # Apply clipping
    
    # Matching cost-based confidence
    for y in range(height):
        for x in range(width):
            # Check if the disparity values are valid (not NaN or infinity)
            if not np.isnan(disp_left[y, x]) and not np.isinf(disp_left[y, x]):
                # Ensure the disparity is within bounds (no out-of-bounds indexing)
                disparity_offset = int(disp_left[y, x])
                if 0 <= x - disparity_offset < width:
                    # Compute the matching cost (SAD between the left and right images)
                    cost = np.abs(I_left[y, x] - I_right[y, x - disparity_offset])
                    # Clip cost before applying exponential
                    confidence_map[y, x] *= np.exp(-min(cost, cost_clip))  # Apply clipping
    
    # Optional: Normalize the confidence map to [0, 1]
    confidence_map -= np.min(confidence_map)
    confidence_map /= np.max(confidence_map)
    
    return confidence_map

# Example Usage
if __name__ == "__main__":
    # Load the left and right images
    I_left = plt.imread('/home/william/extdisk/data/middlebury/middlebury2014/Vintage-perfect/im0.png', format='gray')
    I_right = plt.imread('/home/william/extdisk/data/middlebury/middlebury2014/Vintage-perfect/im1.png', format='gray')

    # Compute disparity map (this is a placeholder, you'd use your stereo matching method)
    # stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    # disp_left = stereo.compute(I_left, I_right).astype(np.float32) / 16.0
    disp_left = read_pfm('/home/william/extdisk/data/middlebury/middlebury2014/Vintage-perfect/disp0.pfm')
    disp_right = read_pfm('/home/william/extdisk/data/middlebury/middlebury2014/Vintage-perfect/disp1.pfm')
    # Compute the confidence map
    confidence_map = compute_confidence_map(disp_left, disp_left, I_left, I_right)

    plt.figure(figsize=(12, 6))

    # First subplot: Display the disparity map
    plt.subplot(1, 2, 1)  # (rows, cols, index)
    plt.imshow(disp_left, cmap='plasma')
    plt.title('Disparity Map')
    plt.colorbar()

    # Second subplot: Display the confidence map
    plt.subplot(1, 2, 2)
    plt.imshow(confidence_map, cmap='magma')
    plt.title('Confidence Map')
    plt.colorbar()

    # Show the plots
    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()