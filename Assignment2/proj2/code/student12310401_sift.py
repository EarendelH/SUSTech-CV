import numpy as np
import cv2


def get_features(image, x, y, feature_width, scales=None):
    """
    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Args:
    -   image: A numpy array of shape (m,n) or (m,n,c). can be grayscale or color, your choice
    -   x: A numpy array of shape (k,), the x-coordinates of interest points
    -   y: A numpy array of shape (k,), the y-coordinates of interest points
    -   feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.
    -   scales: Python list or tuple if you want to detect and describe features
            at multiple scales

    You may also detect and describe features at particular orientations.

    Returns:
    -   fv: A numpy array of shape (k, feat_dim) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """
    assert image.ndim == 2, 'Image must be grayscale'
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    # If you choose to implement rotation invariance, enabling it should not    #
    # decrease your matching accuracy.                                          #
    #############################################################################
    
    # raise NotImplementedError('`get_features` function in ' +
    #     '`student_sift.py` needs to be implemented')

    
    k = len(x)  
    feat_dim = 128  
    fv = np.zeros((k, feat_dim))
    
    dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    magnitude = np.sqrt(dx**2 + dy**2)
    orientation = np.arctan2(dy, dx) * 180 / np.pi  
    orientation = (orientation + 360) % 360
    
    cell_width = feature_width // 4
    
    for i in range(k):
        if (x[i] < feature_width//2 or x[i] >= image.shape[1] - feature_width//2 or
            y[i] < feature_width//2 or y[i] >= image.shape[0] - feature_width//2):
            continue
            
        x_min = int(x[i] - feature_width//2)
        y_min = int(y[i] - feature_width//2)
        x_max = x_min + feature_width
        y_max = y_min + feature_width
        
        window_magnitude = magnitude[y_min:y_max, x_min:x_max]
        window_orientation = orientation[y_min:y_max, x_min:x_max]
        
        hist = np.zeros(128)
        
        for cell_y in range(4):
            for cell_x in range(4):
                cell_y_min = cell_y * cell_width
                cell_y_max = cell_y_min + cell_width
                cell_x_min = cell_x * cell_width
                cell_x_max = cell_x_min + cell_width
                
                cell_magnitude = window_magnitude[cell_y_min:cell_y_max, cell_x_min:cell_x_max]
                cell_orientation = window_orientation[cell_y_min:cell_y_max, cell_x_min:cell_x_max]
                
                for y_idx in range(cell_width):
                    for x_idx in range(cell_width):
                        mag = cell_magnitude[y_idx, x_idx]
                        angle = cell_orientation[y_idx, x_idx]
                        
                        bin_idx = int(angle // 45) % 8
                        
                        hist_idx = (cell_y * 4 + cell_x) * 8 + bin_idx
                        
                        hist[hist_idx] += mag
        
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist = hist / norm
        
        threshold = 0.3
        hist[hist > threshold] = threshold
        
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist = hist / norm
        
        fv[i] = hist
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    return fv
