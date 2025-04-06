import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_interest_points(image, feature_width):
        """
        Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
        You can create additional interest point detector functions (e.g. MSER)
        for extra credit.

        If you're finding spurious interest point detections near the boundaries,
        it is safe to simply suppress the gradients / corners near the edges of
        the image.

        Useful in this function in order to (a) suppress boundary interest
        points (where a feature wouldn't fit entirely in the image, anyway)
        or (b) scale the image filters being used. Or you can ignore it.

        By default you do not need to make scale and orientation invariant
        local features.

        The lecture slides and textbook are a bit vague on how to do the
        non-maximum suppression once you've thresholded the cornerness score.
        You are free to experiment. For example, you could compute connected
        components and take the maximum value within each component.
        Alternatively, you could run a max() operator on each sliding window. You
        could use this to ensure that every interest point is at a local maximum
        of cornerness.

        Args:
        -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
        -   feature_width: integer representing the local feature width in pixels.

        Returns:
        -   x: A numpy array of shape (N,) containing x-coordinates of interest points
        -   y: A numpy array of shape (N,) containing y-coordinates of interest points
        -   confidences (optional): numpy nd-array of dim (N,) containing the strength
                of each interest point
        -   scales (optional): A numpy array of shape (N,) containing the scale at each
                interest point
        -   orientations (optional): A numpy array of shape (N,) containing the orientation
                at each interest point
        """
        confidences, scales, orientations = None, None, None
        #############################################################################
        # TODO: YOUR HARRIS CORNER DETECTOR CODE HERE      
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        Ixx = Ix ** 2
        Iyy = Iy ** 2
        Ixy = Ix * Iy

        gaussion_kernel = cv2.getGaussianKernel(ksize=3, sigma=1)
        Ixx = cv2.filter2D(Ixx, ddepth=-1, kernel=gaussion_kernel) 
        Ixy = cv2.filter2D(Ixy, ddepth=-1, kernel=gaussion_kernel)
        Iyy = cv2.filter2D(Iyy, ddepth=-1, kernel=gaussion_kernel)

        k = 0.04
        R = (Ixx * Iyy - Ixy ** 2) - k * (Ixx + Iyy) ** 2

        R_max = np.max(R)
        threshold = 0.01 * R_max
        
        local_max = np.zeros_like(R, dtype=bool)
        for i in range(1, R.shape[0] - 1):
            for j in range(1, R.shape[1] - 1):
                window = R[i - 1:i + 2, j - 1:j + 2]
                if R[i, j] == np.max(window) and R[i, j] > threshold:
                    local_max[i, j] = True
        
        
        y, x = np.where(local_max)

        if len(x) == 0:
            return np.array([]), np.array([]), confidences, scales, orientations
            
        scores = R[y, x]
        
        max_fetch_point = 1500
        corners_coords = np.column_stack((x, y, scores))
        
        sorted_indices = np.argsort(-corners_coords[:, 2])
        sorted_corners = corners_coords[sorted_indices]
        
        n_points = len(sorted_corners)
        radii = np.zeros(n_points)
        radii[0] = max(image.shape[0], image.shape[1])
        
        for i in range(1, n_points):
            xi, yi = sorted_corners[i, :2]
            stronger_points = sorted_corners[:i, :2]
            distances = np.sqrt(np.sum((stronger_points - np.array([xi, yi])) ** 2, axis=1))
            radii[i] = np.min(distances)
        
        if n_points > max_fetch_point:
            indices_sorted_by_radius = np.argsort(-radii)[:max_fetch_point]
            selected_corners = sorted_corners[indices_sorted_by_radius]
        else:
            selected_corners = sorted_corners
        
        x = selected_corners[:, 0].astype(np.int32)
        y = selected_corners[:, 1].astype(np.int32)
        
        confidences = selected_corners[:, 2]
        #############################################################################

        # raise NotImplementedError('adaptive non-maximal suppression in ' +
        # '`student_harris.py` needs to be implemented')

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return x,y, confidences, scales, orientations


