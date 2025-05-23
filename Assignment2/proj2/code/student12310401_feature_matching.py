import numpy as np


def match_features(features1, features2, x1, y1, x2, y2):
        """
        This function does not need to be symmetric (e.g. it can produce
        different numbers of matches depending on the order of the arguments).

        To start with, simply implement the "ratio test", equation 4.18 in
        section 4.1.3 of Szeliski. There are a lot of repetitive features in
        these images, and all of their descriptors will look similar. The
        ratio test helps us resolve this issue (also see Figure 11 of David
        Lowe's IJCV paper).

        For extra credit you can implement various forms of spatial/geometric
        verification of matches, e.g. using the x and y locations of the features.

        Args:
        -   features1: A numpy array of shape (n,feat_dim) representing one set of
                features, where feat_dim denotes the feature dimensionality
        -   features2: A numpy array of shape (m,feat_dim) representing a second set
                features (m not necessarily equal to n)
        -   x1: A numpy array of shape (n,) containing the x-locations of features1
        -   y1: A numpy array of shape (n,) containing the y-locations of features1
        -   x2: A numpy array of shape (m,) containing the x-locations of features2
        -   y2: A numpy array of shape (m,) containing the y-locations of features2

        Returns:
        -   matches: A numpy array of shape (k,2), where k is the number of matches.
                The first column is an index in features1, and the second column is
                an index in features2
        -   confidences: A numpy array of shape (k,) with the real valued confidence for
                every match

        'matches' and 'confidences' can be empty e.g. (0x2) and (0x1)
        """
        #############################################################################
        # TODO: YOUR CODE HERE                                                  #  
        matches = []
        confidences = []

        for i, feature1 in enumerate(features1):
                distances = np.linalg.norm(features2 - feature1, axis=1)
                
                if len(distances) > 1:
                        sorted_indices = np.argsort(distances)
                        best_match = sorted_indices[0]
                        second_best_match = sorted_indices[1]
                        
                        if distances[best_match] < 0.9 * distances[second_best_match]:
                                matches.append([i, best_match])
                                confidences.append(distances[second_best_match]/distances[best_match])

        matches = np.array(matches)
        confidences = np.array(confidences)

        idxs = np.flipud(confidences.argsort()) 
        matches = matches[idxs]
        confidences = confidences[idxs]

        #############################################################################

        # raise NotImplementedError('`match_features` function in ' +
        # '`student_feature_matching.py` needs to be implemented')

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return matches, confidences
