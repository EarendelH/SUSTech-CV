import cv2
import numpy as np
import pickle
from utils import load_image, load_image_gray 
import cyvlfeat as vlfeat
import sklearn.metrics.pairwise as sklearn_pairwise
from sklearn.svm import LinearSVC
from scipy.stats import mode 
from time import time
from collections import Counter

def get_tiny_images(image_paths):
    """
    Builds tiny image features. Resizes images to 16x16, and normalizes them
    to have zero mean and unit standard deviation.
    """
    print("Getting tiny images...")
    feats = []
    target_size = (16, 16)

    for i, path in enumerate(image_paths):
        # if (i + 1) % 100 == 0:
        #     print(f" Processing tiny image {i+1}/{len(image_paths)}")
        img = load_image_gray(path)
        tiny_img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        
        flat_img = tiny_img.flatten()
        mean = np.mean(flat_img)
        std = np.std(flat_img)
        
        if std > 1e-6:
            normalized_flat_img = (flat_img - mean) / std
        else:
            normalized_flat_img = flat_img - mean
        
        feats.append(normalized_flat_img)

    return np.array(feats, dtype=np.float32)


def build_vocabulary(image_paths, vocab_size, sift_step_size=10, sift_fast_mode=True):
    """
    Samples SIFT descriptors from training images and clusters them using k-means
    to create a visual word vocabulary.

    Args:
    -   image_paths: List of image paths for training.
    -   vocab_size: The desired number of visual words (clusters).
    -   sift_step_size: Step size for dense SIFT.
    -   sift_fast_mode: Whether to use the fast DSIFT approximation.

    Returns:
    -   vocab: A vocab_size x 128 numpy array of cluster centers (visual words).
    """
    print(f"Building vocabulary of size {vocab_size} (SIFT step: {sift_step_size}, fast: {sift_fast_mode})...")
    all_descriptors = []
    
    start_time = time()
    for i, path in enumerate(image_paths):
        # if (i + 1) % 100 == 0:
        #     print(f" Processing image {i+1}/{len(image_paths)} for SIFT (vocab build)...")
        
        img = load_image_gray(path)
        img_single = img.astype(np.float32)

        frames, descriptors = vlfeat.sift.dsift(img_single, 
                                                step=sift_step_size, 
                                                fast=sift_fast_mode)
        
        if descriptors is not None and descriptors.shape[0] > 0:
            all_descriptors.append(descriptors)

    if not all_descriptors:
        print("CRITICAL WARNING: No SIFT descriptors found. Vocabulary will be empty or zeros.")
        sift_dim = 128
        return np.zeros((vocab_size, sift_dim), dtype=np.float32)

    all_descriptors_np = np.vstack(all_descriptors).astype(np.float32)
    print(f"Total SIFT descriptors sampled: {all_descriptors_np.shape[0]}")

    actual_k = vocab_size
    if all_descriptors_np.shape[0] < vocab_size:
        print(f"Warning: Number of SIFT descriptors ({all_descriptors_np.shape[0]}) is less than "
              f"requested vocab_size ({vocab_size}). Using {all_descriptors_np.shape[0]} clusters instead.")
        actual_k = all_descriptors_np.shape[0]
    
    if actual_k == 0:
        print("CRITICAL WARNING: No descriptors to cluster. Returning zero vocabulary.")
        sift_dim = 128
        return np.zeros((vocab_size, sift_dim), dtype=np.float32)

    print(f"Clustering {all_descriptors_np.shape[0]} descriptors into {actual_k} clusters...")
    vocab = vlfeat.kmeans.kmeans(all_descriptors_np, 
                                 actual_k,
                                 initialization="PLUSPLUS",
                                 algorithm="LLOYD",
                                 max_num_iterations=100)
    end_time = time()
    print(f"Vocabulary building took {end_time - start_time:.2f} seconds. Vocab shape: {vocab.shape}")
    return vocab.astype(np.float32)


def get_bags_of_sifts(image_paths, vocab_filename, sift_step_size=5, sift_fast_mode=True):
    """
    Converts images into Bag of SIFT feature representations using a precomputed vocabulary.

    Args:
    -   image_paths: List of image paths.
    -   vocab_filename: Path to the pickled vocabulary file.
    -   sift_step_size: Step size for dense SIFT.
    -   sift_fast_mode: Whether to use the fast DSIFT approximation.

    Returns:
    -   image_feats: An N x vocab_size numpy array of BoS features.
    """
    print(f"Loading vocabulary from {vocab_filename}...")
    try:
        with open(vocab_filename, 'rb') as f:
            vocab = pickle.load(f)
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Vocab file '{vocab_filename}' not found.")
        return np.array([])
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load vocab '{vocab_filename}': {e}")
        return np.array([])

    if not isinstance(vocab, np.ndarray) or vocab.ndim != 2 or (vocab.shape[0] > 0 and vocab.shape[1] != 128):
        print(f"CRITICAL ERROR: Loaded vocabulary is invalid. Shape: {vocab.shape if isinstance(vocab, np.ndarray) else 'Not an ndarray'}")
        return np.array([])
    
    vocab = vocab.astype(np.float32)
    vocab_size = vocab.shape[0]

    if vocab_size == 0:
        print("CRITICAL WARNING: Vocab is empty. Returning N x 0 BoS features.")
        return np.zeros((len(image_paths), 0), dtype=np.float32)

    print(f"Getting Bags of SIFTs (SIFT step: {sift_step_size}, fast: {sift_fast_mode}, vocab size: {vocab_size})...")
    feats = []
    
    start_time = time()
    for i, path in enumerate(image_paths):
        # if (i + 1) % 100 == 0:
        #     print(f" Processing image {i+1}/{len(image_paths)} for BoS features...")

        img = load_image_gray(path)
        img_single = img.astype(np.float32)

        frames, descriptors = vlfeat.sift.dsift(img_single, 
                                                step=sift_step_size, 
                                                fast=sift_fast_mode)
        
        histogram = np.zeros(vocab_size, dtype=np.float32)

        if descriptors is not None and descriptors.shape[0] > 0:
            descriptors_single = descriptors.astype(np.float32)
            try:
                assignments = vlfeat.kmeans.kmeans_quantize(descriptors_single, vocab)
                hist_counts, _ = np.histogram(assignments, bins=np.arange(vocab_size + 1))
                histogram = hist_counts.astype(np.float32)
                hist_sum = np.sum(histogram)
                if hist_sum > 1e-9:
                    histogram = histogram / hist_sum
            except Exception as e:
                print(f"  Warning: Error during SIFT quantization for {path}: {e}. Using zero histogram.")
        feats.append(histogram)
    
    feats_np = np.array(feats, dtype=np.float32)
    end_time = time()
    print(f"BoS feature extraction took {end_time - start_time:.2f} seconds. Features shape: {feats_np.shape}")
    return feats_np


def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats,
                              metric='euclidean', k=1): # Added k parameter
    """
    Predicts categories for test images using k-Nearest Neighbors.

    Args:
    -   train_image_feats: N x d array of training features.
    -   train_labels: List of N training labels.
    -   test_image_feats: M x d array of testing features.
    -   metric: Distance metric for sklearn.metrics.pairwise_distances.
    -   k: Number of nearest neighbors to use.

    Returns:
    -   predicted_labels: List of M predicted labels for test images.
    """
    print(f"Classifying using {k}-Nearest Neighbor(s) with metric: {metric}...")
    
    if train_image_feats.shape[0] == 0:
        print("Warning: Empty training features. Cannot classify.")
        return ["Unknown"] * test_image_feats.shape[0]
    if test_image_feats.shape[0] == 0:
        print("Warning: Empty test features.")
        return []

    train_labels_np = np.array(train_labels)
    
    actual_metric = metric
    if metric.lower() == 'chi2':
        print("Note: For 'chi2' metric with `pairwise_distances`, it often implies `chi2_kernel`. "
              "If true chi-squared distance is needed, 'cosine' is a robust alternative for histograms, or implement chi2 distance separately.")
        actual_metric = 'cosine' 

    try:
        distances = sklearn_pairwise.pairwise_distances(test_image_feats, train_image_feats, metric=actual_metric)
    except ValueError as e:
        print(f"Error computing distances with metric '{actual_metric}': {e}. Trying 'euclidean' as fallback.")
        try:
            distances = sklearn_pairwise.pairwise_distances(test_image_feats, train_image_feats, metric='euclidean')
        except Exception as e_fallback:
            print(f"Fallback to 'euclidean' also failed: {e_fallback}. Cannot classify.")
            return ["Unknown"] * test_image_feats.shape[0]

    if k == 1:
        nearest_neighbor_indices = np.argmin(distances, axis=1)
        predicted_labels = list(train_labels_np[nearest_neighbor_indices])
    else:
        if k <= 0:
            print(f"Warning: k must be positive. Got {k}. Setting k=1.")
            k = 1
        if k > train_image_feats.shape[0]: 
            print(f"Warning: k ({k}) > num training samples ({train_image_feats.shape[0]}). Setting k to num training samples.")
            k = train_image_feats.shape[0]
        
        k_nearest_indices = np.argsort(distances, axis=1)[:, :k]
        k_nearest_labels = train_labels_np[k_nearest_indices] # Shape: (num_test_samples, k)
        
        predicted_labels = []
        for labels_row in k_nearest_labels:
            if len(labels_row) > 0:
                most_common_label = Counter(labels_row).most_common(1)[0][0]
                predicted_labels.append(most_common_label)
            else:
                predicted_labels.append("Unknown") 

    return predicted_labels


def svm_classify(train_image_feats, train_labels, test_image_feats, C_value=7.0): # Added C_value parameter
    """
    Trains 1-vs-all Linear SVMs and predicts categories for test images.

    Args:
    -   train_image_feats: N x d array of training features.
    -   train_labels: List of N training labels.
    -   test_image_feats: M x d array of testing features.
    -   C_value: Regularization parameter C for LinearSVC.

    Returns:
    -   predicted_labels: List of M predicted labels for test images.
    """
    print(f"Training and classifying with 1-vs-all Linear SVMs (C={C_value})...")
    
    if train_image_feats.shape[0] == 0:
        print("Warning: Empty training features. Cannot train SVMs.")
        return ["Unknown"] * test_image_feats.shape[0]
    if test_image_feats.shape[0] == 0:
        print("Warning: Empty test features.")
        return []

    categories = sorted(list(set(train_labels)))
    num_categories = len(categories)

    svm_params = {'random_state': 0, 'tol': 1e-3, 'loss': 'hinge', 'C': C_value, 'max_iter': 2000}
    
    svm_params['dual'] = True 

    trained_svms = {}

    print(f"Training {num_categories} 1-vs-all SVMs...")
    for i, category in enumerate(categories):
        binary_labels = np.array([1 if label == category else 0 for label in train_labels])
        svm = LinearSVC(**svm_params)
        try:
            svm.fit(train_image_feats, binary_labels)
            trained_svms[category] = svm
        except Exception as e:
            print(f"  Failed to train SVM for category '{category}': {e}.")

    num_test_images = test_image_feats.shape[0]
    decision_scores = np.full((num_test_images, num_categories), -np.inf, dtype=np.float32)

    print("Predicting categories using trained SVMs...")
    for i, category in enumerate(categories):
        if category in trained_svms:
            svm = trained_svms[category]
            try:
                scores_for_category = svm.decision_function(test_image_feats)
                decision_scores[:, i] = scores_for_category
            except Exception as e:
                 print(f"  Failed to get decision scores for category '{category}': {e}.")
    
    predicted_indices = np.argmax(decision_scores, axis=1)
    predicted_labels = [categories[idx] for idx in predicted_indices]
    
    return predicted_labels