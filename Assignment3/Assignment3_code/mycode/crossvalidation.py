import numpy as np
import os
import os.path as osp
import glob
import random
import pickle
from time import time

try:
    import student_code_12310401 as sc 
    import utils
except ImportError as e:
    print(f"Error importing student_code_SID or utils: {e}")
    exit()

def get_all_image_paths_by_category(data_path, categories, fmt='jpg'):
    """
    For each category, collects all image paths from 'train' and 'test' subdirectories
    within the given data_path. This pool is then used for random splitting in CV.
    """
    all_paths_by_cat = {cat: [] for cat in categories}
    print(f"Collecting all image paths from '{data_path}/[train|test]/[category_name]/' structure...")

    for cat in categories:
        image_list_for_cat = []
        for subdir_type in ['train', 'test']:
            pth_pattern = osp.join(data_path, subdir_type, cat, f'*.{fmt}')
            paths_in_subdir = glob.glob(pth_pattern)
            image_list_for_cat.extend(paths_in_subdir)

        if not image_list_for_cat:
            print(f"    CRITICAL WARNING: Category '{cat}' has no images found in either 'train' or 'test' subdirectories "
                  f"under '{data_path}'. Cross-validation results will be affected or fail for this category.")
        else:
            random.shuffle(image_list_for_cat)
        all_paths_by_cat[cat] = image_list_for_cat

    print("Finished collecting and shuffling all image paths for each category.")
    return all_paths_by_cat

#Main Cross-Validation Function
def run_cross_validation(
    data_path,
    categories,
    num_iterations,
    feature_type,
    classifier_type,
    k_for_knn=1,
    metric_for_knn='euclidean',
    vocab_size_for_sift=200,
    sift_step_vocab=10,         # SIFT step for building vocabulary
    sift_fast_vocab=True,       # SIFT fast mode for building vocabulary
    sift_step_features=5,       # SIFT step for extracting BoS features
    sift_fast_features=True,    # SIFT fast mode for extracting BoS features
    svm_c_value=5.0,
    force_vocab_rebuild=False   # To force rebuilding vocab
):
    """
    Executes cross-validation.
    For Bag of SIFT, vocabulary is built ONCE if it doesn't exist or if forced,
    using a sample of images, and then reused for all folds.
    Vocabulary filename is now based on its size, e.g., vocab_200.pkl.
    """
    accuracies = []
    all_image_paths_by_cat = get_all_image_paths_by_category(data_path, categories)

    for cat in categories:
        if len(all_image_paths_by_cat[cat]) < 200: # Needs 100 for train, 100 for test
             print(f"Warning: Category '{cat}' has only {len(all_image_paths_by_cat[cat])} images. "
                   f"Required at least 200 for a 100-train/100-test split. "
                   f"Sampling will use available images, which might be fewer than 100/100 for this category.")

    cv_run_vocab_filename = None
    if feature_type == 'bag_of_sifts':
        cv_run_vocab_filename = f"vocab_{vocab_size_for_sift}.pkl"
        print(f"\n--- Vocabulary Handling for Bag of SIFT (Size: {vocab_size_for_sift}) ---")
        print(f"Target vocabulary file: {cv_run_vocab_filename}")

        if not osp.exists(cv_run_vocab_filename) or force_vocab_rebuild:
            if osp.exists(cv_run_vocab_filename) and force_vocab_rebuild:
                print(f"Forcing rebuild of existing vocabulary file: {cv_run_vocab_filename}")
            else:
                print(f"Vocabulary file {cv_run_vocab_filename} not found. Building new vocabulary.")

            images_for_vocab_build = []
            num_images_per_cat_for_vocab = 100 
            for cat in categories:
                images_for_vocab_build.extend(all_image_paths_by_cat[cat][:num_images_per_cat_for_vocab])

            if not images_for_vocab_build:
                print("CRITICAL ERROR: No images collected for vocabulary building. Cannot proceed with BoS.")
                return 0, 0, []

            random.shuffle(images_for_vocab_build) # Shuffle the collected set

            print(f"Building vocabulary using {len(images_for_vocab_build)} images, target size {vocab_size_for_sift}, saving to {cv_run_vocab_filename}")

            vocab_build_start_time = time()
            vocab = sc.build_vocabulary(images_for_vocab_build,
                                        vocab_size_for_sift,
                                        sift_step_size=sift_step_vocab,
                                        sift_fast_mode=sift_fast_vocab)
            vocab_build_end_time = time()

            if vocab is None or vocab.shape[0] == 0:
                print(f"CRITICAL ERROR: Vocabulary building failed or resulted in an empty vocabulary. Cannot proceed with BoS.")
                if osp.exists(cv_run_vocab_filename):
                    try: os.remove(cv_run_vocab_filename)
                    except OSError: pass
                return 0, 0, []

            with open(cv_run_vocab_filename, 'wb') as f_vocab:
                pickle.dump(vocab, f_vocab)
            print(f"--- Vocabulary built in {vocab_build_end_time - vocab_build_start_time:.2f}s and saved. Actual size: {vocab.shape[0]} (Target: {vocab_size_for_sift}) ---")
        else:
            print(f"Using existing vocabulary file: {cv_run_vocab_filename}")
            try:
                with open(cv_run_vocab_filename, 'rb') as f_vocab:
                    vocab = pickle.load(f_vocab)
                if vocab is None or vocab.shape[0] == 0:
                    print(f"WARNING: Existing vocabulary file {cv_run_vocab_filename} is empty or corrupted. Consider rebuilding.")
                else:
                    print(f"Successfully loaded existing vocabulary. Size: {vocab.shape[0]}")
            except Exception as e:
                print(f"ERROR loading existing vocabulary {cv_run_vocab_filename}: {e}. Consider rebuilding.")
                return 0,0,[]


    # --- Cross-Validation Folds ---
    for i in range(num_iterations):
        fold_start_time = time()
        print(f"\n--- Cross-Validation Iteration {i + 1}/{num_iterations} ---")

        current_train_image_paths = []
        current_train_labels = []
        current_test_image_paths = []
        current_test_labels = []

        for cat in categories:
            available_images_for_cat = list(all_image_paths_by_cat[cat]) 
            random.shuffle(available_images_for_cat) 

            num_available = len(available_images_for_cat)

            cat_train_paths = available_images_for_cat[:min(100, num_available)]
            remaining_for_test = available_images_for_cat[len(cat_train_paths):]
            cat_test_paths = remaining_for_test[:min(100, len(remaining_for_test))]

            if len(cat_train_paths) < 100 or len(cat_test_paths) < 100:
                if len(cat_train_paths) > 0 or len(cat_test_paths) > 0: # only print if there's actually data for the category
                    print(f"  Note for category '{cat}': Using {len(cat_train_paths)} train, {len(cat_test_paths)} test (target 100/100).")

            current_train_image_paths.extend(cat_train_paths)
            current_train_labels.extend([cat] * len(cat_train_paths))
            current_test_image_paths.extend(cat_test_paths)
            current_test_labels.extend([cat] * len(cat_test_paths))

        if not current_train_image_paths or not current_test_image_paths:
            print(f"Skipping iteration {i+1}: Not enough images for overall train/test sets after sampling.")
            
            if not current_train_image_paths:
                print("  No training images sampled for this fold.")
            if not current_test_image_paths:
                print("  No test images sampled for this fold.")


        print(f"  Iter {i+1}: {len(current_train_image_paths)} train imgs, {len(current_test_image_paths)} test imgs.")

        train_image_feats, test_image_feats = None, None

        if feature_type == 'tiny_image':
            print("  Extracting Tiny Image features...")
            if current_train_image_paths: train_image_feats = sc.get_tiny_images(current_train_image_paths)
            else: train_image_feats = np.array([]) # Ensure it's an array for shape checks
            if current_test_image_paths: test_image_feats = sc.get_tiny_images(current_test_image_paths)
            else: test_image_feats = np.array([])

        elif feature_type == 'bag_of_sifts':
            if cv_run_vocab_filename is None or not osp.exists(cv_run_vocab_filename):
                print("CRITICAL ERROR: BoS selected, but no valid vocabulary file available for this CV run. Skipping fold.")
                continue # Skip this fold if vocab is missing
            print(f"  Extracting BoS features using shared vocab: {cv_run_vocab_filename}")
            if current_train_image_paths:
                train_image_feats = sc.get_bags_of_sifts(current_train_image_paths,
                                                         cv_run_vocab_filename,
                                                         sift_step_size=sift_step_features,
                                                         sift_fast_mode=sift_fast_features)
            else: train_image_feats = np.array([])

            if current_test_image_paths:
                test_image_feats = sc.get_bags_of_sifts(current_test_image_paths,
                                                        cv_run_vocab_filename,
                                                        sift_step_size=sift_step_features,
                                                        sift_fast_mode=sift_fast_features)
            else: test_image_feats = np.array([])
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")

        valid_train_feats = train_image_feats is not None and hasattr(train_image_feats, 'shape') and train_image_feats.shape[0] == len(current_train_image_paths)
        valid_test_feats = test_image_feats is not None and hasattr(test_image_feats, 'shape') and test_image_feats.shape[0] == len(current_test_image_paths)

        if not current_train_image_paths:
            valid_train_feats = (train_image_feats is not None and hasattr(train_image_feats, 'shape') and train_image_feats.shape[0] == 0)
        if not current_test_image_paths:
            valid_test_feats = (test_image_feats is not None and hasattr(test_image_feats, 'shape') and test_image_feats.shape[0] == 0)


        if not (valid_train_feats and valid_test_feats):
            print(f"  Feature extraction failed or produced inconsistent shapes for fold {i+1}. Skipping classification.")
            print(f"    Train paths: {len(current_train_image_paths)}, Train feats shape: {train_image_feats.shape if hasattr(train_image_feats, 'shape') else 'None'}")
            print(f"    Test paths: {len(current_test_image_paths)}, Test feats shape: {test_image_feats.shape if hasattr(test_image_feats, 'shape') else 'None'}")
            continue

        predicted_categories = None
        if hasattr(train_image_feats, 'shape') and train_image_feats.shape[0] > 0 and \
           hasattr(test_image_feats, 'shape') and test_image_feats.shape[0] > 0 :
            if classifier_type == 'knn':
                print(f"  Running K-NN (k={k_for_knn}, metric='{metric_for_knn}')...")
                predicted_categories = sc.nearest_neighbor_classify(train_image_feats,
                                                                    current_train_labels,
                                                                    test_image_feats,
                                                                    metric=metric_for_knn,
                                                                    k=k_for_knn)
            elif classifier_type == 'svm':
                print(f"  Running SVM (C={svm_c_value})...")
                predicted_categories = sc.svm_classify(train_image_feats,
                                                       current_train_labels,
                                                       test_image_feats,
                                                       C_value=svm_c_value)
            else:
                raise ValueError(f"Unknown classifier_type: {classifier_type}")
        elif hasattr(test_image_feats, 'shape') and test_image_feats.shape[0] == 0 and len(current_test_image_paths) == 0:
             print(f"  No test images/features to classify for fold {i+1}.")
        elif hasattr(train_image_feats, 'shape') and train_image_feats.shape[0] == 0 and len(current_train_image_paths) > 0 :
            print(f"  Training features are empty, but training images were provided. Skipping classification for fold {i+1}.")
        else: 
            print(f"  Skipping classification for fold {i+1} due to issues with feature sets (e.g., train features available but no test features to predict).")
            print(f"    Train feats shape: {train_image_feats.shape if hasattr(train_image_feats, 'shape') else 'None'}")
            print(f"    Test feats shape: {test_image_feats.shape if hasattr(test_image_feats, 'shape') else 'None'}")


        if predicted_categories is not None and len(current_test_labels) > 0:
            if not isinstance(predicted_categories, (list, np.ndarray)):
                print(f"  Warning: Classifier did not return a list or numpy array of predictions. Type: {type(predicted_categories)}")
            elif len(predicted_categories) != len(current_test_labels):
                print(f"  Warning: Number of predictions ({len(predicted_categories)}) does not match number of test labels ({len(current_test_labels)}).")
            else:
                correct_predictions = np.sum(np.array(predicted_categories) == np.array(current_test_labels))
                accuracy_fold = correct_predictions / len(current_test_labels)
                print(f"  Accuracy for fold {i + 1}: {accuracy_fold * 100:.2f}%")
                accuracies.append(accuracy_fold)
        elif len(current_test_labels) == 0 and len(current_test_image_paths) > 0: 
             print(f"  Test images were present ({len(current_test_image_paths)}) but no test labels or predictions failed for fold {i+1}.")
        elif len(current_test_image_paths) == 0: 
            print(f"  No test images sampled for fold {i+1}, cannot calculate accuracy.")

        fold_end_time = time()
        print(f"  Iteration {i+1} completed in {fold_end_time - fold_start_time:.2f} seconds.")

    
    if accuracies:
        mean_accuracy = np.mean(accuracies)
        std_dev_accuracy = np.std(accuracies)
        print(f"\n\n--- Cross-Validation Summary ({feature_type} & {classifier_type}) ---")
        print(f"Num iterations: {num_iterations} (Successful folds recorded: {len(accuracies)})")
        if feature_type == 'bag_of_sifts':
            print(f"Vocabulary file used/created: {cv_run_vocab_filename} (Size: {vocab_size_for_sift})")
        print(f"Mean Accuracy: {mean_accuracy * 100:.2f}%")
        print(f"Std Dev Accuracy: {std_dev_accuracy * 100:.2f}%")
        print(f"Individual accuracies: {[f'{acc*100:.2f}%' for acc in accuracies]}")
        return mean_accuracy, std_dev_accuracy, accuracies
    else:
        print(f"\n\n--- Cross-Validation ({feature_type} & {classifier_type}) ---")
        if feature_type == 'bag_of_sifts' and cv_run_vocab_filename:
             print(f"Vocabulary file targeted: {cv_run_vocab_filename} (Size: {vocab_size_for_sift})")
        print("No accuracies recorded. Check logs for errors (data loading, feature extraction, classification, etc.).")
        return 0, 0, []

# --- Example Usage ---
if __name__ == '__main__':
    DATA_PATH = osp.join('..', 'data')
    CATEGORIES = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office',
                  'Industrial', 'Suburb', 'InsideCity', 'TallBuilding',
                  'Street', 'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest']
    NUM_CV_ITERATIONS = 3 

    print("Starting cross-validation experiments...")
    print(f"Data path: {osp.abspath(DATA_PATH)}")
    print(f"Num iterations per experiment: {NUM_CV_ITERATIONS}")
    print(f"Target vocabulary files will be named like 'vocab_SIZE.pkl' (e.g., vocab_100.pkl) and reused if present.")

    # --- Tiny Images + KNN ---
    print("\n\n===== EXPERIMENT 1: Tiny Images + KNN =====")
    run_cross_validation(
        data_path=DATA_PATH, categories=CATEGORIES, num_iterations=NUM_CV_ITERATIONS,
        feature_type='tiny_image', classifier_type='knn', k_for_knn=1, metric_for_knn='cosine'
    )

    # --- Bag of SIFT + KNN ---
    print("\n\n===== EXPERIMENT 2: Bag of SIFT + KNN =====")
    run_cross_validation(
        data_path=DATA_PATH, categories=CATEGORIES, num_iterations=NUM_CV_ITERATIONS,
        feature_type='bag_of_sifts', classifier_type='knn',
        vocab_size_for_sift=100,      
        sift_step_vocab=10,           
        sift_fast_vocab=True,
        sift_step_features=5,         
        sift_fast_features=True,
        k_for_knn=5,
        metric_for_knn='euclidean',
        force_vocab_rebuild=False     
    )

    # --- Bag of SIFT + SVM ---
    print("\n\n===== EXPERIMENT 3: Bag of SIFT + SVM =====")
    run_cross_validation(
        data_path=DATA_PATH, categories=CATEGORIES, num_iterations=NUM_CV_ITERATIONS,
        feature_type='bag_of_sifts', classifier_type='svm',
        vocab_size_for_sift=200,      
        sift_step_vocab=8,
        sift_fast_vocab=True,
        sift_step_features=4,
        sift_fast_features=True,
        svm_c_value=7.0,
        force_vocab_rebuild=False     
    )
    
    print("\n\n===== EXPERIMENT 4: Bag of SIFT + SVM (Forced Vocab Rebuild) =====")
    run_cross_validation(
        data_path=DATA_PATH, categories=CATEGORIES, num_iterations=NUM_CV_ITERATIONS,
        feature_type='bag_of_sifts', classifier_type='svm',
        vocab_size_for_sift=100,      
        sift_step_vocab=10,           
        sift_fast_vocab=True,
        sift_step_features=5,
        sift_fast_features=True,
        svm_c_value=5.0,
        force_vocab_rebuild=True
    )

    print("\n\n--- All cross-validation experiments complete. ---")