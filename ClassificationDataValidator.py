# ClassificationDataValidator
"""

class ClassificationDataValidator:
    def __init__(self, raise_warning=True):
        """Initialize the ClassificationDataValidator class"""
        self.raise_warning = raise_warning

    def check_data_type(self, y_true, y_pred, y_probs=None):
        """
        Check if input data types are valid.

        Parameters:
        -----------
        y_true : array-like
            Ground truth (correct) labels.
        y_pred : array-like
            Predicted labels.
        y_probs : array-like, optional
            Predicted probabilities for each class (2D array).
        """
        valid_types = (np.ndarray, list)
        if not isinstance(y_true, valid_types) or not isinstance(y_pred, valid_types):
            raise TypeError("y_true and y_pred must be numpy array or list")

        if y_probs is not None:
            if not isinstance(y_probs, valid_types):
                raise TypeError("y_probs must be numpy array or list")
            if len(np.array(y_probs).shape) != 2:
                raise ValueError("y_probs must be a 2D array (samples x classes)")

    def check_missing_values(self, y_true, y_pred, y_probs=None):
        """
        Check for missing values in y_true, y_pred, and y_probs.
        """
        if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
            raise ValueError("Missing values (NaN) detected in y_true or y_pred")

        if y_probs is not None:
            if np.any(np.isnan(y_probs)):
                raise ValueError("Missing values (NaN) detected in y_probs")

    def check_lengths(self, y_true, y_pred, y_probs=None):
        """
        Check if y_true, y_pred, and y_probs have the same length.
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")

        if y_probs is not None:
            if len(y_true) != len(y_probs):
                raise ValueError("y_true and y_probs must have the same length")

    def check_classes(self, y_true, y_pred):
        """
        Check if y_true and y_pred have the same set of classes.
        """
        unique_true = np.unique(y_true)
        unique_pred = np.unique(y_pred)
        if not np.array_equal(unique_true, unique_pred):
            raise ValueError("y_true and y_pred must have the same set of classes")

    def check_class_labels(self, y_true, y_pred):
        """
        Check if y_true and y_pred contain valid class labels (integers).
        """
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)

        # Skip check if arrays are empty
        if len(y_true_arr) == 0 or len(y_pred_arr) == 0:
            return

        if not np.issubdtype(y_true_arr.dtype, np.integer) or not np.issubdtype(y_pred_arr.dtype, np.integer):
            raise TypeError("y_true and y_pred must contain integer class labels")

    def check_outliers(self, y_true, threshold=3):
        """
        Check for outliers in y_true using Z-score.
        """
        z_scores = (y_true - np.mean(y_true)) / np.std(y_true)
        if np.any(np.abs(z_scores) > threshold):
            if self.raise_warning:
                print("Warning: Outliers detected in y_true")
            else:
                raise ValueError("Outliers detected in y_true")

    def check_distribution(self, y_true, distribution='normal'):
        """
        Check if y_true follows a specific distribution.
        """
        if distribution == 'normal':
            stat, p_value = kstest(y_true, 'norm')
            if p_value < 0.05:
                if self.raise_warning:
                    print("Warning: y_true does not follow a normal distribution")
                else:
                    raise ValueError("y_true does not follow a normal distribution")

    def check_correlation(self, y_true, y_pred, threshold=0.8):
        """
        Check for high correlation between y_true and y_pred.
        """
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        if corr > threshold:
            if self.raise_warning:
                print(f"Warning: High correlation detected between y_true and y_pred (corr={corr})")
            else:
                raise ValueError(f"High correlation detected between y_true and y_pred (corr={corr})")

    def check_missing_large_data(self, y_true, y_pred, sample_size=1000):
        """
        Check for missing values in large data using sampling.
        """
        indices = np.random.choice(len(y_true), sample_size, replace=False)
        if np.any(np.isnan(y_true[indices])) or np.any(np.isnan(y_pred[indices])):
            if self.raise_warning:
                print("Warning: Missing values (NaN) detected in sampled data")
            else:
                raise ValueError("Missing values (NaN) detected in sampled data")
    def check_empty_arrays(self, y_true, y_pred):
        """
        Check if input arrays are empty.
        """
        if len(y_true) == 0 or len(y_pred) == 0:
            raise ValueError("Input arrays cannot be empty")

    def validate_probabilities(self, y_probs):
        """
        Check if probabilities are valid (between 0 and 1 and sum to 1).
        """
        if np.any(y_probs < 0) or np.any(y_probs > 1):
            raise ValueError("y_probs must have values between 0 and 1")

        # Check if probabilities sum to 1 for each sample
        if not np.allclose(np.sum(y_probs, axis=1), 1):
            raise ValueError("Probabilities in y_probs must sum to 1 for each sample")

    def validate_top_k(self, y_probs, k):
        """
        Check if k is valid for top-k accuracy.
        """
        if k > y_probs.shape[1]:
            raise ValueError(f"k={k} is greater than the number of classes {y_probs.shape[1]}")
        if k <= 0:
            raise ValueError(f"k={k} must be a positive integer")

    def check_class_balance(self, y_true, threshold=0.1):
        """
        Check if classes are imbalanced.
        """
        class_counts = np.bincount(y_true)
        min_count = np.min(class_counts)
        max_count = np.max(class_counts)

        if min_count / max_count < threshold:
            if self.raise_warning:
                print(f"Warning: Class imbalance detected (min_count={min_count}, max_count={max_count})")
            else:
                raise ValueError(f"Class imbalance detected (min_count={min_count}, max_count={max_count})")
    def check_probabilities_classes(self, y_true, y_probs):
        """
        Check if the number of classes in y_probs matches the number of classes in y_true,
        and that the probabilities make sense for the given true labels.
        """
        n_classes_true = len(np.unique(y_true))
        n_classes_probs = y_probs.shape[1]

        if n_classes_probs != n_classes_true:
            raise ValueError(
                f"Number of classes in y_probs ({n_classes_probs}) "
                f"does not match number of classes in y_true ({n_classes_true})"
            )

        # Additional check that the predicted class probabilities make sense
        predicted_classes = np.argmax(y_probs, axis=1)
        if not np.array_equal(np.sort(np.unique(predicted_classes)), np.sort(np.unique(y_true))):
            if self.raise_warning:
                print("Warning: Predicted classes from probabilities don't match true class labels")
            else:
                raise ValueError("Predicted classes from probabilities don't match true class labels")

    def validate_all(self, y_true, y_pred, y_probs=None, check_outliers=False, check_distribution=False, check_correlation=False, check_missing_large=False, check_class_balance=False, sample_size=1000):
        """
        Run all validation checks.
        """
        self.check_empty_arrays(y_true, y_pred)  # Check for empty arrays first
        self.check_data_type(y_true, y_pred, y_probs)
        self.check_missing_values(y_true, y_pred, y_probs)
        self.check_lengths(y_true, y_pred, y_probs)
        self.check_classes(y_true, y_pred)
        self.check_class_labels(y_true, y_pred)

        if y_probs is not None:
            self.validate_probabilities(y_probs)
            self.check_probabilities_classes(y_true, y_probs)

        if check_outliers:
            self.check_outliers(y_true)
        if check_distribution:
            self.check_distribution(y_true)
        if check_correlation:
            self.check_correlation(y_true, y_pred)
        if check_missing_large:
            self.check_missing_large_data(y_true, y_pred, sample_size)
        if check_class_balance:
            self.check_class_balance(y_true)

        return True  # Return True if all checks pass



# Metadata for classification metrics
CLASSIFICATION_METRIC_DETAILS = {
    "accuracy_score": {
        "y_true": "Ground truth (correct) labels. Expected as a NumPy array, list, or PyTorch tensor.",
        "y_pred": "Predicted labels. Expected as a NumPy array, list, or PyTorch tensor.",
        "normalize": "If True, return the fraction of correctly classified samples; if False, return the count.",
        "sample_weights": "Optional array of weights for each sample. If None, all samples are weighted equally.",
        "force_finite": "If True, forces the result to be finite even if there are issues like division by zero.",
        "check_outliers": "If True, checks for outliers in y_true using Z-score.",
        "check_distribution": "If True, checks if y_true follows a specific distribution (e.g., normal).",
        "check_correlation": "If True, checks for high correlation between y_true and y_pred.",
        "check_missing_large": "If True, checks for missing values in large datasets using sampling.",
        "check_class_balance": "If True, checks for class imbalance in y_true.",
        "sample_size": "Number of samples to use for large dataset checks.",
        "handle_empty": "Specifies how to handle empty input arrays ('raise', 'warn', or 'ignore')."
    },
    "precision_score": {
        "y_true": "Ground truth (correct) labels. Expected as a NumPy array, list, or PyTorch tensor.",
        "y_pred": "Predicted labels. Expected as a NumPy array, list, or PyTorch tensor.",
        "average": "Method for averaging scores ('binary', 'micro', 'macro', 'weighted', or None).",
        "sample_weights": "Optional array of weights for each sample. If None, all samples are weighted equally.",
        "force_finite": "If True, forces the result to be finite even if there are issues like division by zero.",
        "check_outliers": "If True, checks for outliers in y_true using Z-score.",
        "check_distribution": "If True, checks if y_true follows a specific distribution (e.g., normal).",
        "check_correlation": "If True, checks for high correlation between y_true and y_pred.",
        "check_missing_large": "If True, checks for missing values in large datasets using sampling.",
        "check_class_balance": "If True, checks for class imbalance in y_true.",
        "sample_size": "Number of samples to use for large dataset checks.",
        "handle_empty": "Specifies how to handle empty input arrays ('raise', 'warn', or 'ignore')."
    },
    "recall_score": {
        "y_true": "Ground truth (correct) labels. Expected as a NumPy array, list, or PyTorch tensor.",
        "y_pred": "Predicted labels. Expected as a NumPy array, list, or PyTorch tensor.",
        "average": "Method for averaging scores ('binary', 'micro', 'macro', 'weighted', or None).",
        "sample_weights": "Optional array of weights for each sample. If None, all samples are weighted equally.",
        "force_finite": "If True, forces the result to be finite even if there are issues like division by zero.",
        "check_outliers": "If True, checks for outliers in y_true using Z-score.",
        "check_distribution": "If True, checks if y_true follows a specific distribution (e.g., normal).",
        "check_correlation": "If True, checks for high correlation between y_true and y_pred.",
        "check_missing_large": "If True, checks for missing values in large datasets using sampling.",
        "check_class_balance": "If True, checks for class imbalance in y_true.",
        "sample_size": "Number of samples to use for large dataset checks.",
        "handle_empty": "Specifies how to handle empty input arrays ('raise', 'warn', or 'ignore')."
    },
    "f1_score": {
        "y_true": "Ground truth (correct) labels. Expected as a NumPy array, list, or PyTorch tensor.",
        "y_pred": "Predicted labels. Expected as a NumPy array, list, or PyTorch tensor.",
        "average": "Method for averaging scores ('binary', 'micro', 'macro', 'weighted', or None).",
        "sample_weights": "Optional array of weights for each sample. If None, all samples are weighted equally.",
        "force_finite": "If True, forces the result to be finite even if there are issues like division by zero.",
        "check_outliers": "If True, checks for outliers in y_true using Z-score.",
        "check_distribution": "If True, checks if y_true follows a specific distribution (e.g., normal).",
        "check_correlation": "If True, checks for high correlation between y_true and y_pred.",
        "check_missing_large": "If True, checks for missing values in large datasets using sampling.",
        "check_class_balance": "If True, checks for class imbalance in y_true.",
        "sample_size": "Number of samples to use for large dataset checks.",
        "handle_empty": "Specifies how to handle empty input arrays ('raise', 'warn', or 'ignore')."
    },
    "balanced_accuracy": {
        "y_true": "Ground truth (correct) labels. Expected as a NumPy array, list, or PyTorch tensor.",
        "y_pred": "Predicted labels. Expected as a NumPy array, list, or PyTorch tensor.",
        "sample_weights": "Optional array of weights for each sample. If None, all samples are weighted equally.",
        "adjusted": "If True, adjusts the accuracy score for chance.",
        "force_finite": "If True, forces the result to be finite even if there are issues like division by zero.",
        "check_outliers": "If True, checks for outliers in y_true using Z-score.",
        "check_distribution": "If True, checks if y_true follows a specific distribution (e.g., normal).",
        "check_correlation": "If True, checks for high correlation between y_true and y_pred.",
        "check_missing_large": "If True, checks for missing values in large datasets using sampling.",
        "check_class_balance": "If True, checks for class imbalance in y_true.",
        "sample_size": "Number of samples to use for large dataset checks.",
        "handle_empty": "Specifies how to handle empty input arrays ('raise', 'warn', or 'ignore')."
    },
    "matthews_correlation_coefficient": {
        "y_true": "Ground truth (correct) labels. Expected as a NumPy array, list, or PyTorch tensor.",
        "y_pred": "Predicted labels. Expected as a NumPy array, list, or PyTorch tensor.",
        "sample_weights": "Optional array of weights for each sample. If None, all samples are weighted equally.",
        "average": "Method for averaging scores ('binary', 'micro', 'macro', 'weighted', or None).",
        "force_finite": "If True, forces the result to be finite even if there are issues like division by zero.",
        "check_outliers": "If True, checks for outliers in y_true using Z-score.",
        "check_distribution": "If True, checks if y_true follows a specific distribution (e.g., normal).",
        "check_correlation": "If True, checks for high correlation between y_true and y_pred.",
        "check_missing_large": "If True, checks for missing values in large datasets using sampling.",
        "check_class_balance": "If True, checks for class imbalance in y_true.",
        "sample_size": "Number of samples to use for large dataset checks.",
        "handle_empty": "Specifies how to handle empty input arrays ('raise', 'warn', or 'ignore')."
    },
    "cohens_kappa": {
        "y_true": "Ground truth (correct) labels. Expected as a NumPy array, list, or PyTorch tensor.",
        "y_pred": "Predicted labels. Expected as a NumPy array, list, or PyTorch tensor.",
        "sample_weights": "Optional array of weights for each sample. If None, all samples are weighted equally.",
        "force_finite": "If True, forces the result to be finite even if there are issues like division by zero.",
        "check_outliers": "If True, checks for outliers in y_true using Z-score.",
        "check_distribution": "If True, checks if y_true follows a specific distribution (e.g., normal).",
        "check_correlation": "If True, checks for high correlation between y_true and y_pred.",
        "check_missing_large": "If True, checks for missing values in large datasets using sampling.",
        "check_class_balance": "If True, checks for class imbalance in y_true.",
        "sample_size": "Number of samples to use for large dataset checks.",
        "handle_empty": "Specifies how to handle empty input arrays ('raise', 'warn', or 'ignore')."
    },
    "fbeta_score": {
        "y_true": "Ground truth (correct) labels. Expected as a NumPy array, list, or PyTorch tensor.",
        "y_pred": "Predicted labels. Expected as a NumPy array, list, or PyTorch tensor.",
        "beta": "Weight of precision in harmonic mean.",
        "average": "Method for averaging scores ('binary', 'micro', 'macro', 'weighted', or None).",
        "sample_weights": "Optional array of weights for each sample. If None, all samples are weighted equally.",
        "force_finite": "If True, forces the result to be finite even if there are issues like division by zero.",
        "check_outliers": "If True, checks for outliers in y_true using Z-score.",
        "check_distribution": "If True, checks if y_true follows a specific distribution (e.g., normal).",
        "check_correlation": "If True, checks for high correlation between y_true and y_pred.",
        "check_missing_large": "If True, checks for missing values in large datasets using sampling.",
        "check_class_balance": "If True, checks for class imbalance in y_true.",
        "sample_size": "Number of samples to use for large dataset checks.",
        "handle_empty": "Specifies how to handle empty input arrays ('raise', 'warn', or 'ignore')."
    },
    "jaccard_score": {
        "y_true": "Ground truth (correct) labels. Expected as a NumPy array, list, or PyTorch tensor.",
        "y_pred": "Predicted labels. Expected as a NumPy array, list, or PyTorch tensor.",
        "average": "Method for averaging scores ('binary', 'micro', 'macro', 'weighted', or None).",
        "sample_weights": "Optional array of weights for each sample. If None, all samples are weighted equally.",
        "force_finite": "If True, forces the result to be finite even if there are issues like division by zero.",
        "check_outliers": "If True, checks for outliers in y_true using Z-score.",
        "check_distribution": "If True, checks if y_true follows a specific distribution (e.g., normal).",
        "check_correlation": "If True, checks for high correlation between y_true and y_pred.",
        "check_missing_large": "If True, checks for missing values in large datasets using sampling.",
        "check_class_balance": "If True, checks for class imbalance in y_true.",
        "sample_size": "Number of samples to use for large dataset checks.",
        "handle_empty": "Specifies how to handle empty input arrays ('raise', 'warn', or 'ignore')."
    },
    "hamming_loss": {
        "y_true": "Ground truth (correct) labels. Expected as a NumPy array, list, or PyTorch tensor.",
        "y_pred": "Predicted labels. Expected as a NumPy array, list, or PyTorch tensor.",
        "sample_weights": "Optional array of weights for each sample. If None, all samples are weighted equally.",
        "force_finite": "If True, forces the result to be finite even if there are issues like division by zero.",
        "check_outliers": "If True, checks for outliers in y_true using Z-score.",
        "check_distribution": "If True, checks if y_true follows a specific distribution (e.g., normal).",
        "check_correlation": "If True, checks for high correlation between y_true and y_pred.",
        "check_missing_large": "If True, checks for missing values in large datasets using sampling.",
        "check_class_balance": "If True, checks for class imbalance in y_true.",
        "sample_size": "Number of samples to use for large dataset checks."
    },
    "log_loss": {
        "y_true": "Ground truth (correct) labels. Expected as a NumPy array, list, or PyTorch tensor.",
        "y_pred_proba": "Predicted probabilities for each class. Expected as a 2D array (samples x classes).",
        "eps": "Small value added to avoid taking the log of zero.",
        "normalize": "If True, return the average loss; if False, return the sum of losses.",
        "sample_weights": "Optional array of weights for each sample. If None, all samples are weighted equally.",
        "force_finite": "If True, forces the result to be finite even if there are issues like division by zero.",
        "check_outliers": "If True, checks for outliers in y_true using Z-score.",
        "check_distribution": "If True, checks if y_true follows a specific distribution (e.g., normal).",
        "check_correlation": "If True, checks for high correlation between y_true validate_missing_metrics validation: Union Validation: np_validation: Union __metrics",
        "check_missing_large" : "Optional, If True, check for missing values in large data using sampling.",
        "check_class_balance": "If True, checks for class imbalance in y_true.",
        "sample_size": "Number of samples to use for large dataset checks."
    },
    "contusion_matrix": {
        "y_true": "Ground truth (correct) labels. Expected as a NumPy array, list, or PyTorch tensor.",
        "y_pred_proba": "Predicted probabilities for each class. Expected as a 2D array (samples x classes).",
        "sample_weights": "Optional array of weights for each sample. If None, all samples are weighted equally.",
        "force_finite": "If True, forces the result to be finite even if there are issues like division by zero.",
        "check_outliers": "If True, checks for outliers in y_true using Z-score.",
        "check_distribution": "If True, checks if y_true follows a specific distribution (e.g., normal).",
        "check_correlation": "If True, checks for high correlation between y_true validate_missing_metrics validation: Union Validation: np_validation: Union __metrics",
        "check_missing_large" : "Optional, If True, check for missing values in large data using sampling.",
        "check_class_balance": "If True, checks for class imbalance in y_true.",
        "sample_size": "Number of samples to use for large dataset checks."
    },
    "top_k_accuracy": {
        "y_true": "Ground truth (correct) labels. Expected as a NumPy array, list, or PyTorch tensor.",
        "y_pred_proba": "Predicted probabilities for each class. Expected as a 2D array (samples x classes).",
        "k": "int, optional. Number of top predictions to consider.",
        "normalize": "If True, return the average loss; if False, return the sum of losses.",
        "sample_weights": "Optional array of weights for each sample. If None, all samples are weighted equally.",
        "force_finite": "If True, forces the result to be finite even if there are issues like division by zero.",
        "check_outliers": "If True, checks for outliers in y_true using Z-score.",
        "check_distribution": "If True, checks if y_true follows a specific distribution (e.g., normal).",
        "check_correlation": "If True, checks for high correlation between y_true validate_missing_metrics validation: Union Validation: np_validation: Union __metrics",
        "check_missing_large" : "Optional, If True, check for missing values in large data using sampling.",
        "check_class_balance": "If True, checks for class imbalance in y_true.",
        "sample_size": "Number of samples to use for large dataset checks."
    },
}
def get_metric_details(metric_name):
    """
    Returns the parameter descriptions for a given metric.
    """
    if metric_name in CLASSIFICATION_METRIC_DETAILS:
        return CLASSIFICATION_METRIC_DETAILS[metric_name]
    else:
        raise ValueError(f"Metric '{metric_name}' not found in the image-to-image translation module.")

