

# ClusteringDataValidator
"""

import numpy as np
from scipy.stats import kstest
from scipy.spatial.distance import cdist
class ClusteringDataValidator:
    def __init__(self, raise_warning=True):
        """Initialize the ClusteringDataValidator class"""
        self.raise_warning = raise_warning

    def check_data_type(self, X, labels=None):
        """
        Check if input data types are valid.
        """
        valid_types = (np.ndarray, list)
        if not isinstance(X, valid_types):
            raise TypeError("X must be a numpy array or list")

        # Convert X to numpy array if it's a list
        if isinstance(X, list):
            X = np.array(X)

        if labels is not None:
            if not isinstance(labels, valid_types):
                raise TypeError("labels must be a numpy array or list")

    def check_missing_values(self, X, labels=None):
        """
        Check for missing values in X and labels.
        """
        if np.any(np.isnan(X)):
            raise ValueError("Missing values (NaN) detected in X")

        if labels is not None:
            if np.any(np.isnan(labels)):
                raise ValueError("Missing values (NaN) detected in labels")

    def check_lengths(self, X, labels=None):
        """
        Check if X and labels have the same length.
        """
        if labels is not None:
            if len(X) != len(labels):
                raise ValueError("X and labels must have the same length")

    def check_feature_dimensions(self, X, min_features=1):
        """
        Check if X has at least the minimum number of features.
        """
        if X.shape[1] < min_features:
            raise ValueError(f"X must have at least {min_features} features")

    def check_outliers(self, X, threshold=3):
        """
        Check for outliers in X using Z-score.
        """
        z_scores = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        if np.any(np.abs(z_scores) > threshold):
            if self.raise_warning:
                print("Warning: Outliers detected in X")
            else:
                raise ValueError("Outliers detected in X")

    def check_distribution(self, X, distribution='normal'):
        """
        Check if X follows a specific distribution.
        """
        if distribution == 'normal':
            for feature in range(X.shape[1]):
                stat, p_value = kstest(X[:, feature], 'norm')
                if p_value < 0.05:
                    if self.raise_warning:
                        print(f"Warning: Feature {feature} does not follow a normal distribution")
                    else:
                        raise ValueError(f"Feature {feature} does not follow a normal distribution")

    def check_correlation(self, X, threshold=0.8):
        """
        Check for high correlation between features in X.
        """
        corr_matrix = np.corrcoef(X, rowvar=False)
        for i in range(corr_matrix.shape[0]):
            for j in range(i + 1, corr_matrix.shape[1]):
                if abs(corr_matrix[i, j]) > threshold:
                    if self.raise_warning:
                        print(f"Warning: High correlation detected between features {i} and {j} (corr={corr_matrix[i, j]})")
                    else:
                        raise ValueError(f"High correlation detected between features {i} and {j} (corr={corr_matrix[i, j]})")

    def check_missing_large_data(self, X, sample_size=1000):
        """
        Check for missing values in large data using sampling.
        """
        indices = np.random.choice(len(X), sample_size, replace=False)
        if np.any(np.isnan(X[indices])):
            if self.raise_warning:
                print("Warning: Missing values (NaN) detected in sampled data")
            else:
                raise ValueError("Missing values (NaN) detected in sampled data")

    def check_cluster_labels(self, cluster_labels):
        """
        Check if cluster_labels contain valid integer values.
        """
        if not np.issubdtype(np.array(cluster_labels).dtype, np.integer):
            raise TypeError("cluster_labels must contain integer values")

    def check_number_of_clusters(self, labels, min_clusters=2, max_clusters=None):
        """
        Check if the number of clusters is within a valid range.
        """
        unique_clusters = np.unique(labels)
        num_clusters = len(unique_clusters)

        if num_clusters < min_clusters:
            raise ValueError(f"Number of clusters ({num_clusters}) is less than the minimum allowed ({min_clusters})")

        if max_clusters is not None and num_clusters > max_clusters:
            raise ValueError(f"Number of clusters ({num_clusters}) exceeds the maximum allowed ({max_clusters})")

    def check_cluster_balance(self, labels, threshold=0.1):
        """
        Check if clusters are balanced.
        """
        cluster_counts = np.bincount(labels)
        min_count = np.min(cluster_counts)
        max_count = np.max(cluster_counts)

        if min_count / max_count < threshold:
            if self.raise_warning:
                print(f"Warning: Cluster imbalance detected (min_count={min_count}, max_count={max_count})")
            else:
                raise ValueError(f"Cluster imbalance detected (min_count={min_count}, max_count={max_count})")

    def check_cluster_separation(self, X, labels, min_distance=0.1):
        """
        Check if clusters are well-separated based on pairwise distances.
        """
        unique_clusters = np.unique(labels)
        centroids = np.array([np.mean(X[labels == c], axis=0) for c in unique_clusters])
        distances = cdist(centroids, centroids)
        np.fill_diagonal(distances, np.inf)  # Ignore self-distances

        if np.min(distances) < min_distance:
            if self.raise_warning:
                print(f"Warning: Some clusters are too close to each other (min_distance={np.min(distances)})")
            else:
                raise ValueError(f"Some clusters are too close to each other (min_distance={np.min(distances)})")

    def validate_all(self, X, labels=None, check_outliers=False, check_distribution=False,
                     check_correlation=False, check_missing_large=False, min_features=1,
                     min_clusters=2, max_clusters=None, check_balance=False, check_separation=False,
                     min_distance=0.1, sample_size=1000):
        """
        Run all validation checks.
        """
        self.check_data_type(X, labels)
        self.check_missing_values(X, labels)
        self.check_lengths(X, labels)
        self.check_feature_dimensions(X, min_features)

        if labels is not None:
            self.check_cluster_labels(labels)
            self.check_number_of_clusters(labels, min_clusters, max_clusters)
            if check_balance:
                self.check_cluster_balance(labels)
            if check_separation:
                self.check_cluster_separation(X, labels, min_distance)

        if check_outliers:
            self.check_outliers(X)
        if check_distribution:
            self.check_distribution(X)
        if check_correlation:
            self.check_correlation(X)
        if check_missing_large:
            self.check_missing_large_data(X, sample_size)

        return True  # Return True if all checks pass


# Metadata for clustering metrics
CLUSTERING_METRIC_DETAILS = {
    "rand_score": {
        "labels_true": "Ground truth cluster labels. Expected as a NumPy array or list.",
        "labels_pred": "Predicted cluster labels. Expected as a NumPy array or list.",
        "validator": "Optional ClusteringDataValidator instance for input validation.",
        "validator_params": "Additional parameters for the validator."
    },
    "adjusted_rand_score": {
        "labels_true": "Ground truth cluster labels. Expected as a NumPy array or list.",
        "labels_pred": "Predicted cluster labels. Expected as a NumPy array or list.",
        "validator": "Optional ClusteringDataValidator instance for input validation.",
        "validator_params": "Additional parameters for the validator."
    },
    "mutual_info_score": {
        "labels_true": "Ground truth cluster labels. Expected as a NumPy array or list.",
        "labels_pred": "Predicted cluster labels. Expected as a NumPy array or list.",
        "validator": "Optional ClusteringDataValidator instance for input validation.",
        "validator_params": "Additional parameters for the validator."
    },
    "adjusted_mutual_info_score": {
        "labels_true": "Ground truth cluster labels. Expected as a NumPy array or list.",
        "labels_pred": "Predicted cluster labels. Expected as a NumPy array or list.",
        "validator": "Optional ClusteringDataValidator instance for input validation.",
        "validator_params": "Additional parameters for the validator."
    },
    "silhouette_score": {
        "X": "Feature matrix of shape (n_samples, n_features). Expected as a NumPy array.",
        "labels": "Cluster labels. Expected as a NumPy array or list.",
        "metric": "Distance metric to use ('euclidean', 'cosine', etc.). Default is 'euclidean'.",
        "validator": "Optional ClusteringDataValidator instance for input validation.",
        "validator_params": "Additional parameters for the validator."
    },
    "calinski_harabasz_score": {
        "X": "Feature matrix of shape (n_samples, n_features). Expected as a NumPy array.",
        "labels": "Cluster labels. Expected as a NumPy array or list.",
        "validator": "Optional ClusteringDataValidator instance for input validation.",
        "validator_params": "Additional parameters for the validator."
    },
    "davies_bouldin_score": {
        "X": "Feature matrix of shape (n_samples, n_features). Expected as a NumPy array.",
        "labels": "Cluster labels. Expected as a NumPy array or list.",
        "metric": "Distance metric to use ('euclidean', 'cosine', etc.). Default is 'euclidean'.",
        "validator": "Optional ClusteringDataValidator instance for input validation.",
        "validator_params": "Additional parameters for the validator."
    },
    "homogeneity_score": {
        "labels_true": "Ground truth class labels. Expected as a NumPy array or list.",
        "labels_pred": "Cluster labels to evaluate. Expected as a NumPy array or list.",
        "validator": "Optional ClusteringDataValidator instance for input validation.",
        "validator_params": "Additional parameters for the validator."
    },
    "completeness_score": {
        "labels_true": "Ground truth class labels. Expected as a NumPy array or list.",
        "labels_pred": "Cluster labels to evaluate. Expected as a NumPy array or list.",
        "validator": "Optional ClusteringDataValidator instance for input validation.",
        "validator_params": "Additional parameters for the validator."
    },
    "v_measure_score": {
        "labels_true": "Ground truth class labels. Expected as a NumPy array or list.",
        "labels_pred": "Cluster labels to evaluate. Expected as a NumPy array or list.",
        "beta": "Weight for completeness vs homogeneity. Default is 1.0.",
        "validator": "Optional ClusteringDataValidator instance for input validation.",
        "validator_params": "Additional parameters for the validator."
    },
    "fowlkes_mallows_score": {
        "labels_true": "Ground truth class labels. Expected as a NumPy array or list.",
        "labels_pred": "Cluster labels to evaluate. Expected as a NumPy array or list.",
        "validator": "Optional ClusteringDataValidator instance for input validation.",
        "validator_params": "Additional parameters for the validator."
    }
}

def get_metric_details(metric_name):
    """
    Returns the parameter descriptions for a given clustering metric.
    """
    if metric_name in CLUSTERING_METRIC_DETAILS:
        return CLUSTERING_METRIC_DETAILS[metric_name]
    else:
        raise ValueError(f"Metric '{metric_name}' not found in the clustering module.")