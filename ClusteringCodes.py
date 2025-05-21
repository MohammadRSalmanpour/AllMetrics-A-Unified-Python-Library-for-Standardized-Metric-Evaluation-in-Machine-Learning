

# rand_score
"""

import numpy as np
from scipy.special import comb
def rand_score(labels_true, labels_pred, validator=None, **validator_params):
    """
    Compute the Rand Score between two clusterings with full data validation.

    Parameters:
    -----------
    labels_true : array-like of shape (n_samples,)
        Ground truth cluster labels
    labels_pred : array-like of shape (n_samples,)
        Predicted cluster labels
    validator : ClusteringDataValidator or None, default=None
        Optional validator instance. If None, creates one with validator_params
    **validator_params : dict
        Parameters to pass to ClusteringDataValidator if creating new instance

    Returns:
    --------
    score : float
        Rand Score between 0.0 and 1.0 (1.0 is perfect agreement)

    Examples:
    --------
    >>> true_labels = [0, 0, 1, 1, 1, 1]
    >>> pred_labels = [0, 0, 1, 1, 2, 2]
    >>> rand_score(true_labels, pred_labels)
    0.8666666666666667
    """
    # Initialize validator if not provided (with invalid parameter handling)
    if validator is None:
        # Transfer only valid parameters to the validator
        valid_validator_params = {k: v for k, v in validator_params.items()
                                if k in ['raise_warning']}  # Only valid parameters
        validator = ClusteringDataValidator(**valid_validator_params)
    # Convert and validate inputs
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

    validator.check_data_type(labels_true, labels_pred)
    validator.check_missing_values(labels_true, labels_pred)
    validator.check_lengths(labels_true, labels_pred)
    validator.check_cluster_labels(labels_true)
    validator.check_cluster_labels(labels_pred)

    n = len(labels_true)
    a = 0  # Pairs in the same cluster in both
    b = 0  # Pairs in different clusters in both

    # Count all pairs
    for i in range(n):
        for j in range(i + 1, n):
            same_true = labels_true[i] == labels_true[j]
            same_pred = labels_pred[i] == labels_pred[j]
            if same_true and same_pred:
                a += 1
            elif not same_true and not same_pred:
                b += 1

    total_pairs = comb(n, 2)
    ri = (a + b) / total_pairs
    return float(ri)

"""# adjusted_rand_index"""

def adjusted_rand_score(labels_true, labels_pred, validator=None, **validator_params):
    """
    Compute the Adjusted Rand Score between two clusterings.

    Parameters:
    -----------
    labels_true : array-like of shape (n_samples,)
        Ground truth cluster labels
    labels_pred : array-like of shape (n_samples,)
        Predicted cluster labels
    validator : ClusteringDataValidator or None, default=None
        Optional validator instance
    **validator_params : dict
        Parameters for validator if creating new instance

    Returns:
    --------
    score : float
        Adjusted Rand Score between -1.0 and 1.0 (1.0 is perfect agreement)
    """
    # Initialize validator if not provided
    if validator is None:
        validator = ClusteringDataValidator(**validator_params)

    # Convert and validate inputs (reuses same validation as rand_score)
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

    validator.check_data_type(labels_true, labels_pred)
    validator.check_missing_values(labels_true, labels_pred)
    validator.check_lengths(labels_true, labels_pred)
    validator.check_cluster_labels(labels_true)
    validator.check_cluster_labels(labels_pred)

    # Create contingency matrix
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    contingency = np.zeros((len(classes), len(clusters)), dtype=np.int64)

    for i, c in enumerate(classes):
        for j, k in enumerate(clusters):
            contingency[i, j] = np.sum((labels_true == c) & (labels_pred == k))

    # Calculate components
    sum_comb_c = sum(comb(n_c, 2) for n_c in contingency.sum(axis=1))
    sum_comb_k = sum(comb(n_k, 2) for n_k in contingency.sum(axis=0))
    sum_comb = sum(comb(n_ij, 2) for n_ij in contingency.ravel())
    total_pairs = comb(len(labels_true), 2)

    # Compute Adjusted Rand Index
    numerator = sum_comb - (sum_comb_c * sum_comb_k) / total_pairs
    denominator = 0.5 * (sum_comb_c + sum_comb_k) - (sum_comb_c * sum_comb_k) / total_pairs

    if denominator == 0:
        return 0.0

    ari = numerator / denominator
    return float(ari)

"""# mutual_info_score"""

import numpy as np
from math import log
from scipy.special import comb
def _validate_inputs(labels_true, labels_pred, validator=None, **validator_params):
    """
    Compute Mutual Information between two clusterings.

    Mutual Information measures the agreement between two clusterings,
    ignoring permutations. Higher values indicate better agreement.

    Parameters:
    -----------
    labels_true : array-like of shape (n_samples,)
        Ground truth cluster labels.
    labels_pred : array-like of shape (n_samples,)
        Predicted cluster labels.
    check_input : bool, optional (default=True)
        Whether to validate input data using ClusteringDataValidator.
    check_outliers : bool, optional (default=False)
        Check for outliers in the data.
    check_distribution : bool, optional (default=False)
        Check if data follows normal distribution.
    check_correlation : bool, optional (default=False)
        Check for high correlation between features.
    check_missing_large : bool, optional (default=False)
        Check for missing values in large datasets using sampling.
    check_cluster_balance : bool, optional (default=False)
        Check if clusters are balanced.
    check_cluster_separation : bool, optional (default=False)
        Check if clusters are well-separated.
    sample_size : int, optional (default=1000)
        Sample size for checking large datasets.
    min_distance : float, optional (default=0.1)
        Minimum acceptable distance between clusters.
    eps : float, optional (default=1e-12)
        Small value to avoid numerical instability.

    Returns:
    --------
    float
        Mutual information score in bits.
    """
    if validator is None:
        validator = ClusteringDataValidator(**validator_params)

    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

    validator.check_data_type(labels_true, labels_pred)
    validator.check_missing_values(labels_true, labels_pred)
    validator.check_lengths(labels_true, labels_pred)
    validator.check_cluster_labels(labels_true)
    validator.check_cluster_labels(labels_pred)

    return labels_true, labels_pred

def _contingency_matrix(labels_true, labels_pred):
    """Contingency matrix for mutual information calculations"""
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    contingency = np.zeros((len(classes), len(clusters)), dtype=np.float64)

    for i, c in enumerate(classes):
        for j, k in enumerate(clusters):
            contingency[i, j] = np.sum((labels_true == c) & (labels_pred == k))
    return contingency

def _entropy(labels):
    """Calculate entropy"""
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)
    return -np.sum(probs * np.log(probs + 1e-10))  # +1e-10 for numerical stability

def mutual_info_score(labels_true, labels_pred, validator=None, **validator_params):
    """
    Compute Mutual Information between two clusterings.

    Mutual Information measures the agreement between two clusterings,
    ignoring permutations. Higher values indicate better agreement.

    Parameters:
    -----------
    labels_true : array-like of shape (n_samples,)
        Ground truth cluster labels.
    labels_pred : array-like of shape (n_samples,)
        Predicted cluster labels.
    check_input : bool, optional (default=True)
        Whether to validate input data using ClusteringDataValidator.
    check_outliers : bool, optional (default=False)
        Check for outliers in the data.
    check_distribution : bool, optional (default=False)
        Check if data follows normal distribution.
    check_correlation : bool, optional (default=False)
        Check for high correlation between features.
    check_missing_large : bool, optional (default=False)
        Check for missing values in large datasets using sampling.
    check_cluster_balance : bool, optional (default=False)
        Check if clusters are balanced.
    check_cluster_separation : bool, optional (default=False)
        Check if clusters are well-separated.
    sample_size : int, optional (default=1000)
        Sample size for checking large datasets.
    min_distance : float, optional (default=0.1)
        Minimum acceptable distance between clusters.
    eps : float, optional (default=1e-12)
        Small value to avoid numerical instability.

    Returns:
    --------
    float
        Mutual information score in bits.
    """
    labels_true, labels_pred = _validate_inputs(labels_true, labels_pred, validator, **validator_params)
    contingency = _contingency_matrix(labels_true, labels_pred)
    n_samples = len(labels_true)

    # Calculate probabilities
    pxy = contingency / n_samples
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)

    # Calculate MI
    mi = 0.0
    for i in range(contingency.shape[0]):
        for j in range(contingency.shape[1]):
            if pxy[i, j] > 0:
                mi += pxy[i, j] * log(pxy[i, j] / (px[i] * py[j]))

    return max(mi, 0.0)

"""# normalized_mutual_info_score"""

def normalized_mutual_info_score(labels_true, labels_pred, validator=None, **validator_params):
    """
    Compute normalized version of Mutual Information (NMI)
    """
    labels_true, labels_pred = _validate_inputs(labels_true, labels_pred, validator, **validator_params)
    mi = mutual_info_score(labels_true, labels_pred)
    h_true = _entropy(labels_true)
    h_pred = _entropy(labels_pred)

    # Special cases
    if h_true == 0 and h_pred == 0:
        return 1.0
    if h_true == 0 or h_pred == 0:
        return 0.0

    return mi / max(np.sqrt(h_true * h_pred), 1e-10)

"""# silhouette_score"""

def silhouette_score(X, labels, validator=None, metric='euclidean', **validator_params):
    """
    Compute Silhouette Coefficient with full validation

    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Input feature matrix
    labels : array-like of shape (n_samples,)
        Cluster labels
    validator : ClusteringDataValidator or None
        Validator instance
    metric : str, default='euclidean'
        Distance metric ('euclidean', 'cosine', etc.)
    **validator_params : dict
        Validator parameters

    Returns:
    --------
    silhouette : float
        Silhouette score between -1 and 1
    """
    # Initialize validator
    if validator is None:
        validator = ClusteringDataValidator(**validator_params)

    # Convert and validate inputs
    X = np.asarray(X)
    labels = np.asarray(labels)

    validator.check_data_type(X, labels)
    validator.check_missing_values(X, labels)
    validator.check_lengths(X, labels)
    validator.check_cluster_labels(labels)
    validator.check_feature_dimensions(X)

    # Get unique labels and check cluster count
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    if n_clusters == 1:
        return 0.0  # All samples in single cluster

    # Compute distance matrix
    distance_matrix = cdist(X, X, metric=metric)

    # Initialize arrays for a and b
    a = np.zeros(X.shape[0])
    b = np.zeros(X.shape[0])

    for i in range(X.shape[0]):
        # Intra-cluster distance (a)
        cluster_mask = labels == labels[i]
        cluster_mask[i] = False  # Exclude self
        if np.any(cluster_mask):
            a[i] = np.mean(distance_matrix[i, cluster_mask])
        else:
            a[i] = 0.0  # Single point in cluster

        # Inter-cluster distances (b)
        other_clusters = [l for l in unique_labels if l != labels[i]]
        b_vals = []

        for cluster in other_clusters:
            other_mask = labels == cluster
            b_vals.append(np.mean(distance_matrix[i, other_mask]))

        b[i] = np.min(b_vals) if b_vals else 0.0

    # Compute silhouette scores
    silhouette_scores = np.where(
        (a == 0) & (b == 0),
        0.0,  # Handle 0/0 case
        (b - a) / np.maximum(a, b)
    )

    return float(np.mean(silhouette_scores))

"""# calinski_harabasz_index"""

def calinski_harabasz_score(X, labels, validator=None, **validator_params):
    """
    Compute the Calinski-Harabasz Index with full data validation.

    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix
    labels : array-like of shape (n_samples,)
        Cluster labels
    validator : ClusteringDataValidator or None, default=None
        Optional validator instance
    **validator_params : dict
        Parameters for validator if creating new instance

    Returns:
    --------
    ch_score : float
        Calinski-Harabasz Index (higher is better)

    Examples:
    --------
    >>> from sklearn.datasets import make_blobs
    >>> X, y = make_blobs(n_samples=500, centers=3)
    >>> calinski_harabasz_score(X, y)
    423.45
    """
    # Initialize validator if not provided
    if validator is None:
        # We just transfer the valid parameteres to validator
        valid_validator_params = {k: v for k, v in validator_params.items()
                                if k in ['raise_warning']}  # just valid parameteres
        validator = ClusteringDataValidator(**valid_validator_params)
    # Convert inputs
    X = np.asarray(X)
    labels = np.asarray(labels)

    # Validate inputs
    validator.check_data_type(X, labels)
    validator.check_missing_values(X, labels)
    validator.check_lengths(X, labels)
    validator.check_cluster_labels(labels)
    validator.check_feature_dimensions(X)

    # Get unique labels and check cluster count
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    n_samples = X.shape[0]

    if n_clusters == 1:
        return 0.0  # Edge case when all samples are in one cluster

    # Calculate overall centroid
    overall_centroid = np.mean(X, axis=0)

    # Calculate between-cluster dispersion (BGSS)
    bgss = 0.0
    cluster_sizes = []
    cluster_centroids = []

    for label in unique_labels:
        cluster_points = X[labels == label]
        cluster_size = len(cluster_points)
        cluster_centroid = np.mean(cluster_points, axis=0)

        bgss += cluster_size * np.sum((cluster_centroid - overall_centroid) ** 2)
        cluster_sizes.append(cluster_size)
        cluster_centroids.append(cluster_centroid)

    # Calculate within-cluster dispersion (WGSS)
    wgss = 0.0
    for i, label in enumerate(unique_labels):
        cluster_points = X[labels == label]
        wgss += np.sum(cdist(cluster_points, [cluster_centroids[i]], 'sqeuclidean'))

    # Compute Calinski-Harabasz Index
    ch_score = (bgss / (n_clusters - 1)) / (wgss / (n_samples - n_clusters))

    return float(ch_score)

"""# davies_bouldin_index"""

def davies_bouldin_score(X, labels, validator=None, metric='euclidean', **validator_params):
    """
    Compute Davies-Bouldin Index with full data validation.

    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix
    labels : array-like of shape (n_samples,)
        Cluster labels
    validator : ClusteringDataValidator or None, default=None
        Optional validator instance
    metric : str or callable, default='euclidean'
        Distance metric to use
    **validator_params : dict
        Parameters for validator if creating new instance

    Returns:
    --------
    db_score : float
        Davies-Bouldin Index (lower is better)

    Examples:
    --------
    >>> from sklearn.datasets import make_blobs
    >>> X, y = make_blobs(n_samples=500, centers=3)
    >>> davies_bouldin_score(X, y)
    0.356
    """
    # Initialize validator if not provided
    if validator is None:
        valid_validator_params = {k: v for k, v in validator_params.items()
                                if k in ['raise_warning']}
        validator = ClusteringDataValidator(**valid_validator_params)
    # Convert inputs
    X = np.asarray(X)
    labels = np.asarray(labels)

    # Validate inputs
    validator.check_data_type(X, labels)
    validator.check_missing_values(X, labels)
    validator.check_lengths(X, labels)
    validator.check_cluster_labels(labels)
    validator.check_feature_dimensions(X)

    # Get unique labels and check cluster count
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters == 1:
        return 0.0  # Perfect score when all in one cluster

    # Calculate cluster centroids and diameters
    centroids = []
    diameters = []

    for label in unique_labels:
        cluster_points = X[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)

        # Calculate intra-cluster distances
        if len(cluster_points) > 1:
            intra_distances = cdist(cluster_points, [centroid], metric=metric)
            diameter = np.mean(intra_distances)
        else:
            diameter = 0.0  # Single point cluster
        diameters.append(diameter)

    centroids = np.array(centroids)
    diameters = np.array(diameters)

    # Compute pairwise similarity matrix
    db_matrix = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            if i != j:
                # Distance between centroids
                centroid_distance = cdist([centroids[i]], [centroids[j]], metric=metric)[0][0]
                # Similarity measure
                db_matrix[i,j] = (diameters[i] + diameters[j]) / centroid_distance

    # Compute Davies-Bouldin Index
    db_score = np.mean(np.max(db_matrix, axis=1))
    return float(db_score)

"""# homogeneity_score"""

def homogeneity_score(labels_true, labels_pred, validator=None, **validator_params):
    """
    Compute the Homogeneity Score with full data validation.

    A clustering result satisfies homogeneity if all of its clusters
    contain only data points which are members of a single class.

    Parameters:
    -----------
    labels_true : array-like of shape (n_samples,)
        Ground truth class labels
    labels_pred : array-like of shape (n_samples,)
        Cluster labels to evaluate
    validator : ClusteringDataValidator or None, default=None
        Optional validator instance
    **validator_params : dict
        Parameters for validator if creating new instance

    Returns:
    --------
    homogeneity : float
        Score between 0.0 and 1.0 (1.0 stands for perfect homogeneity)

    Examples:
    --------
    >>> labels_true = [0, 0, 1, 1]
    >>> labels_pred = [1, 1, 0, 0]
    >>> homogeneity_score(labels_true, labels_pred)
    1.0
    """
    # Initialize validator if not provided
    if validator is None:
        validator = ClusteringDataValidator(**validator_params)

    # Convert and validate inputs
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

    validator.check_data_type(labels_true, labels_pred)
    validator.check_missing_values(labels_true, labels_pred)
    validator.check_lengths(labels_true, labels_pred)
    validator.check_cluster_labels(labels_true)
    validator.check_cluster_labels(labels_pred)

    # Check for edge cases
    if len(labels_true) == 0:
        return 1.0

    # Create contingency matrix
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    contingency = np.zeros((len(classes), len(clusters)), dtype=np.float64)

    for i, c in enumerate(classes):
        for j, k in enumerate(clusters):
            contingency[i, j] = np.sum((labels_true == c) & (labels_pred == k))

    # Calculate entropy terms
    h_c_given_k = 0.0  # Conditional entropy of classes given clusters
    h_c = 0.0          # Entropy of classes

    cluster_sizes = np.sum(contingency, axis=0)
    class_sizes = np.sum(contingency, axis=1)
    n_samples = len(labels_true)

    # Calculate H(C|K)
    for j in range(len(clusters)):
        cluster_size = cluster_sizes[j]
        if cluster_size == 0:
            continue
        cluster_entropy = 0.0
        for i in range(len(classes)):
            p_ij = contingency[i, j] / cluster_size
            if p_ij > 0:
                cluster_entropy -= p_ij * np.log(p_ij)
        h_c_given_k += (cluster_size / n_samples) * cluster_entropy

    # Calculate H(C)
    for i in range(len(classes)):
        p_i = class_sizes[i] / n_samples
        if p_i > 0:
            h_c -= p_i * np.log(p_i)

    # Handle perfect homogeneity case
    if h_c == 0:
        return 1.0

    return 1.0 - (h_c_given_k / h_c)

"""# completeness_score"""

def completeness_score(labels_true, labels_pred, validator=None, **validator_params):
    """
    Compute the Completeness Score (dual to homogeneity).

    A clustering result satisfies completeness if all members of a given class
    are assigned to the same cluster.
    """
    # Simply reverse the arguments for completeness
    return homogeneity_score(labels_pred, labels_true, validator, **validator_params)

"""# v_measure_score"""

def v_measure_score(labels_true, labels_pred, beta=1.0, validator=None, **validator_params):
    """
    Compute V-measure (harmonic mean of homogeneity and completeness).

    Parameters:
    -----------
    beta : float, default=1.0
        Weight for completeness vs homogeneity
    """
    h = homogeneity_score(labels_true, labels_pred, validator, **validator_params)
    c = completeness_score(labels_true, labels_pred, validator, **validator_params)

    if (h + c) == 0:
        return 0.0

    return (1 + beta) * h * c / (beta * h + c)

"""# fowlkes_mallows_score"""

def fowlkes_mallows_score(labels_true, labels_pred, validator=None, **validator_params):
    """
    Compute the Fowlkes-Mallows Index with full data validation.

    Parameters:
    -----------
    labels_true : array-like of shape (n_samples,)
        Ground truth class labels
    labels_pred : array-like of shape (n_samples,)
        Cluster labels to evaluate
    validator : ClusteringDataValidator or None, default=None
        Optional validator instance
    **validator_params : dict
        Parameters for validator if creating new instance

    Returns:
    --------
    fmi : float
        Fowlkes-Mallows Index between 0.0 and 1.0 (1.0 stands for perfect matching)

    Examples:
    --------
    >>> labels_true = [0, 0, 1, 1]
    >>> labels_pred = [1, 1, 0, 0]
    >>> fowlkes_mallows_score(labels_true, labels_pred)
    1.0
    """
    # Initialize validator if not provided
    if validator is None:
        validator = ClusteringDataValidator(**validator_params)

    # Convert and validate inputs
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

    validator.check_data_type(labels_true, labels_pred)
    validator.check_missing_values(labels_true, labels_pred)
    validator.check_lengths(labels_true, labels_pred)
    validator.check_cluster_labels(labels_true)
    validator.check_cluster_labels(labels_pred)

    n_samples = len(labels_true)

    # Calculate contingency matrix
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    contingency = np.zeros((len(classes), len(clusters)), dtype=np.int64)

    for i, c in enumerate(classes):
        for j, k in enumerate(clusters):
            contingency[i, j] = np.sum((labels_true == c) & (labels_pred == k))

    # Calculate TP, FP, FN
    tp_plus_fp = sum(comb(n_c, 2) for n_c in np.ravel(contingency.sum(axis=0)))
    tp_plus_fn = sum(comb(n_k, 2) for n_k in np.ravel(contingency.sum(axis=1)))
    tp = sum(comb(n_ij, 2) for n_ij in np.ravel(contingency))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp

    # Handle edge cases
    if tp == 0:
        return 0.0

    # Calculate FMI
    fmi = tp / np.sqrt((tp + fp) * (tp + fn))
    return float(fmi)