
# accuracy_score
"""

import numpy as np
import warnings

def accuracy_score(
    y_true,
    y_pred,
    normalize=True,
    sample_weights=None,
    force_finite=False,
    check_outliers=False,
    check_distribution=False,
    check_correlation=False,
    check_missing_large=False,
    check_class_balance=False,
    sample_size=1000,
    handle_empty='raise'
):
    """
    Compute Accuracy Score with robust validation.

    Accuracy measures the proportion of correctly classified samples.

    Parameters:
    -----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated target values.
    normalize : bool, optional
        If True, return fraction correct. If False, return count correct.
    sample_weights : array-like, optional
        Sample weights for weighted accuracy.
    force_finite : bool, optional
        If True, handles infinite values by conversion/removal.
    check_outliers : bool, optional
        Check for outliers in y_true.
    check_distribution : bool, optional
        Check if y_true follows normal distribution.
    check_correlation : bool, optional
        Check correlation between y_true and y_pred.
    check_missing_large : bool, optional
        Check for missing values in large datasets.
    check_class_balance : bool, optional
        Check for class imbalance.
    sample_size : int, optional
        Sample size for large dataset checks.
    handle_empty : str, optional
        How to handle empty inputs: 'raise', 'warn', 'ignore'

    Returns:
    --------
    float or int
        Accuracy score (fraction if normalize=True, count otherwise)
    """
    # Convert inputs to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Handle empty arrays
    if len(y_true) == 0 or len(y_pred) == 0:
        if handle_empty == 'raise':
            raise ValueError("Input arrays cannot be empty")
        elif handle_empty == 'warn':
            warnings.warn("Input arrays are empty - returning 0")
            return 0.0 if normalize else 0
        else:
            return 0.0 if normalize else 0

    # Data validation
    validator = ClassificationDataValidator()
    validator.validate_all(
        y_true,
        y_pred,
        check_outliers=check_outliers,
        check_distribution=check_distribution,
        check_correlation=check_correlation,
        check_missing_large=check_missing_large,
        check_class_balance=check_class_balance,
        sample_size=sample_size
    )

    # Handle infinite values
    if force_finite:
        y_true = np.nan_to_num(y_true, nan=np.nan, posinf=np.nan, neginf=np.nan)
        y_pred = np.nan_to_num(y_pred, nan=np.nan, posinf=np.nan, neginf=np.nan)
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        if len(y_true) == 0:
            return 0.0 if normalize else 0

    # Calculate correct predictions
    correct = (y_true == y_pred)

    # Apply sample weights if provided
    if sample_weights is not None:
        sample_weights = np.array(sample_weights)
        if len(sample_weights) != len(y_true):
            raise ValueError("sample_weights must have same length as inputs")
        correct = correct.astype(float) * sample_weights

    # Calculate accuracy
    if normalize:
        if sample_weights is not None:
            return np.sum(correct) / np.sum(sample_weights)
        return np.mean(correct)
    else:
        return np.sum(correct)

"""# precision_score"""

import numpy as np
import warnings

def precision_score(
    y_true,
    y_pred,
    *,
    average='none',
    sample_weights=None,
    force_finite=False,
    check_outliers=False,
    check_distribution=False,
    check_correlation=False,
    check_missing_large=False,
    check_class_balance=False,
    sample_size=1000,
    handle_empty='raise'
):
    """
    Complete precision_score implementation with validator and proper weight handling.
    """
    # Convert all inputs to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights)

    # Handle empty arrays
    if len(y_true) == 0 or len(y_pred) == 0:
        if handle_empty == 'raise':
            raise ValueError("Input arrays cannot be empty")
        elif handle_empty == 'warn':
            warnings.warn("Input arrays are empty - returning 0")
            return 0.0
        else:
            return 0.0

    # Data validation - THIS IS THE IMPORTANT PART THAT WAS REMOVED
    validator = ClassificationDataValidator()
    validator.validate_all(
        y_true,
        y_pred,
        check_outliers=check_outliers,
        check_distribution=check_distribution,
        check_correlation=check_correlation,
        check_missing_large=check_missing_large,
        check_class_balance=check_class_balance,
        sample_size=sample_size
    )

    # Handle infinite values
    if force_finite:
        y_true = np.nan_to_num(y_true, nan=np.nan, posinf=np.nan, neginf=np.nan)
        y_pred = np.nan_to_num(y_pred, nan=np.nan, posinf=np.nan, neginf=np.nan)
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        if sample_weights is not None:
            sample_weights = sample_weights[mask]
        if len(y_true) == 0:
            return 0.0

    # Get unique classes
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(unique_classes)

    # Initialize precision array
    precisions = np.zeros(n_classes)
    class_counts = np.zeros(n_classes)

    # Calculate precision for each class
    for i, cls in enumerate(unique_classes):
        true_positives = (y_true == cls) & (y_pred == cls)
        false_positives = (y_true != cls) & (y_pred == cls)

        if sample_weights is not None:
            tp = np.sum(sample_weights[true_positives])
            fp = np.sum(sample_weights[false_positives])
            class_counts[i] = np.sum(sample_weights[y_true == cls])
        else:
            tp = np.sum(true_positives)
            fp = np.sum(false_positives)
            class_counts[i] = np.sum(y_true == cls)

        denominator = tp + fp
        precisions[i] = tp / denominator if denominator > 0 else 0.0

    # Handle averaging
    if average == 'binary':
        if n_classes == 1:
            # Special case: only one class exists (all predictions correct by definition)
            return 1.0
        elif n_classes != 2:
            raise ValueError("Binary recall requires exactly 2 classes")
        return precisions[1]
    elif average == 'micro':
        if sample_weights is not None:
            tp_total = np.sum(sample_weights[y_true == y_pred])
            fp_total = np.sum(sample_weights[y_true != y_pred])
        else:
            tp_total = np.sum(y_true == y_pred)
            fp_total = np.sum(y_true != y_pred)
        denominator = tp_total + fp_total
        return tp_total / denominator if denominator > 0 else 0.0
    elif average == 'macro':
        return np.mean(precisions)
    elif average == 'weighted':
        if np.sum(class_counts) == 0:
            return 0.0
        return np.sum(precisions * class_counts) / np.sum(class_counts)
    elif average == 'none':
        return precisions
    else:
        raise ValueError("Invalid average option. Choose from: 'binary', 'micro', 'macro', 'weighted', 'none'")

"""# recall_score"""

def recall_score(
    y_true,
    y_pred,
    *,
    average='none',
    sample_weights=None,
    force_finite=False,
    check_outliers=False,
    check_distribution=False,
    check_correlation=False,
    check_missing_large=False,
    check_class_balance=False,
    sample_size=1000,
    handle_empty='raise'
):
    """
    Compute Recall Score with identical validation to precision_score.

    Recall = True Positives / (True Positives + False Negatives)
    """
    # Convert inputs to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights)

    # Handle empty arrays
    if len(y_true) == 0 or len(y_pred) == 0:
        if handle_empty == 'raise':
            raise ValueError("Input arrays cannot be empty")
        elif handle_empty == 'warn':
            warnings.warn("Input arrays are empty - returning 0")
            return 0.0
        else:
            return 0.0

    # Data validation
    validator = ClassificationDataValidator()
    validator.validate_all(
        y_true,
        y_pred,
        check_outliers=check_outliers,
        check_distribution=check_distribution,
        check_correlation=check_correlation,
        check_missing_large=check_missing_large,
        check_class_balance=check_class_balance,
        sample_size=sample_size
    )

    # Handle infinite values
    if force_finite:
        y_true = np.nan_to_num(y_true, nan=np.nan, posinf=np.nan, neginf=np.nan)
        y_pred = np.nan_to_num(y_pred, nan=np.nan, posinf=np.nan, neginf=np.nan)
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        if sample_weights is not None:
            sample_weights = sample_weights[mask]
        if len(y_true) == 0:
            return 0.0

    # Get unique classes
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(unique_classes)

    # Initialize recall array
    recalls = np.zeros(n_classes)
    class_counts = np.zeros(n_classes)

    # Calculate recall for each class
    for i, cls in enumerate(unique_classes):
        true_positives = (y_true == cls) & (y_pred == cls)
        false_negatives = (y_true == cls) & (y_pred != cls)

        if sample_weights is not None:
            tp = np.sum(sample_weights[true_positives])
            fn = np.sum(sample_weights[false_negatives])
            class_counts[i] = np.sum(sample_weights[y_true == cls])
        else:
            tp = np.sum(true_positives)
            fn = np.sum(false_negatives)
            class_counts[i] = np.sum(y_true == cls)

        denominator = tp + fn
        recalls[i] = tp / denominator if denominator > 0 else 0.0

    # Handle averaging
    if average == 'binary':
        if n_classes == 1:
            # Special case: only one class exists (all predictions correct by definition)
            return 1.0
        elif n_classes != 2:
            raise ValueError("Binary recall requires exactly 2 classes")
        return recalls[1]
    elif average == 'micro':
        if sample_weights is not None:
            tp_total = np.sum(sample_weights[y_true == y_pred])
            fn_total = np.sum(sample_weights[y_true != y_pred])
        else:
            tp_total = np.sum(y_true == y_pred)
            fn_total = np.sum(y_true != y_pred)
        denominator = tp_total + fn_total
        return tp_total / denominator if denominator > 0 else 0.0
    elif average == 'macro':
        return np.mean(recalls)
    elif average == 'weighted':
        if np.sum(class_counts) == 0:
            return 0.0
        return np.sum(recalls * class_counts) / np.sum(class_counts)
    elif average == 'none':
        return recalls
    else:
        raise ValueError("Invalid average option. Choose from: 'binary', 'micro', 'macro', 'weighted', 'none'")

"""# f1_score"""

def f1_score(
    y_true,
    y_pred,
    *,
    average='none',
    sample_weights=None,
    force_finite=False,
    check_outliers=False,
    check_distribution=False,
    check_correlation=False,
    check_missing_large=False,
    check_class_balance=False,
    sample_size=1000,
    handle_empty='raise'
):
    """
    Compute F1 Score using precision and recall with identical validation.

    F1 = 2 * (precision * recall) / (precision + recall)
    """
    # Compute precision and recall with identical parameters
    precision = precision_score(
        y_true=y_true,
        y_pred=y_pred,
        average=average,
        sample_weights=sample_weights,
        force_finite=force_finite,
        check_outliers=check_outliers,
        check_distribution=check_distribution,
        check_correlation=check_correlation,
        check_missing_large=check_missing_large,
        check_class_balance=check_class_balance,
        sample_size=sample_size,
        handle_empty=handle_empty
    )

    recall = recall_score(
        y_true=y_true,
        y_pred=y_pred,
        average=average,
        sample_weights=sample_weights,
        force_finite=force_finite,
        check_outliers=check_outliers,
        check_distribution=check_distribution,
        check_correlation=check_correlation,
        check_missing_large=check_missing_large,
        check_class_balance=check_class_balance,
        sample_size=sample_size,
        handle_empty=handle_empty
    )

    # Handle different average cases
    if average == 'none':
        with np.errstate(divide='ignore', invalid='ignore'):
            f1 = 2 * (precision * recall) / (precision + recall)
        f1[np.isnan(f1)] = 0  # Handle division by zero
        return f1
    else:
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

"""# balanced_accuracy"""

import numpy as np
import warnings

def balanced_accuracy(
    y_true,
    y_pred,
    *,
    sample_weights=None,
    adjusted=False,
    force_finite=False,
    check_outliers=False,
    check_distribution=False,
    check_correlation=False,
    check_missing_large=False,
    check_class_balance=False,
    sample_size=1000,
    handle_empty='raise'
):
    """
    Compute Balanced Accuracy with robust validation.

    Balanced Accuracy is the average of recall obtained on each class.
    The adjusted version corrects for chance (ranges from 0 to 1).

    Parameters:
    -----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated target values.
    sample_weights : array-like, optional
        Sample weights for weighted recall calculation.
    adjusted : bool, optional
        If True, returns adjusted balanced accuracy (chance-corrected).
    force_finite : bool, optional
        If True, handles infinite values by conversion/removal.
    check_outliers : bool, optional
        Check for outliers in y_true.
    check_distribution : bool, optional
        Check if y_true follows normal distribution.
    check_correlation : bool, optional
        Check correlation between y_true and y_pred.
    check_missing_large : bool, optional
        Check for missing values in large datasets.
    check_class_balance : bool, optional
        Check for class imbalance.
    sample_size : int, optional
        Sample size for large dataset checks.
    handle_empty : str, optional
        How to handle empty inputs: 'raise', 'warn', 'ignore'

    Returns:
    --------
    float
        Balanced accuracy score (between 0 and 1, or negative infinity to 1 when adjusted)
    """
    # Convert inputs to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights)

    # Handle empty arrays
    if len(y_true) == 0 or len(y_pred) == 0:
        if handle_empty == 'raise':
            raise ValueError("Input arrays cannot be empty")
        elif handle_empty == 'warn':
            warnings.warn("Input arrays are empty - returning 0")
            return 0.0
        else:
            return 0.0

    # Data validation
    validator = ClassificationDataValidator()
    validator.validate_all(
        y_true,
        y_pred,
        check_outliers=check_outliers,
        check_distribution=check_distribution,
        check_correlation=check_correlation,
        check_missing_large=check_missing_large,
        check_class_balance=check_class_balance,
        sample_size=sample_size
    )

    # Handle infinite values
    if force_finite:
        y_true = np.nan_to_num(y_true, nan=np.nan, posinf=np.nan, neginf=np.nan)
        y_pred = np.nan_to_num(y_pred, nan=np.nan, posinf=np.nan, neginf=np.nan)
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        if sample_weights is not None:
            sample_weights = sample_weights[mask]
        if len(y_true) == 0:
            return 0.0

    # Get unique classes
    unique_classes = np.unique(y_true)
    n_classes = len(unique_classes)

    # Calculate recall for each class
    recalls = np.zeros(n_classes)

    for i, cls in enumerate(unique_classes):
        true_positives = (y_true == cls) & (y_pred == cls)
        false_negatives = (y_true == cls) & (y_pred != cls)

        if sample_weights is not None:
            tp = np.sum(sample_weights[true_positives])
            fn = np.sum(sample_weights[false_negatives])
        else:
            tp = np.sum(true_positives)
            fn = np.sum(false_negatives)

        denominator = tp + fn
        recalls[i] = tp / denominator if denominator > 0 else 0.0

    # Calculate balanced accuracy
    balanced_acc = np.mean(recalls)

    # Calculate adjusted version if requested
    if adjusted:
        if n_classes == 1:
            return 0.0  # Adjusted is undefined for single class
        balanced_acc = (balanced_acc - 1/n_classes) / (1 - 1/n_classes)

    return balanced_acc

"""# matthews_correlation_coefficient"""

import numpy as np
import warnings

def matthews_correlation_coefficient(
    y_true,
    y_pred,
    *,
    sample_weights=None,
    average='macro',
    force_finite=False,
    check_outliers=False,
    check_distribution=False,
    check_correlation=False,
    check_missing_large=False,
    check_class_balance=False,
    sample_size=1000,
    handle_empty='raise'
):
    """
    Compute Matthews Correlation Coefficient (MCC) with robust edge case handling.

    MCC is a reliable measure of classification quality that works well even with
    imbalanced classes. It ranges from -1 (total disagreement) to +1 (perfect prediction).

    Parameters:
    -----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated target values.
    sample_weights : array-like, optional
        Sample weights for weighted calculations.
    average : str, optional
        Averaging method: 'macro' (default) or 'weighted'.
    force_finite : bool, optional
        Handle infinite values by conversion/removal.
    check_outliers : bool, optional
        Check for outliers in y_true.
    check_distribution : bool, optional
        Check if y_true follows normal distribution.
    check_correlation : bool, optional
        Check correlation between y_true and y_pred.
    check_missing_large : bool, optional
        Check for missing values in large datasets.
    check_class_balance : bool, optional
        Check if classes are imbalanced.
    sample_size : int, optional
        Sample size for large dataset checks.
    handle_empty : str, optional
        How to handle empty inputs: 'raise', 'warn', 'ignore'

    Returns:
    --------
    float
        MCC score between -1 and +1
    """
    # Convert inputs to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights)

    # Handle empty arrays
    if len(y_true) == 0 or len(y_pred) == 0:
        if handle_empty == 'raise':
            raise ValueError("Input arrays cannot be empty")
        elif handle_empty == 'warn':
            warnings.warn("Input arrays are empty - returning 0")
            return 0.0
        else:
            return 0.0

    # Data validation - modified to handle edge cases
    validator = ClassificationDataValidator()

    # Skip class validation for single-class cases
    if len(np.unique(y_true)) > 1 or len(np.unique(y_pred)) > 1:
        try:
            validator.validate_all(
                y_true,
                y_pred,
                check_outliers=check_outliers,
                check_distribution=check_distribution,
                check_correlation=check_correlation,
                check_missing_large=check_missing_large,
                check_class_balance=check_class_balance,
                sample_size=sample_size
            )
        except ValueError as e:
            if "must have the same set of classes" in str(e):
                # For cases where predictions and true labels have completely different classes
                return -1.0  # Maximum disagreement
            raise

    # Handle infinite values
    if force_finite:
        y_true = np.nan_to_num(y_true, nan=np.nan, posinf=np.nan, neginf=np.nan)
        y_pred = np.nan_to_num(y_pred, nan=np.nan, posinf=np.nan, neginf=np.nan)
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        if sample_weights is not None:
            sample_weights = sample_weights[mask]
        if len(y_true) == 0:
            return 0.0

    # Get all classes that appear in either y_true or y_pred
    all_classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(all_classes)

    # Handle single-class case
    if n_classes == 1:
        return 0.0  # No correlation can be measured

    # Initialize confusion matrix
    if sample_weights is not None:
        confusion_mat = np.zeros((n_classes, n_classes), dtype=float)
    else:
        confusion_mat = np.zeros((n_classes, n_classes), dtype=int)

    # Build confusion matrix with class mapping
    class_to_idx = {cls: i for i, cls in enumerate(all_classes)}
    for true_cls, pred_cls in zip(y_true, y_pred):
        i = class_to_idx[true_cls]
        j = class_to_idx[pred_cls]
        if sample_weights is not None:
            # Need to get the weight for this specific sample
            weight = sample_weights[np.where((y_true == true_cls) & (y_pred == pred_cls))[0][0]]
            confusion_mat[i, j] += weight
        else:
            confusion_mat[i, j] += 1

    # Calculate components for MCC
    tp = np.diag(confusion_mat)
    fp = np.sum(confusion_mat, axis=0) - tp
    fn = np.sum(confusion_mat, axis=1) - tp
    tn = np.sum(confusion_mat) - (tp + fp + fn)

    # Calculate MCC per class
    with np.errstate(divide='ignore', invalid='ignore'):
        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc_per_class = numerator / denominator

    # Handle division by zero cases
    mcc_per_class[denominator == 0] = 0

    # Handle averaging
    if average == 'macro':
        return np.mean(mcc_per_class)
    elif average == 'weighted':
        class_counts = np.sum(confusion_mat, axis=1)
        if np.sum(class_counts) == 0:
            return 0.0
        return np.sum(mcc_per_class * class_counts) / np.sum(class_counts)
    else:
        raise ValueError("Invalid average option. Choose from 'macro' or 'weighted'")

"""# cohens_kappa"""

import numpy as np
import warnings

def cohens_kappa(
    y_true,
    y_pred,
    *,
    sample_weights=None,
    force_finite=False,
    check_outliers=False,
    check_distribution=False,
    check_correlation=False,
    check_missing_large=False,
    check_class_balance=False,
    sample_size=1000,
    handle_empty='raise'
):
    """
    Compute Cohen's Kappa with robust validation and edge case handling.

    Cohen's Kappa measures inter-rater agreement, accounting for chance agreement.
    Range: -1 (complete disagreement) to 1 (perfect agreement).

    Parameters:
    -----------
    y_true : array-like
        Ground truth target values.
    y_pred : array-like
        Predicted target values.
    sample_weights : array-like, optional
        Sample weights for weighted calculations.
    force_finite : bool, optional
        Handle infinite values by conversion/removal.
    check_outliers : bool, optional
        Check for outliers in y_true.
    check_distribution : bool, optional
        Check if y_true follows normal distribution.
    check_correlation : bool, optional
        Check correlation between y_true and y_pred.
    check_missing_large : bool, optional
        Check for missing values in large datasets.
    check_class_balance : bool, optional
        Check if classes are imbalanced.
    sample_size : int, optional
        Sample size for large dataset checks.
    handle_empty : str, optional
        How to handle empty inputs: 'raise', 'warn', 'ignore'

    Returns:
    --------
    float
        Cohen's Kappa between -1 and 1
    """
    # Convert inputs to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights)

    # Handle empty arrays
    if len(y_true) == 0 or len(y_pred) == 0:
        if handle_empty == 'raise':
            raise ValueError("Input arrays cannot be empty")
        elif handle_empty == 'warn':
            warnings.warn("Input arrays are empty - returning 0")
            return 0.0
        else:
            return 0.0

    # Data validation with edge case handling
    validator = ClassificationDataValidator()

    # Skip class validation for single-class cases
    if len(np.unique(y_true)) > 1 or len(np.unique(y_pred)) > 1:
        try:
            validator.validate_all(
                y_true,
                y_pred,
                check_outliers=check_outliers,
                check_distribution=check_distribution,
                check_correlation=check_correlation,
                check_missing_large=check_missing_large,
                check_class_balance=check_class_balance,
                sample_size=sample_size
            )
        except ValueError as e:
            if "must have the same set of classes" in str(e):
                # For cases where predictions and true labels have completely different classes
                return -1.0  # Maximum disagreement
            raise

    # Handle infinite values
    if force_finite:
        y_true = np.nan_to_num(y_true, nan=np.nan, posinf=np.nan, neginf=np.nan)
        y_pred = np.nan_to_num(y_pred, nan=np.nan, posinf=np.nan, neginf=np.nan)
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        if sample_weights is not None:
            sample_weights = sample_weights[mask]
        if len(y_true) == 0:
            return 0.0

    # Get all classes that appear in either y_true or y_pred
    all_classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(all_classes)

    # Handle single-class case
    if n_classes == 1:
        return 1.0  # Perfect agreement by definition

    # Initialize confusion matrix
    if sample_weights is not None:
        confusion_mat = np.zeros((n_classes, n_classes), dtype=float)
    else:
        confusion_mat = np.zeros((n_classes, n_classes), dtype=int)

    # Build confusion matrix with class mapping
    class_to_idx = {cls: i for i, cls in enumerate(all_classes)}
    for true_cls, pred_cls in zip(y_true, y_pred):
        i = class_to_idx[true_cls]
        j = class_to_idx[pred_cls]
        if sample_weights is not None:
            # Get the weight for this specific sample
            weight = sample_weights[np.where((y_true == true_cls) & (y_pred == pred_cls))[0][0]]
            confusion_mat[i, j] += weight
        else:
            confusion_mat[i, j] += 1

    # Calculate observed agreement
    total = np.sum(confusion_mat)
    if total == 0:
        return 0.0
    observed_agreement = np.trace(confusion_mat) / total

    # Calculate expected agreement
    row_sums = np.sum(confusion_mat, axis=1)
    col_sums = np.sum(confusion_mat, axis=0)
    expected_agreement = np.sum(row_sums * col_sums) / (total ** 2)

    # Calculate Cohen's Kappa
    if expected_agreement == 1:
        return 1.0  # Perfect agreement
    if observed_agreement == 1 and expected_agreement == 0:
        return 1.0  # Edge case: perfect agreement with zero chance agreement

    kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
    return np.clip(kappa, -1.0, 1.0)  # Ensure within valid range

"""# fbeta_score"""

def fbeta_score(
    y_true,
    y_pred,
    beta,
    *,
    average='binary',
    sample_weights=None,
    force_finite=False,
    check_outliers=False,
    check_distribution=False,
    check_correlation=False,
    check_missing_large=False,
    check_class_balance=False,
    sample_size=1000,
    handle_empty='raise'
):
    """
    Compute Fβ-Score with proper multi-class handling.
    """
    # Validate beta parameter
    if beta <= 0:
        raise ValueError("beta should be >0 in the F-beta score")

    # Convert inputs to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights)

    # Handle empty arrays
    if len(y_true) == 0 or len(y_pred) == 0:
        if handle_empty == 'raise':
            raise ValueError("Input arrays cannot be empty")
        elif handle_empty == 'warn':
            warnings.warn("Input arrays are empty - returning 0")
            return 0.0
        else:
            return 0.0

    # Data validation
    validator = ClassificationDataValidator()
    validator.validate_all(
        y_true,
        y_pred,
        check_outliers=check_outliers,
        check_distribution=check_distribution,
        check_correlation=check_correlation,
        check_missing_large=check_missing_large,
        check_class_balance=check_class_balance,
        sample_size=sample_size
    )

    # Handle infinite values
    if force_finite:
        y_true = np.nan_to_num(y_true, nan=np.nan, posinf=np.nan, neginf=np.nan)
        y_pred = np.nan_to_num(y_pred, nan=np.nan, posinf=np.nan, neginf=np.nan)
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        if sample_weights is not None:
            sample_weights = sample_weights[mask]
        if len(y_true) == 0:
            return 0.0

    # Get unique classes
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(unique_classes)

    # For binary case with single class present, return 1 if perfect, 0 otherwise
    if average == 'binary' and n_classes == 1:
        return 1.0 if np.all(y_true == y_pred) else 0.0

    # Initialize arrays
    fbeta_scores = np.zeros(n_classes)
    class_counts = np.zeros(n_classes)

    # Calculate precision and recall for each class
    for i, cls in enumerate(unique_classes):
        true_positives = (y_true == cls) & (y_pred == cls)
        false_positives = (y_true != cls) & (y_pred == cls)
        false_negatives = (y_true == cls) & (y_pred != cls)

        if sample_weights is not None:
            tp = np.sum(sample_weights[true_positives])
            fp = np.sum(sample_weights[false_positives])
            fn = np.sum(sample_weights[false_negatives])
            class_counts[i] = np.sum(sample_weights[y_true == cls])
        else:
            tp = np.sum(true_positives)
            fp = np.sum(false_positives)
            fn = np.sum(false_negatives)
            class_counts[i] = np.sum(y_true == cls)

        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Calculate Fβ-Score
        if precision + recall == 0:
            fbeta_scores[i] = 0
        else:
            fbeta_scores[i] = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)

    # Handle averaging
    if average == 'binary':
        if n_classes != 2:
            raise ValueError("Binary Fβ-Score requires exactly 2 classes")
        return fbeta_scores[1]
    elif average == 'micro':
        if sample_weights is not None:
            tp_total = np.sum(sample_weights[y_true == y_pred])
            fp_total = np.sum(sample_weights[y_true != y_pred])
            fn_total = np.sum(sample_weights[y_true != y_pred])
        else:
            tp_total = np.sum(y_true == y_pred)
            fp_total = np.sum(y_true != y_pred)
            fn_total = np.sum(y_true != y_pred)

        precision_micro = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
        recall_micro = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0

        if precision_micro + recall_micro == 0:
            return 0
        return (1 + beta**2) * (precision_micro * recall_micro) / (beta**2 * precision_micro + recall_micro)
    elif average == 'macro':
        return np.mean(fbeta_scores)
    elif average == 'weighted':
        if np.sum(class_counts) == 0:
            return 0.0
        return np.sum(fbeta_scores * class_counts) / np.sum(class_counts)
    elif average == 'none':
        return fbeta_scores
    else:
        raise ValueError("Invalid average option. Choose from: 'binary', 'micro', 'macro', 'weighted', 'none'")

"""# jaccard_score"""

import numpy as np
import warnings

def jaccard_score(
    y_true,
    y_pred,
    *,
    average='binary',
    sample_weights=None,
    force_finite=False,
    check_outliers=False,
    check_distribution=False,
    check_correlation=False,
    check_missing_large=False,
    check_class_balance=False,
    sample_size=1000,
    handle_empty='raise'
):
    """
    Compute Jaccard Score with robust validation and edge case handling.

    Jaccard Score (Intersection over Union) measures similarity between sample sets.
    Range: 0 (no similarity) to 1 (identical sets).

    Parameters:
    -----------
    y_true : array-like
        Ground truth target values.
    y_pred : array-like
        Predicted target values.
    average : str, optional
        Averaging method: 'binary', 'micro', 'macro', 'weighted', or 'none'.
    sample_weights : array-like, optional
        Sample weights for weighted calculations.
    force_finite : bool, optional
        Handle infinite values by conversion/removal.
    check_outliers : bool, optional
        Check for outliers in y_true.
    check_distribution : bool, optional
        Check if y_true follows normal distribution.
    check_correlation : bool, optional
        Check correlation between y_true and y_pred.
    check_missing_large : bool, optional
        Check for missing values in large datasets.
    check_class_balance : bool, optional
        Check if classes are imbalanced.
    sample_size : int, optional
        Sample size for large dataset checks.
    handle_empty : str, optional
        How to handle empty inputs: 'raise', 'warn', 'ignore'

    Returns:
    --------
    float or ndarray
        Jaccard Score between 0 and 1
    """
    # Convert inputs to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights)

    # Handle empty arrays
    if len(y_true) == 0 or len(y_pred) == 0:
        if handle_empty == 'raise':
            raise ValueError("Input arrays cannot be empty")
        elif handle_empty == 'warn':
            warnings.warn("Input arrays are empty - returning 0")
            return 0.0
        else:
            return 0.0

    # Data validation with edge case handling
    validator = ClassificationDataValidator()

    # Skip class validation for single-class cases
    if len(np.unique(y_true)) > 1 or len(np.unique(y_pred)) > 1:
        try:
            validator.validate_all(
                y_true,
                y_pred,
                check_outliers=check_outliers,
                check_distribution=check_distribution,
                check_correlation=check_correlation,
                check_missing_large=check_missing_large,
                check_class_balance=check_class_balance,
                sample_size=sample_size
            )
        except ValueError as e:
            if "must have the same set of classes" in str(e):
                # For cases where predictions and true labels have completely different classes
                return 0.0  # No similarity
            raise

    # Handle infinite values
    if force_finite:
        y_true = np.nan_to_num(y_true, nan=np.nan, posinf=np.nan, neginf=np.nan)
        y_pred = np.nan_to_num(y_pred, nan=np.nan, posinf=np.nan, neginf=np.nan)
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        if sample_weights is not None:
            sample_weights = sample_weights[mask]
        if len(y_true) == 0:
            return 0.0

    # Get all classes that appear in either y_true or y_pred
    all_classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(all_classes)

    # Handle single-class case
    if n_classes == 1:
        return 1.0 if np.all(y_true == y_pred) else 0.0

    # Initialize arrays
    jaccard_scores = np.zeros(n_classes)
    class_counts = np.zeros(n_classes)

    # Calculate Jaccard score for each class
    for i, cls in enumerate(all_classes):
        true_positives = (y_true == cls) & (y_pred == cls)
        false_positives = (y_true != cls) & (y_pred == cls)
        false_negatives = (y_true == cls) & (y_pred != cls)

        if sample_weights is not None:
            tp = np.sum(sample_weights[true_positives])
            fp = np.sum(sample_weights[false_positives])
            fn = np.sum(sample_weights[false_negatives])
            class_counts[i] = np.sum(sample_weights[y_true == cls])
        else:
            tp = np.sum(true_positives)
            fp = np.sum(false_positives)
            fn = np.sum(false_negatives)
            class_counts[i] = np.sum(y_true == cls)

        union = tp + fp + fn
        jaccard_scores[i] = tp / union if union > 0 else 0.0

    # Handle averaging
    if average == 'binary':
        if n_classes != 2:
            raise ValueError("Binary Jaccard Score requires exactly 2 classes")
        return jaccard_scores[1]  # Score for positive class
    elif average == 'micro':
        if sample_weights is not None:
            tp_total = np.sum(sample_weights[y_true == y_pred])
            fp_total = np.sum(sample_weights[y_true != y_pred])
            fn_total = np.sum(sample_weights[y_true != y_pred])
        else:
            tp_total = np.sum(y_true == y_pred)
            fp_total = np.sum(y_true != y_pred)
            fn_total = np.sum(y_true != y_pred)

        union_total = tp_total + fp_total + fn_total
        return tp_total / union_total if union_total > 0 else 0.0
    elif average == 'macro':
        return np.mean(jaccard_scores)
    elif average == 'weighted':
        if np.sum(class_counts) == 0:
            return 0.0
        return np.sum(jaccard_scores * class_counts) / np.sum(class_counts)
    elif average == 'none':
        return jaccard_scores
    else:
        raise ValueError("Invalid average option. Choose from: 'binary', 'micro', 'macro', 'weighted', 'none'")

"""# hamming_loss"""

import numpy as np

def hamming_loss(
    y_true,
    y_pred,
    sample_weights=None,
    force_finite=False,
    check_outliers=False,
    check_distribution=False,
    check_correlation=False,
    check_missing_large=False,
    check_class_balance=False,
    sample_size=1000
):
    """
    Compute Hamming Loss.

    Hamming Loss is the fraction of labels that are incorrectly predicted.

    Parameters:
    -----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated target values.
    sample_weights : array-like, optional
        Sample weights. If provided, the errors are weighted by these values.
    force_finite : bool, optional
        If True, replaces infinite values with a finite value (e.g., 0).
    check_outliers : bool, optional
        If True, check for outliers in y_true.
    check_distribution : bool, optional
        If True, check if y_true follows a normal distribution.
    check_correlation : bool, optional
        If True, check for high correlation between y_true and y_pred.
    check_missing_large : bool, optional
        If True, check for missing values in large data using sampling.
    check_class_balance : bool, optional
        If True, check if classes are imbalanced.
    sample_size : int, optional
        Sample size for checking missing values in large data.

    Returns:
    --------
    float
        The computed Hamming Loss.
    """

    validator = ClassificationDataValidator()
    validator.validate_all(
        y_true,
        y_pred,
        check_outliers=check_outliers,
        check_distribution=check_distribution,
        check_correlation=check_correlation,
        check_missing_large=check_missing_large,
        check_class_balance=check_class_balance,
        sample_size=sample_size
    )


    if force_finite:
        y_true = np.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)


    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    incorrect_labels = np.sum(y_true != y_pred)


    hamming_loss = incorrect_labels / (y_true.size)

    if sample_weights is not None:
        sample_weights = np.array(sample_weights)
        if len(sample_weights) != len(y_true):
            raise ValueError("sample_weights must have the same length as y_true and y_pred")
        hamming_loss = np.sum((y_true != y_pred) * sample_weights) / np.sum(sample_weights)

    return hamming_loss

"""# log_loss"""

import numpy as np

def log_loss(
    y_true,
    y_pred_proba,
    eps=1e-15,
    normalize=True,
    sample_weights=None,
    force_finite=False,
    check_outliers=False,
    check_distribution=False,
    check_correlation=False,
    check_missing_large=False,
    check_class_balance=False,
    sample_size=1000
):
    """
    Compute Log Loss (Logarithmic Loss).

    Log Loss measures the performance of a classification model where the prediction is a probability value between 0 and 1.

    Parameters:
    -----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred_proba : array-like
        Predicted probabilities for each class.
    eps : float, optional
        A small value to avoid log(0) which is undefined.
    normalize : bool, optional
        If True, return the mean loss per sample. If False, return the sum of the losses.
    sample_weights : array-like, optional
        Sample weights. If provided, the errors are weighted by these values.
    force_finite : bool, optional
        If True, replaces infinite values with a finite value (e.g., 0).
    check_outliers : bool, optional
        If True, check for outliers in y_true.
    check_distribution : bool, optional
        If True, check if y_true follows a normal distribution.
    check_correlation : bool, optional
        If True, check for high correlation between y_true and y_pred.
    check_missing_large : bool, optional
        If True, check for missing values in large data using sampling.
    check_class_balance : bool, optional
        If True, check if classes are imbalanced.
    sample_size : int, optional
        Sample size for checking missing values in large data.

    Returns:
    --------
    float
        The computed Log Loss.
    """

    validator = ClassificationDataValidator()
    validator.validate_all(
        y_true,
        y_pred_proba,
        check_outliers=check_outliers,
        check_distribution=check_distribution,
        check_correlation=check_correlation,
        check_missing_large=check_missing_large,
        check_class_balance=check_class_balance,
        sample_size=sample_size
    )

    if force_finite:
        y_true = np.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)
        y_pred_proba = np.nan_to_num(y_pred_proba, nan=0.0, posinf=0.0, neginf=0.0)


    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)

    if len(y_pred_proba.shape) == 1:
        y_pred_proba = np.vstack([1 - y_pred_proba, y_pred_proba]).T  

    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)


    if len(y_true.shape) == 1:
        n_classes = y_pred_proba.shape[1]
        y_true_one_hot = np.eye(n_classes)[y_true]
    else:
        y_true_one_hot = y_true


    losses = -np.sum(y_true_one_hot * np.log(y_pred_proba), axis=1)

    if sample_weights is not None:
        sample_weights = np.array(sample_weights)
        if len(sample_weights) != len(y_true):
            raise ValueError("sample_weights must have the same length as y_true")
        losses *= sample_weights


    if normalize:
        return np.mean(losses)
    else:
        return np.sum(losses)

"""# confusion_matrix"""

import numpy as np

def confusion_matrix(
    y_true,
    y_pred,
    labels=None,
    sample_weights=None,
    force_finite=False,
    check_outliers=False,
    check_distribution=False,
    check_correlation=False,
    check_missing_large=False,
    check_class_balance=False,
    sample_size=1000
):
    """
    Compute Confusion Matrix.

    Confusion Matrix is a table that describes the performance of a classification model.

    Parameters:
    -----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated target values.
    labels : array-like, optional
        List of labels to index the matrix. If None, all unique labels in y_true and y_pred are used.
    sample_weights : array-like, optional
        Sample weights. If provided, the counts are weighted by these values.
    force_finite : bool, optional
        If True, replaces infinite values with a finite value (e.g., 0).
    check_outliers : bool, optional
        If True, check for outliers in y_true.
    check_distribution : bool, optional
        If True, check if y_true follows a normal distribution.
    check_correlation : bool, optional
        If True, check for high correlation between y_true and y_pred.
    check_missing_large : bool, optional
        If True, check for missing values in large data using sampling.
    check_class_balance : bool, optional
        If True, check if classes are imbalanced.
    sample_size : int, optional
        Sample size for checking missing values in large data.

    Returns:
    --------
    numpy.ndarray
        The computed Confusion Matrix.
    """

    validator = ClassificationDataValidator()
    validator.validate_all(
        y_true,
        y_pred,
        check_outliers=check_outliers,
        check_distribution=check_distribution,
        check_correlation=check_correlation,
        check_missing_large=check_missing_large,
        check_class_balance=check_class_balance,
        sample_size=sample_size
    )

    if force_finite:
        y_true = np.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    n_labels = len(labels)


    cm = np.zeros((n_labels, n_labels), dtype=int)

  
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            if sample_weights is None:
                cm[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))
            else:
                cm[i, j] = np.sum(sample_weights[(y_true == true_label) & (y_pred == pred_label)])

    return cm

"""# top_k_accuracy"""

import numpy as np

def top_k_accuracy(
    y_true,
    y_pred_proba,
    k=1,
    normalize=True,
    sample_weights=None,
    force_finite=False,
    check_outliers=False,
    check_distribution=False,
    check_correlation=False,
    check_missing_large=False,
    check_class_balance=False,
    sample_size=1000
):
    """
    Compute Top-K Accuracy.

    Top-K Accuracy measures the proportion of times the true label is among the top K predicted classes.

    Parameters:
    -----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred_proba : array-like
        Predicted probabilities for each class.
    k : int, optional
        Number of top predictions to consider.
    normalize : bool, optional
        If True, return the fraction of correctly classified samples. If False, return the count of correctly classified samples.
    sample_weights : array-like, optional
        Sample weights. If provided, the errors are weighted by these values.
    force_finite : bool, optional
        If True, replaces infinite values with a finite value (e.g., 0).
    check_outliers : bool, optional
        If True, check for outliers in y_true.
    check_distribution : bool, optional
        If True, check if y_true follows a normal distribution.
    check_correlation : bool, optional
        If True, check for high correlation between y_true and y_pred.
    check_missing_large : bool, optional
        If True, check for missing values in large data using sampling.
    check_class_balance : bool, optional
        If True, check if classes are imbalanced.
    sample_size : int, optional
        Sample size for checking missing values in large data.

    Returns:
    --------
    float or int
        The computed Top-K Accuracy.
    """

    validator = ClassificationDataValidator()
    validator.validate_all(
        y_true,
        y_pred_proba,
        check_outliers=check_outliers,
        check_distribution=check_distribution,
        check_correlation=check_correlation,
        check_missing_large=check_missing_large,
        check_class_balance=check_class_balance,
        sample_size=sample_size
    )

    if force_finite:
        y_true = np.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)
        y_pred_proba = np.nan_to_num(y_pred_proba, nan=0.0, posinf=0.0, neginf=0.0)

    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)


    if len(y_pred_proba.shape) == 1:
        y_pred_proba = np.vstack([1 - y_pred_proba, y_pred_proba]).T  # حالت Binary

    if k > y_pred_proba.shape[1]:
        raise ValueError(f"k={k} is greater than the number of classes {y_pred_proba.shape[1]}")

    top_k_classes = np.argsort(y_pred_proba, axis=1)[:, -k:]


    correct_predictions = np.array([y_true[i] in top_k_classes[i] for i in range(len(y_true))])

    if sample_weights is not None:
        sample_weights = np.array(sample_weights)
        if len(sample_weights) != len(y_true):
            raise ValueError("sample_weights must have the same length as y_true")
        correct_predictions = correct_predictions * sample_weights

    if normalize:
        return np.sum(correct_predictions) / np.sum(sample_weights) if sample_weights is not None else np.mean(correct_predictions)
    else:
        return np.sum(correct_predictions)