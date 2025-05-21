

# Data Validation
"""

list_of_metrics = [ "dice_score", "iou_score", "sensitivity", "specificity", "precision", "hausdorff_distance"]

class DataValidator:
    def __init__(self, raise_warning=True):
        """Initialize the DataValidator class"""
        self.raise_warning = raise_warning

    def check_data_type(self, y_true, y_pred):
        """Check if input data types are valid"""
        valid_types = (np.ndarray, pd.Series, pd.DataFrame, list)
        if not isinstance(y_true, valid_types) or not isinstance(y_pred, valid_types):
            raise TypeError("y_true and y_pred must be numpy array, pandas series, or list")

    def check_shapes(self, y_true, y_pred):
        """Check if y_true and y_pred have the same shape"""
        if np.shape(y_true) != np.shape(y_pred):
            raise ValueError("y_true and y_pred must have the same shape")

    def check_missing_values(self, y_true, y_pred):
        """Check for missing values"""
        if np.any(pd.isnull(y_true)) or np.any(pd.isnull(y_pred)):
            raise ValueError("Missing values (NaN) detected in data")

    def check_inf_values(self, y_true, y_pred):
        """Check for infinite values"""
        if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
            raise ValueError("Infinite values (inf) detected in data")

    def check_lengths(self, y_true, y_pred):
        """Check if y_true and y_pred have the same length"""
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")

    def check_numeric_values(self, y_true, y_pred):
        """Check if values are numeric"""
        if not np.issubdtype(np.array(y_true).dtype, np.number) or not np.issubdtype(np.array(y_pred).dtype, np.number):
            raise TypeError("y_true and y_pred must contain numeric values")

    def check_variance(self, y_true, y_pred):
        """Check if variance of y_true is zero (can cause issues in R-squared calculation)"""
        if np.var(y_true) == 0:
            raise ValueError("Variance of y_true is zero. R-squared may not be meaningful")

    def check_non_negative(self, y_true, y_pred):
        """Check that values are non-negative for Logarithmic Mean Squared Error"""
        if np.any(y_true < -1) or np.any(y_pred < -1):
            raise ValueError("y_true and y_pred must be greater than or equal to -1 for log-based metrics")

    def check_multicollinearity(self, X, threshold=0.9):
        """Check for multicollinearity in input features"""
        if isinstance(X, pd.DataFrame):
            corr_matrix = X.corr().abs()
            high_corr = (corr_matrix > threshold).sum().sum() - len(X.columns)
            if high_corr > 0:
                raise ValueError("High multicollinearity detected in input features")
        else:
            if self.raise_warning:
                print("Warning: Multicollinearity check requires a pandas DataFrame")

    def check_outliers(self, y_true, y_pred, threshold=3):
        """Check for outliers using Z-score"""
        z_scores = (y_true - np.mean(y_true)) / np.std(y_true)
        if np.any(np.abs(z_scores) > threshold):
            raise ValueError("Outliers detected in y_true")

    def check_distribution(self, y_true, y_pred, distribution='normal'):
        """Check if data follows a specific distribution"""
        if distribution == 'normal':
            stat, p_value = kstest(y_true, 'norm')
            if p_value < 0.05:
                raise ValueError("y_true does not follow a normal distribution")

    def check_correlation(self, y_true, y_pred, threshold=0.8):
        """Check for high correlation between y_true and y_pred"""
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        if corr > threshold:
            raise ValueError("High correlation detected between y_true and y_pred")

    def check_missing_values_large_data(self, y_true, y_pred, sample_size=1000):
        """Check for missing values in large data using sampling"""
        indices = np.random.choice(len(y_true), sample_size, replace=False)
        if np.any(pd.isnull(y_true[indices])) or np.any(pd.isnull(y_pred[indices])):
            raise ValueError("Missing values (NaN) detected in data")

    def validate_all(self, y_true, y_pred, log_based=False, check_outliers=False, check_distribution=False, check_correlation=False):
        """Run all validation checks"""
        self.check_data_type(y_true, y_pred)
        self.check_shapes(y_true, y_pred)
        self.check_missing_values(y_true, y_pred)
        self.check_inf_values(y_true, y_pred)
        self.check_lengths(y_true, y_pred)
        self.check_numeric_values(y_true, y_pred)
        self.check_variance(y_true, y_pred)
        if log_based:
            self.check_non_negative(y_true, y_pred)
        if check_outliers:
            self.check_outliers(y_true, y_pred)
        if check_distribution:
            self.check_distribution(y_true, y_pred)
        if check_correlation:
            self.check_correlation(y_true, y_pred)
        return True  # Return True if all checks pass


class SegmentationDataValidator(DataValidator):
    def __init__(self, raise_warning: bool = True):
        super().__init__(raise_warning=raise_warning)

    def _ensure_numpy(self, data: Union[np.ndarray, list, torch.Tensor]) -> np.ndarray:
        """Convert input to numpy array if needed."""
        if str(type(data)) == "<class 'torch.Tensor'>":
            data = data.detach().cpu().numpy()
        elif isinstance(data, list):
            data = np.array(data)
        return data

    def check_empty_masks(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Check if masks are empty and raise appropriate warnings."""
        if np.all(y_true == 0):
            warnings.warn("y_true is completely empty (all background pixels).",
                         UserWarning, stacklevel=2)
        if np.all(y_pred == 0):
            warnings.warn("y_pred is completely empty (all background pixels).",
                         UserWarning, stacklevel=2)

    def check_binary(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Check if inputs are strictly binary (0 or 1)."""
        y_true = self._ensure_numpy(y_true)
        y_pred = self._ensure_numpy(y_pred)

        for name, data in [("y_true", y_true), ("y_pred", y_pred)]:
            unique = np.unique(data)
            if not np.all(np.isin(unique, [0, 1])):
                raise ValueError(f"{name} must be binary (0 or 1). Found values: {unique}")

    def check_spatial_consistency(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Ensure spatial dimensions match (HxW, HxWxD, etc.)."""
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true {y_true.shape} != y_pred {y_pred.shape}"
            )

    def check_connected_components(self, mask: np.ndarray, max_components: int = 10) -> None:
        """Warn if mask has too many disconnected regions (possible noise)."""
        labeled, n_components = label(mask)
        if n_components > max_components:
            warnings.warn(f"Mask has {n_components} connected components (possible noise).")

    def check_class_imbalance(self, y_true: np.ndarray, threshold: float = 0.01, is_binary: bool = False) -> None:
        """Warn if a class is extremely rare."""
        counts = np.bincount(y_true.flatten())
        total = np.sum(counts)

        for class_id, count in enumerate(counts):
            ratio = count / total
            if ratio < threshold:
                class_type = "foreground" if (is_binary and class_id == 1) else f"class {class_id}"
                warnings.warn(f"{class_type} is rare ({ratio:.2%} pixels).", UserWarning, stacklevel=2)

    def validate_all(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
        is_binary: bool = False,
        is_multiclass: bool = False,
        is_probabilistic: bool = False,
        n_classes: Optional[int] = None,
        check_connected_components: bool = False,
        check_class_imbalance: bool = False,
        check_empty_masks: bool = True,
    ) -> bool:
        """
        Run all validations with complete checks.
        """
        # Convert to numpy first
        y_true = self._ensure_numpy(y_true)
        y_pred = self._ensure_numpy(y_pred)

        # General data checks
        super().check_data_type(y_true, y_pred)
        super().check_shapes(y_true, y_pred)
        super().check_missing_values(y_true, y_pred)
        super().check_inf_values(y_true, y_pred)

        if not is_probabilistic:
            super().check_numeric_values(y_true, y_pred)

        # Segmentation-specific checks
        self.check_spatial_consistency(y_true, y_pred)

        if check_empty_masks:
            self.check_empty_masks(y_true, y_pred)

        if is_binary and not is_probabilistic:
            self.check_binary(y_true, y_pred)
        elif is_multiclass:
            if n_classes is None:
                raise ValueError("n_classes must be specified for multi-class validation.")
            if len(np.unique(y_true)) < 2:
                warnings.warn("Less than 2 classes in y_true.")

        if check_connected_components and not is_probabilistic:
            self.check_connected_components(y_true)
            self.check_connected_components(y_pred)

        if check_class_imbalance and is_multiclass:
            self.check_class_imbalance(y_true, is_binary=False)

        return True

# Metadata for segmentation metrics
SEGMENTATION_METRIC_DETAILS = {
    "dice_score": {
        "y_true": "Ground truth (correct) mask. Expected as a NumPy array or PyTorch tensor.",
        "y_pred": "Predicted mask. Expected as a NumPy array or PyTorch tensor.",
        "smooth": "Small smoothing factor to avoid division by zero. Default is 1e-6.",
        "is_binary": "If True, assumes binary segmentation. Default is False.",
        "is_multiclass": "If True, assumes multi-class segmentation. Default is False.",
        "n_classes": "Number of classes for multi-class segmentation. Required if is_multiclass is True.",
        "threshold": "Threshold for binarizing probabilistic predictions. Default is 0.5.",
        "is_probabilistic": "If True, assumes y_pred contains probabilities. Default is False.",
        "ignore_background": "If True, ignores the background class in calculations. Default is False.",
        "validator": "Optional SegmentationDataValidator instance for input validation.",
        "validator_kwargs": "Additional parameters for the validator."
    },
    "iou_score": {
        "y_true": "Ground truth (correct) mask. Expected as a NumPy array or PyTorch tensor.",
        "y_pred": "Predicted mask. Expected as a NumPy array or PyTorch tensor.",
        "smooth": "Small smoothing factor to avoid division by zero. Default is 1e-6.",
        "is_binary": "If True, assumes binary segmentation. Default is False.",
        "is_multiclass": "If True, assumes multi-class segmentation. Default is False.",
        "is_probabilistic": "If True, assumes y_pred contains probabilities. Default is False.",
        "threshold": "Threshold for binarizing probabilistic predictions. Default is 0.5.",
        "n_classes": "Number of classes for multi-class segmentation. Required if is_multiclass is True.",
        "ignore_background": "If True, ignores the background class in calculations. Default is False.",
        "validator": "Optional SegmentationDataValidator instance for input validation.",
        "validator_kwargs": "Additional parameters for the validator."
    },
    "sensitivity": {
        "y_true": "Ground truth (correct) mask. Expected as a NumPy array or PyTorch tensor.",
        "y_pred": "Predicted mask. Expected as a NumPy array or PyTorch tensor.",
        "smooth": "Small smoothing factor to avoid division by zero. Default is 1e-6.",
        "is_probabilistic": "If True, assumes y_pred contains probabilities. Default is False.",
        "threshold": "Threshold for binarizing probabilistic predictions. Default is 0.5.",
        "validator": "Optional SegmentationDataValidator instance for input validation.",
        "validator_kwargs": "Additional parameters for the validator."
    },
    "specificity": {
        "y_true": "Ground truth (correct) mask. Expected as a NumPy array or PyTorch tensor.",
        "y_pred": "Predicted mask. Expected as a NumPy array or PyTorch tensor.",
        "smooth": "Small smoothing factor to avoid division by zero. Default is 1e-6.",
        "is_probabilistic": "If True, assumes y_pred contains probabilities. Default is False.",
        "threshold": "Threshold for binarizing probabilistic predictions. Default is 0.5.",
        "validator": "Optional SegmentationDataValidator instance for input validation.",
        "validator_kwargs": "Additional parameters for the validator."
    },
    "precision": {
        "y_true": "Ground truth (correct) mask. Expected as a NumPy array or PyTorch tensor.",
        "y_pred": "Predicted mask. Expected as a NumPy array or PyTorch tensor.",
        "smooth": "Small smoothing factor to avoid division by zero. Default is 1e-6.",
        "is_probabilistic": "If True, assumes y_pred contains probabilities. Default is False.",
        "threshold": "Threshold for binarizing probabilistic predictions. Default is 0.5.",
        "validator": "Optional SegmentationDataValidator instance for input validation.",
        "validator_kwargs": "Additional parameters for the validator."
    },
    "hausdorff_distance": {
        "y_true": "Ground truth (correct) mask. Expected as a NumPy array or PyTorch tensor.",
        "y_pred": "Predicted mask. Expected as a NumPy array or PyTorch tensor.",
        "percentile": "Percentile value for computing the Hausdorff distance. Optional.",
        "is_probabilistic": "If True, assumes y_pred contains probabilities. Default is False.",
        "threshold": "Threshold for binarizing probabilistic predictions. Default is 0.5.",
        "validator": "Optional SegmentationDataValidator instance for input validation.",
        "validator_kwargs": "Additional parameters for the validator."
    }
}

def get_metric_details(metric_name):
    """
    Returns the parameter descriptions for a given segmentation metric.
    """
    if metric_name in SEGMENTATION_METRIC_DETAILS:
        return SEGMENTATION_METRIC_DETAILS[metric_name]
    else:
        raise ValueError(f"Metric '{metric_name}' not found in the segmentation module.")