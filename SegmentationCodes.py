

# dice_score
"""

def dice_score(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    smooth: float = 1e-6,
    is_binary: bool = False,
    is_multiclass: bool = False,
    n_classes: Optional[int] = None,
    threshold: float = 0.5,
    is_probabilistic: bool = False,
    ignore_background: bool = False,
    validator: Optional[SegmentationDataValidator] = None,
    **validator_kwargs,
) -> Union[float, np.ndarray]:
    """
    Compute Dice Score with complete validation.
    """
    if validator is None:
        validator = SegmentationDataValidator()

    # Run all checks with proper parameters
    validator.validate_all(
        y_true,
        y_pred,
        is_binary=is_binary,
        is_multiclass=is_multiclass,
        is_probabilistic=is_probabilistic,
        n_classes=n_classes,
        **validator_kwargs
    )

    y_true = validator._ensure_numpy(y_true)
    y_pred = validator._ensure_numpy(y_pred)
    # Binarize probabilistic predictions
    if is_probabilistic:
        y_pred = (y_pred > threshold).astype(np.uint8)

    if is_multiclass:
        if n_classes is None:
            raise ValueError("n_classes must be specified for multi-class Dice.")

        dice_scores = []
        start_class = 1 if ignore_background else 0

        for class_idx in range(start_class, n_classes):
            true_class = (y_true == class_idx).astype(np.uint8)
            pred_class = (y_pred == class_idx).astype(np.uint8)

            intersection = np.sum(true_class * pred_class)
            union = np.sum(true_class) + np.sum(pred_class)

            dice_scores.append((2. * intersection + smooth) / (union + smooth))

        return np.mean(dice_scores)

    # Binary case
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    return (2. * intersection + smooth) / (union + smooth)

"""# iou_score"""

def iou_score(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    smooth: float = 1e-6,
    is_binary: bool = False,
    is_multiclass: bool = False,
    is_probabilistic: bool = False,
    threshold: float = 0.5,
    n_classes: Optional[int] = None,
    ignore_background: bool = False,
    validator: Optional[SegmentationDataValidator] = None,
) -> Union[float, np.ndarray]:
    """
    Compute Intersection over Union (Jaccard Index) for segmentation.
    """
    if validator is None:
        validator = SegmentationDataValidator()

    validator.validate_all(
        y_true,
        y_pred,
        is_binary=is_binary,
        is_multiclass=is_multiclass,
        is_probabilistic=is_probabilistic,
        n_classes=n_classes
    )

    y_true = validator._ensure_numpy(y_true)
    y_pred = validator._ensure_numpy(y_pred)

    if is_probabilistic:
        y_pred = (y_pred > threshold).astype(np.uint8)

    if is_multiclass:
        if n_classes is None:
            raise ValueError("n_classes must be specified for multi-class IoU.")

        iou_scores = []
        start_class = 1 if ignore_background else 0

        for class_idx in range(start_class, n_classes):
            true_class = (y_true == class_idx).astype(np.uint8)
            pred_class = (y_pred == class_idx).astype(np.uint8)

            intersection = np.sum(true_class * pred_class)
            union = np.sum(true_class) + np.sum(pred_class) - intersection

            iou_scores.append((intersection + smooth) / (union + smooth))

        return np.mean(iou_scores)

    # Binary case
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

"""# sensitivity (Recall)"""

def sensitivity(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    smooth: float = 1e-6,
    is_probabilistic: bool = False,
    threshold: float = 0.5,
    validator: Optional[SegmentationDataValidator] = None,
) -> float:
    """
    Compute Sensitivity (True Positive Rate) for binary segmentation.
    """
    if validator is None:
        validator = SegmentationDataValidator()

    validator.validate_all(
        y_true,
        y_pred,
        is_binary=True,
        is_probabilistic=is_probabilistic
    )

    y_true = validator._ensure_numpy(y_true)
    y_pred = validator._ensure_numpy(y_pred)

    if is_probabilistic:
        y_pred = (y_pred > threshold).astype(np.uint8)

    true_positives = np.sum(y_true * y_pred)
    actual_positives = np.sum(y_true)

    return (true_positives + smooth) / (actual_positives + smooth)

"""# specificity"""

def specificity(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    smooth: float = 1e-6,
    is_probabilistic: bool = False,
    threshold: float = 0.5,
    validator: Optional[SegmentationDataValidator] = None,
) -> float:
    """
    Compute Specificity (True Negative Rate) for binary segmentation.
    """
    if validator is None:
        validator = SegmentationDataValidator()

    validator.validate_all(
        y_true,
        y_pred,
        is_binary=True,
        is_probabilistic=is_probabilistic
    )

    y_true = validator._ensure_numpy(y_true)
    y_pred = validator._ensure_numpy(y_pred)

    if is_probabilistic:
        y_pred = (y_pred > threshold).astype(np.uint8)

    true_negatives = np.sum((1 - y_true) * (1 - y_pred))
    actual_negatives = np.sum(1 - y_true)

    return (true_negatives + smooth) / (actual_negatives + smooth)

"""# precision"""

def precision(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    smooth: float = 1e-6,
    is_probabilistic: bool = False,
    threshold: float = 0.5,
    validator: Optional[SegmentationDataValidator] = None,
) -> float:
    """
    Compute Precision for binary segmentation.
    """
    if validator is None:
        validator = SegmentationDataValidator()

    validator.validate_all(
        y_true,
        y_pred,
        is_binary=True,
        is_probabilistic=is_probabilistic
    )

    y_true = validator._ensure_numpy(y_true)
    y_pred = validator._ensure_numpy(y_pred)

    if is_probabilistic:
        y_pred = (y_pred > threshold).astype(np.uint8)

    true_positives = np.sum(y_true * y_pred)
    predicted_positives = np.sum(y_pred)

    return (true_positives + smooth) / (predicted_positives + smooth)

"""# hausdorff_distance"""

def hausdorff_distance(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    percentile: Optional[float] = None,
    is_probabilistic: bool = False,
    threshold: float = 0.5,
    validator: Optional[SegmentationDataValidator] = None,
) -> float:
    """
    Compute Hausdorff Distance between boundaries of segmentation masks.

    Parameters:
    -----------
    y_true : array-like
        Ground truth binary mask
    y_pred : array-like
        Predicted mask (binary or probabilistic)
    percentile : float, optional
        Percentile for partial Hausdorff distance (0-100)
    is_probabilistic : bool, default=False
        Whether y_pred contains probabilistic values
    threshold : float, default=0.5
        Threshold for binarizing probabilistic predictions
    validator : SegmentationValidator, optional
        Custom validator instance
    """
    if validator is None:
        validator = SegmentationDataValidator()

    validator.validate_all(
        y_true,
        y_pred,
        is_binary=not is_probabilistic,
        is_probabilistic=is_probabilistic
    )

    y_true = validator._ensure_numpy(y_true)
    y_pred = validator._ensure_numpy(y_pred)

    if is_probabilistic:
        y_pred = (y_pred > threshold).astype(bool)
    else:
        y_pred = y_pred.astype(bool)

    y_true = y_true.astype(bool)

    # Extract coordinates of boundary points
    true_coords = np.argwhere(y_true)
    pred_coords = np.argwhere(y_pred)

    if len(true_coords) == 0 or len(pred_coords) == 0:
        return float('inf')

    if percentile is not None:
        # Compute partial Hausdorff distance
        distances = []
        for i in range(true_coords.shape[0]):
            distances.append(np.min(np.linalg.norm(pred_coords - true_coords[i], axis=1)))
        return np.percentile(distances, percentile)
    else:
        # Compute full Hausdorff distance
        return max(
            directed_hausdorff(true_coords, pred_coords)[0],
            directed_hausdorff(pred_coords, true_coords)[0]
        )