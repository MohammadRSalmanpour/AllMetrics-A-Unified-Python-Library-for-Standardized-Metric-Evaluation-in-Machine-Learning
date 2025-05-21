

# mean_absolute_error
"""

def mean_absolute_error(
    y_true,
    y_pred,
    sample_weights=None,
    method='mean',
    multioutput='uniform_average',
    force_finite=False,
    check_outliers=False,
    check_distribution=False,
    check_correlation=False,
    check_missing_large=False,
    sample_size=1000
):
    """
    Compute Mean Absolute Error (MAE).

    Parameters:
    -----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated target values.
    sample_weights : array-like, optional
        Sample weights. If provided, the errors are weighted by these values.
    method : str, optional
        Method to compute the error. Options: 'mean', 'sum', 'none'.
    multioutput : str, optional
        Defines how to aggregate errors for multi-output targets. Options: 'uniform_average', 'raw_values'.
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
    sample_size : int, optional
        Sample size for checking missing values in large data.

    Returns:
    --------
    float or array-like
        The computed MAE.
    """
    # تبدیل ورودی‌ها به آرایه NumPy
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    validator = DataValidator()

    # اعتبارسنجی داده‌ها
    validator.validate_all(
        y_true,
        y_pred,
        check_outliers=check_outliers,
        check_distribution=check_distribution,
        check_correlation=check_correlation,
        check_missing_large=check_missing_large,
        sample_size=sample_size
    )

    # جایگزینی مقادیر نامتناهی (در صورت نیاز)
    if force_finite:
        y_true = np.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)

    # محاسبه خطا
    errors = np.abs(y_true - y_pred)

    # اعمال وزن‌های نمونه (در صورت وجود)
    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights, dtype=np.float64)
        if len(sample_weights) != len(y_true):
            raise ValueError("sample_weights must have the same length as y_true and y_pred")
        errors *= sample_weights
        errors/=sample_weights

    # ترکیب نتایج برای خروجی‌های چندگانه
    if multioutput == 'raw_values':
        if method == 'mean':
            return np.mean(errors, axis=0)
        elif method == 'sum':
            return np.sum(errors, axis=0)
        elif method == 'none':
            return errors
        else:
            raise ValueError("Invalid method. Choose from 'mean', 'sum', or 'none'")
    elif multioutput == 'uniform_average':
        if method == 'mean':
            return np.mean(errors)
        elif method == 'sum':
            return np.sum(errors)
        elif method == 'none':
            return errors
        else:
            raise ValueError("Invalid method. Choose from 'mean', 'sum', or 'none'")
    else:
        raise ValueError("Invalid multioutput option. Choose from 'uniform_average' or 'raw_values'.")

"""# mean_squared_error"""

def mean_squared_error(
    y_true,
    y_pred,
    sample_weights=None,
    squared=True,
    method='mean',
    multioutput='uniform_average',
    force_finite=False,
    check_outliers=False,
    check_distribution=False,
    check_correlation=False,
    check_missing_large=False,
    sample_size=1000
):
    """
    Compute Mean Squared Error (MSE) or Root Mean Squared Error (RMSE).

    Parameters:
    -----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated target values.
    sample_weights : array-like, optional
        Sample weights. If provided, the errors are weighted by these values.
    squared : bool, optional
        If True, returns MSE. If False, returns RMSE.
    method : str, optional
        Method to compute the error. Options: 'mean', 'sum', 'none'.
    multioutput : str, optional
        Defines how to aggregate errors for multi-output targets.
        Options: 'uniform_average', 'raw_values'.
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
    sample_size : int, optional
        Sample size for checking missing values in large data.

    Returns:
    --------
    float or array-like
        The computed MSE or RMSE.
    """
    # Convert inputs to numpy arrays
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Initialize and run validator
    validator = DataValidator()
    validator.validate_all(
        y_true,
        y_pred,
        check_outliers=check_outliers,
        check_distribution=check_distribution,
        check_correlation=check_correlation,
        check_missing_large=check_missing_large,
        sample_size=sample_size
    )

    # Handle infinite values if requested
    if force_finite:
        y_true = np.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)

    # Calculate squared errors
    squared_errors = np.square(y_true - y_pred)

    # Apply sample weights if provided
    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights, dtype=np.float64)
        if len(sample_weights) != len(y_true):
            raise ValueError("sample_weights must have the same length as y_true and y_pred")
        squared_errors *= sample_weights
        squared_errors /= np.mean(sample_weights)  # Normalize by mean weight

    # Calculate result based on method and multioutput
    if multioutput == 'raw_values':
        if method == 'mean':
            result = np.mean(squared_errors, axis=0)
        elif method == 'sum':
            result = np.sum(squared_errors, axis=0)
        elif method == 'none':
            result = squared_errors
        else:
            raise ValueError("Invalid method. Choose from 'mean', 'sum', or 'none'")
    elif multioutput == 'uniform_average':
        if method == 'mean':
            result = np.mean(squared_errors)
        elif method == 'sum':
            result = np.sum(squared_errors)
        elif method == 'none':
            result = squared_errors
        else:
            raise ValueError("Invalid method. Choose from 'mean', 'sum', or 'none'")
    else:
        raise ValueError("Invalid multioutput option. Choose from 'uniform_average' or 'raw_values'")

    # Return MSE or RMSE based on 'squared' parameter
    return result if squared else np.sqrt(result)

"""# mean_absolute_percentage_error"""

def mean_absolute_percentage_error(
    y_true,
    y_pred,
    sample_weights=None,
    method='mean',
    multioutput='uniform_average',
    force_finite=False,
    check_outliers=False,
    check_distribution=False,
    check_correlation=False,
    check_missing_large=False,
    sample_size=1000,
    epsilon=1e-10  # Small value to avoid division by zero
):
    """
    Compute Mean Absolute Percentage Error (MAPE).

    Parameters:
    -----------
    y_true : array-like
        Ground truth (correct) target values. Must be non-zero.
    y_pred : array-like
        Estimated target values.
    sample_weights : array-like, optional
        Sample weights. If provided, the errors are weighted by these values.
    method : str, optional
        Method to compute the error. Options: 'mean', 'sum', 'none'.
    multioutput : str, optional
        Defines how to aggregate errors for multi-output targets.
        Options: 'uniform_average', 'raw_values'.
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
    sample_size : int, optional
        Sample size for checking missing values in large data.
    epsilon : float, optional
        Small constant to avoid division by zero (default: 1e-10).

    Returns:
    --------
    float or array-like
        The computed MAPE (in percentage).

    Raises:
    -------
    ValueError
        If y_true contains zeros (or values close to zero).
    """
    # Convert inputs to numpy arrays with float64 dtype
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Initialize and run validator
    validator = DataValidator()
    validator.validate_all(
        y_true,
        y_pred,
        check_outliers=check_outliers,
        check_distribution=check_distribution,
        check_correlation=check_correlation,
        check_missing_large=check_missing_large,
        sample_size=sample_size
    )

    # Check for zero values in y_true
    if np.any(np.abs(y_true) < epsilon):
        raise ValueError("y_true contains values that are too close to zero, which would cause division errors")

    # Handle infinite values if requested
    if force_finite:
        y_true = np.nan_to_num(y_true, nan=epsilon, posinf=np.nanmax(y_true), neginf=np.nanmin(y_true))
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=np.nanmax(y_pred), neginf=np.nanmin(y_pred))

    # Calculate absolute percentage errors with epsilon for numerical stability
    ape = np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon)) * 100

    # Apply sample weights if provided
    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights, dtype=np.float64)
        if len(sample_weights) != len(y_true):
            raise ValueError("sample_weights must have the same length as y_true and y_pred")
        ape *= sample_weights
        ape /= np.mean(sample_weights)  # Normalize by mean weight

    # Calculate result based on method and multioutput
    if multioutput == 'raw_values':
        if method == 'mean':
            result = np.mean(ape, axis=0)
        elif method == 'sum':
            result = np.sum(ape, axis=0)
        elif method == 'none':
            result = ape
        else:
            raise ValueError("Invalid method. Choose from 'mean', 'sum', or 'none'")
    elif multioutput == 'uniform_average':
        if method == 'mean':
            result = np.mean(ape)
        elif method == 'sum':
            result = np.sum(ape)
        elif method == 'none':
            result = ape
        else:
            raise ValueError("Invalid method. Choose from 'mean', 'sum', or 'none'")
    else:
        raise ValueError("Invalid multioutput option. Choose from 'uniform_average' or 'raw_values'")

    return result

"""# mean_squared_log_error"""

def mean_squared_log_error(
    y_true,
    y_pred,
    sample_weights=None,
    squared=True,
    method='mean',
    multioutput='uniform_average',
    force_finite=False,
    check_outliers=False,
    check_distribution=False,
    check_correlation=False,
    check_missing_large=False,
    sample_size=1000,
    epsilon=1e-15
):
    """
    Compute Mean Squared Logarithmic Error (MSLE) or Root Mean Squared Logarithmic Error (RMSLE).

    Parameters:
    -----------
    y_true : array-like
        Ground truth (correct) target values. Must be > -1.
    y_pred : array-like
        Estimated target values. Must be > -1.
    sample_weights : array-like, optional
        Sample weights. If provided, the errors are weighted by these values.
    squared : bool, optional
        If True, returns MSLE. If False, returns RMSLE.
    method : str, optional
        Method to compute the error. Options: 'mean', 'sum', 'none'.
    multioutput : str, optional
        Defines how to aggregate errors for multi-output targets.
        Options: 'uniform_average', 'raw_values'.
    force_finite : bool, optional
        If True, replaces invalid values with finite values.
    check_outliers : bool, optional
        If True, check for outliers in y_true.
    check_distribution : bool, optional
        If True, check if y_true follows a normal distribution.
    check_correlation : bool, optional
        If True, check for high correlation between y_true and y_pred.
    check_missing_large : bool, optional
        If True, check for missing values in large data using sampling.
    sample_size : int, optional
        Sample size for checking missing values in large data.
    epsilon : float, optional
        Small constant to ensure numerical stability (default: 1e-15).

    Returns:
    --------
    float or array-like
        The computed MSLE or RMSLE.

    Raises:
    -------
    ValueError
        If inputs contain values <= -1 which would result in undefined logarithms.
    """
    # Convert inputs to numpy arrays with float64 dtype
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Initialize and run validator
    validator = DataValidator()
    validator.validate_all(
        y_true,
        y_pred,
        log_based=True,  # Enable special checks for log-based metrics
        check_outliers=check_outliers,
        check_distribution=check_distribution,
        check_correlation=check_correlation,
        check_missing_large=check_missing_large,
        sample_size=sample_size
    )

    # Check for values <= -1 which would make log1p undefined
    if np.any(y_true <= -1 + epsilon) or np.any(y_pred <= -1 + epsilon):
        raise ValueError("All values must be greater than -1 for log-based metrics")

    # Handle invalid values if requested
    if force_finite:
        y_true = np.clip(y_true, -1 + epsilon, np.inf)
        y_pred = np.clip(y_pred, -1 + epsilon, np.inf)
        y_true = np.nan_to_num(y_true, nan=-1 + epsilon, posinf=np.nanmax(y_true))
        y_pred = np.nan_to_num(y_pred, nan=-1 + epsilon, posinf=np.nanmax(y_pred))

    # Calculate squared logarithmic errors with numerical stability
    log_diff = np.log1p(y_true + epsilon) - np.log1p(y_pred + epsilon)
    squared_log_errors = np.square(log_diff)

    # Apply sample weights if provided
    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights, dtype=np.float64)
        if len(sample_weights) != len(y_true):
            raise ValueError("sample_weights must have the same length as y_true and y_pred")
        squared_log_errors *= sample_weights
        squared_log_errors /= np.mean(sample_weights)  # Normalize by mean weight

    # Calculate result based on method and multioutput
    if multioutput == 'raw_values':
        if method == 'mean':
            result = np.mean(squared_log_errors, axis=0)
        elif method == 'sum':
            result = np.sum(squared_log_errors, axis=0)
        elif method == 'none':
            result = squared_log_errors
        else:
            raise ValueError("Invalid method. Choose from 'mean', 'sum', or 'none'")
    elif multioutput == 'uniform_average':
        if method == 'mean':
            result = np.mean(squared_log_errors)
        elif method == 'sum':
            result = np.sum(squared_log_errors)
        elif method == 'none':
            result = squared_log_errors
        else:
            raise ValueError("Invalid method. Choose from 'mean', 'sum', or 'none'")
    else:
        raise ValueError("Invalid multioutput option. Choose from 'uniform_average' or 'raw_values'")

    return result if squared else np.sqrt(result)

"""# mean_bias_deviation"""

def mean_bias_deviation(
    y_true,
    y_pred,
    sample_weights=None,
    method='mean',
    multioutput='uniform_average',
    force_finite=False,
    check_outliers=False,
    check_distribution=False,
    check_correlation=False,
    check_missing_large=False,
    sample_size=1000,
    relative=False
):
    """
    Compute Mean Bias Deviation (MBD), also known as Mean Bias Error (MBE).

    Measures the average deviation between true values (y_true) and predicted values (y_pred).
    A positive value indicates underprediction bias, while negative indicates overprediction.

    Parameters:
    -----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated target values.
    sample_weights : array-like, optional
        Sample weights. If provided, the errors are weighted by these values.
    method : str, optional
        Method to compute the error. Options: 'mean', 'sum', 'none'.
    multioutput : str, optional
        Defines how to aggregate errors for multi-output targets.
        Options: 'uniform_average', 'raw_values'.
    force_finite : bool, optional
        If True, replaces infinite values with finite alternatives.
    check_outliers : bool, optional
        If True, check for outliers in y_true.
    check_distribution : bool, optional
        If True, check if y_true follows a normal distribution.
    check_correlation : bool, optional
        If True, check for high correlation between y_true and y_pred.
    check_missing_large : bool, optional
        If True, check for missing values in large data using sampling.
    sample_size : int, optional
        Sample size for checking missing values in large data.
    relative : bool, optional
        If True, returns relative MBD (divided by mean of y_true).

    Returns:
    --------
    float or array-like
        The computed MBD (in same units as y_true) or relative MBD (unitless).

    Notes:
    ------
    - MBD is sensitive to the scale of the target variable
    - Consider using relative=True for comparing across different scales
    - Unlike MAE, MBD preserves the direction of errors (sign matters)
    """
    # Convert inputs to numpy arrays with float64 dtype
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Initialize and run validator
    validator = DataValidator()
    validator.validate_all(
        y_true,
        y_pred,
        check_outliers=check_outliers,
        check_distribution=check_distribution,
        check_correlation=check_correlation,
        check_missing_large=check_missing_large,
        sample_size=sample_size
    )

    # Handle infinite values if requested
    if force_finite:
        y_true = np.nan_to_num(y_true,
                              nan=np.nanmean(y_true),
                              posinf=np.nanmax(y_true),
                              neginf=np.nanmin(y_true))
        y_pred = np.nan_to_num(y_pred,
                              nan=np.nanmean(y_pred),
                              posinf=np.nanmax(y_pred),
                              neginf=np.nanmin(y_pred))

    # Calculate bias deviations
    bias_deviations = y_true - y_pred

    # Apply sample weights if provided
    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights, dtype=np.float64)
        if len(sample_weights) != len(y_true):
            raise ValueError("sample_weights must have the same length as y_true and y_pred")
        bias_deviations *= sample_weights
        if method in ['mean', 'uniform_average']:
            bias_deviations /= np.mean(sample_weights)

    # Convert to relative if requested
    if relative:
        y_mean = np.mean(np.abs(y_true))
        if np.isclose(y_mean, 0):
            raise ValueError("Cannot compute relative MBD - mean of y_true is zero")
        bias_deviations /= y_mean

    # Calculate result based on method and multioutput
    if multioutput == 'raw_values':
        if method == 'mean':
            result = np.mean(bias_deviations, axis=0)
        elif method == 'sum':
            result = np.sum(bias_deviations, axis=0)
        elif method == 'none':
            result = bias_deviations
        else:
            raise ValueError("Invalid method. Choose from 'mean', 'sum', or 'none'")
    elif multioutput == 'uniform_average':
        if method == 'mean':
            result = np.mean(bias_deviations)
        elif method == 'sum':
            result = np.sum(bias_deviations)
        elif method == 'none':
            result = bias_deviations
        else:
            raise ValueError("Invalid method. Choose from 'mean', 'sum', or 'none'")
    else:
        raise ValueError("Invalid multioutput option. Choose from 'uniform_average' or 'raw_values'")

    return result

"""# median_absolute_error"""

def median_absolute_error(
    y_true,
    y_pred,
    sample_weights=None,
    force_finite=False,
    check_outliers=False,
    check_distribution=False,
    check_correlation=False,
    check_missing_large=False,
    sample_size=1000,
    scale=None
):
    """
    Compute Median Absolute Error (MedAE), a robust regression metric.

    MedAE measures the median of absolute differences between true and predicted values,
    making it resistant to outliers. Optionally returns a scaled version (similar to MAD).

    Parameters:
    -----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated target values.
    sample_weights : array-like, optional
        Sample weights for weighted median calculation.
    force_finite : bool, optional
        If True, replaces invalid values with finite alternatives.
    check_outliers : bool, optional
        If True, check for outliers in y_true.
    check_distribution : bool, optional
        If True, check if y_true follows a normal distribution.
    check_correlation : bool, optional
        If True, check correlation between y_true and y_pred.
    check_missing_large : bool, optional
        If True, check for missing values using sampling for large datasets.
    sample_size : int, optional
        Sample size for missing value checks in large data.
    scale : {None, 'normal', 'mad'}, optional
        Scale option: None returns raw median, 'normal' scales by 1.4826 for normal
        distribution consistency, 'mad' returns Median Absolute Deviation.

    Returns:
    --------
    float
        The computed MedAE (optionally scaled).

    Notes:
    ------
    - MedAE is more robust to outliers than MAE
    - For normal distributions, scale='normal' makes MedAE comparable to std deviation
    - Weighted median is calculated using linear interpolation
    """
    # Convert inputs to numpy arrays with float64 dtype
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Initialize and run validator
    validator = DataValidator()
    validator.validate_all(
        y_true,
        y_pred,
        check_outliers=check_outliers,
        check_distribution=check_distribution,
        check_correlation=check_correlation,
        check_missing_large=check_missing_large,
        sample_size=sample_size
    )

    # Handle invalid values if requested
    if force_finite:
        y_true = np.nan_to_num(y_true,
                             nan=np.nanmedian(y_true),
                             posinf=np.nanmax(y_true),
                             neginf=np.nanmin(y_true))
        y_pred = np.nan_to_num(y_pred,
                             nan=np.nanmedian(y_pred),
                             posinf=np.nanmax(y_pred),
                             neginf=np.nanmin(y_pred))

    # Calculate absolute errors
    abs_errors = np.abs(y_true - y_pred)

    # Compute weighted or unweighted median
    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights, dtype=np.float64)
        if len(sample_weights) != len(y_true):
            raise ValueError("sample_weights must have same length as y_true and y_pred")

        # Sort values and weights together
        sort_idx = np.argsort(abs_errors)
        sorted_errors = abs_errors[sort_idx]
        sorted_weights = sample_weights[sort_idx]

        # Calculate weighted median
        cum_weights = np.cumsum(sorted_weights)
        cutoff = cum_weights[-1] / 2.0
        medae = np.interp(cutoff, cum_weights, sorted_errors)
    else:
        medae = np.median(abs_errors)

    # Apply scaling if requested
    if scale == 'normal':
        return medae * 1.4826  # For consistency with normal distribution std
    elif scale == 'mad':
        return medae
    elif scale is None:
        return medae
    else:
        raise ValueError("scale must be None, 'normal', or 'mad'")

"""# symmetric_mean_absolute_percentage_error"""

def symmetric_mean_absolute_percentage_error(
    y_true,
    y_pred,
    sample_weights=None,
    method='mean',
    multioutput='uniform_average',
    force_finite=False,
    check_outliers=False,
    check_distribution=False,
    check_correlation=False,
    check_missing_large=False,
    sample_size=1000,
    epsilon=1e-10
):
    """
    Compute Symmetric Mean Absolute Percentage Error (sMAPE).

    sMAPE is an accuracy measure based on percentage errors that treats over- and under-predictions
    equally. The output ranges from 0% to 200%.

    Parameters:
    -----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated target values.
    sample_weights : array-like, optional
        Sample weights. If provided, the errors are weighted by these values.
    method : str, optional
        Method to compute the error. Options: 'mean', 'sum', 'none'.
    multioutput : str, optional
        Defines how to aggregate errors for multi-output targets.
        Options: 'uniform_average', 'raw_values'.
    force_finite : bool, optional
        If True, replaces invalid values with finite alternatives.
    check_outliers : bool, optional
        If True, check for outliers in y_true.
    check_distribution : bool, optional
        If True, check if y_true follows a normal distribution.
    check_correlation : bool, optional
        If True, check correlation between y_true and y_pred.
    check_missing_large : bool, optional
        If True, check for missing values using sampling for large datasets.
    sample_size : int, optional
        Sample size for missing value checks in large data.
    epsilon : float, optional
        Small constant to avoid division by zero (default: 1e-10).

    Returns:
    --------
    float or array-like
        The computed sMAPE (percentage between 0% and 200%).

    Notes:
    ------
    - sMAPE is symmetric but has different interpretation than regular MAPE
    - Returns 0% for perfect predictions, 200% when predictions are completely wrong
    - More stable than MAPE when y_true contains zeros
    """
    # Convert inputs to numpy arrays with float64 dtype
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Initialize and run validator
    validator = DataValidator()
    validator.validate_all(
        y_true,
        y_pred,
        check_outliers=check_outliers,
        check_distribution=check_distribution,
        check_correlation=check_correlation,
        check_missing_large=check_missing_large,
        sample_size=sample_size
    )

    # Handle invalid values if requested
    if force_finite:
        y_true = np.nan_to_num(y_true,
                             nan=np.nanmean(y_true),
                             posinf=np.nanmax(y_true),
                             neginf=np.nanmin(y_true))
        y_pred = np.nan_to_num(y_pred,
                             nan=np.nanmean(y_pred),
                             posinf=np.nanmax(y_pred),
                             neginf=np.nanmin(y_pred))

    # Calculate denominator with epsilon for numerical stability
    denominator = (np.abs(y_true) + np.abs(y_pred) + epsilon)

    # Calculate sMAPE errors (range 0-200)
    smape_errors = 200 * np.abs(y_true - y_pred) / denominator

    # Apply sample weights if provided
    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights, dtype=np.float64)
        if len(sample_weights) != len(y_true):
            raise ValueError("sample_weights must have same length as y_true and y_pred")
        smape_errors *= sample_weights
        if method in ['mean', 'uniform_average']:
            smape_errors /= np.mean(sample_weights)

    # Calculate result based on method and multioutput
    if multioutput == 'raw_values':
        if method == 'mean':
            result = np.mean(smape_errors, axis=0)
        elif method == 'sum':
            result = np.sum(smape_errors, axis=0)
        elif method == 'none':
            result = smape_errors
        else:
            raise ValueError("Invalid method. Choose from 'mean', 'sum', or 'none'")
    elif multioutput == 'uniform_average':
        if method == 'mean':
            result = np.mean(smape_errors)
        elif method == 'sum':
            result = np.sum(smape_errors)
        elif method == 'none':
            result = smape_errors
        else:
            raise ValueError("Invalid method. Choose from 'mean', 'sum', or 'none'")
    else:
        raise ValueError("Invalid multioutput option. Choose from 'uniform_average' or 'raw_values'")

    return result

"""# relative_squared_error"""

def relative_squared_error(
    y_true,
    y_pred,
    sample_weights=None,
    force_finite=False,
    check_outliers=False,
    check_distribution=False,
    check_correlation=False,
    check_missing_large=False,
    sample_size=1000,
    epsilon=1e-10,
    baseline='mean'
):
    """
    Compute Relative Squared Error (RSE), also known as Normalized Squared Error.

    RSE measures model performance relative to a baseline model (default: mean predictor).
    Values < 1 indicate better performance than baseline, > 1 worse performance.

    Parameters:
    -----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated target values.
    sample_weights : array-like, optional
        Sample weights for weighted calculations.
    force_finite : bool, optional
        If True, replaces invalid values with finite alternatives.
    check_outliers : bool, optional
        If True, check for outliers in y_true.
    check_distribution : bool, optional
        If True, check if y_true follows a normal distribution.
    check_correlation : bool, optional
        If True, check correlation between y_true and y_pred.
    check_missing_large : bool, optional
        If True, check for missing values using sampling for large datasets.
    sample_size : int, optional
        Sample size for missing value checks in large data.
    epsilon : float, optional
        Small constant for numerical stability (default: 1e-10).
    baseline : {'mean', 'median', 'constant'}, optional
        Baseline model to compare against ('mean', 'median', or zero).

    Returns:
    --------
    float
        The computed RSE (non-negative).

    Raises:
    -------
    ValueError
        If denominator is zero (all y_true values are identical to baseline).
    """
    # Convert inputs to numpy arrays with float64 dtype
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Initialize and run validator
    validator = DataValidator()
    validator.validate_all(
        y_true,
        y_pred,
        check_outliers=check_outliers,
        check_distribution=check_distribution,
        check_correlation=check_correlation,
        check_missing_large=check_missing_large,
        sample_size=sample_size
    )

    # Handle invalid values if requested
    if force_finite:
        y_true = np.nan_to_num(y_true,
                             nan=np.nanmean(y_true),
                             posinf=np.nanmax(y_true),
                             neginf=np.nanmin(y_true))
        y_pred = np.nan_to_num(y_pred,
                             nan=np.nanmean(y_pred),
                             posinf=np.nanmax(y_pred),
                             neginf=np.nanmin(y_pred))

    # Calculate baseline predictions
    if baseline == 'mean':
        baseline_pred = np.mean(y_true)
    elif baseline == 'median':
        baseline_pred = np.median(y_true)
    elif baseline == 'constant':
        baseline_pred = 0
    else:
        raise ValueError("baseline must be 'mean', 'median', or 'constant'")

    # Calculate numerator (model errors)
    numerator = np.square(y_true - y_pred)

    # Calculate denominator (baseline errors)
    denominator = np.square(y_true - baseline_pred)

    # Apply sample weights if provided
    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights, dtype=np.float64)
        if len(sample_weights) != len(y_true):
            raise ValueError("sample_weights must have same length as y_true and y_pred")
        numerator *= sample_weights
        denominator *= sample_weights

    # Sum the weighted errors
    sum_numerator = np.sum(numerator)
    sum_denominator = np.sum(denominator)

    # Handle zero denominator case
    if sum_denominator < epsilon:
        raise ValueError(
            f"Denominator of RSE is too small (near zero). "
            f"This occurs when all y_true values are very close to the baseline ({baseline})."
        )

    return sum_numerator / sum_denominator

"""# r_squared"""

def r_squared(
    y_true,
    y_pred,
    sample_weights=None,
    adjusted=False,
    n_features=None,
    force_finite=False,
    check_outliers=False,
    check_distribution=False,
    check_correlation=False,
    check_missing_large=False,
    sample_size=1000,
    epsilon=1e-10,
    method='standard'
):
    """
    Compute R-squared (coefficient of determination) and its variants.

    R² measures the proportion of variance in the dependent variable that is predictable
    from the independent variables. Supports weighted, adjusted, and alternative calculations.

    Parameters:
    -----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated target values.
    sample_weights : array-like, optional
        Sample weights for weighted R² calculation.
    adjusted : bool, optional
        If True, computes Adjusted R-squared (default: False).
    n_features : int, optional
        Number of features/predictors (required if adjusted=True).
    force_finite : bool, optional
        If True, replaces invalid values with finite alternatives.
    check_outliers : bool, optional
        If True, check for outliers in y_true.
    check_distribution : bool, optional
        If True, check if y_true follows a normal distribution.
    check_correlation : bool, optional
        If True, check correlation between y_true and y_pred.
    check_missing_large : bool, optional
        If True, check for missing values using sampling for large datasets.
    sample_size : int, optional
        Sample size for missing value checks in large data.
    epsilon : float, optional
        Small constant for numerical stability (default: 1e-10).
    method : {'standard', 'variance'}, optional
        Calculation method: 'standard' (1 - SS_res/SS_tot) or
        'variance' (var(y_pred)/var(y_true)).

    Returns:
    --------
    float
        The computed R² or Adjusted R² (between -∞ and 1).

    Raises:
    -------
    ValueError
        - If variance of y_true is zero (perfect prediction)
        - If n_features not provided when adjusted=True
        - If sample size <= n_features + 1 when adjusted=True
    """
    # Convert inputs to numpy arrays with float64 dtype
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Initialize and run validator
    validator = DataValidator()
    validator.validate_all(
        y_true,
        y_pred,
        check_outliers=check_outliers,
        check_distribution=check_distribution,
        check_correlation=check_correlation,
        check_missing_large=check_missing_large,
        sample_size=sample_size
    )

    # Handle invalid values if requested
    if force_finite:
        y_true = np.nan_to_num(y_true,
                             nan=np.nanmean(y_true),
                             posinf=np.nanmax(y_true),
                             neginf=np.nanmin(y_true))
        y_pred = np.nan_to_num(y_pred,
                             nan=np.nanmean(y_pred),
                             posinf=np.nanmax(y_pred),
                             neginf=np.nanmin(y_pred))

    # Check for zero variance in y_true
    y_true_var = np.var(y_true)
    if y_true_var < epsilon:
        raise ValueError("Variance of y_true is zero. R² is undefined in this case.")

    # Calculate R² using specified method
    if method == 'standard':
        # Standard calculation (1 - SS_res/SS_tot)
        if sample_weights is not None:
            sample_weights = np.asarray(sample_weights, dtype=np.float64)
            if len(sample_weights) != len(y_true):
                raise ValueError("sample_weights must have same length as y_true and y_pred")

            # Weighted sums of squares
            weighted_mean = np.average(y_true, weights=sample_weights)
            ss_total = np.sum(sample_weights * (y_true - weighted_mean)**2)
            ss_residual = np.sum(sample_weights * (y_true - y_pred)**2)
        else:
            ss_total = np.sum((y_true - np.mean(y_true))**2)
            ss_residual = np.sum((y_true - y_pred)**2)

        r2 = 1 - (ss_residual / (ss_total + epsilon))

    elif method == 'variance':
        # Variance ratio method (var(y_pred)/var(y_true))
        if sample_weights is not None:
            sample_weights = np.asarray(sample_weights, dtype=np.float64)
            y_pred_var = np.average((y_pred - np.average(y_pred, weights=sample_weights))**2,
                                weights=sample_weights)
            y_true_var = np.average((y_true - np.average(y_true, weights=sample_weights))**2,
                                weights=sample_weights)
        else:
            y_pred_var = np.var(y_pred)
            y_true_var = np.var(y_true)

        r2 = y_pred_var / (y_true_var + epsilon)
    else:
        raise ValueError("method must be 'standard' or 'variance'")

    # Calculate Adjusted R² if requested
    if adjusted:
        if n_features is None:
            raise ValueError("n_features must be provided for adjusted R² calculation")
        n_samples = len(y_true)
        if n_samples <= n_features + 1:
            raise ValueError(
                f"Sample size ({n_samples}) must be greater than "
                f"n_features + 1 ({n_features + 1}) for adjusted R²"
            )
        return 1 - ((1 - r2) * (n_samples - 1) / (n_samples - n_features - 1))

    return r2

"""# explained_variance"""

def explained_variance(
    y_true,
    y_pred,
    sample_weights=None,
    multioutput='uniform_average',
    force_finite=False,
    check_outliers=False,
    check_distribution=False,
    check_correlation=False,
    check_missing_large=False,
    sample_size=1000,
    epsilon=1e-10,
    method='standard'
):
    """
    Compute Explained Variance Score, a measure of model quality.

    The explained variance score measures the proportion of variance in y_true that is explained by y_pred.
    Best possible score is 1.0, lower values indicate worse performance.

    Parameters:
    -----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated target values.
    sample_weights : array-like, optional
        Sample weights for weighted calculations.
    multioutput : str, optional
        Defines aggregation for multi-output targets:
        'uniform_average' : average of scores
        'raw_values' : return full scores array
        'variance_weighted' : weight by variance of each target
    force_finite : bool, optional
        If True, replaces invalid values with finite alternatives.
    check_outliers : bool, optional
        If True, check for outliers in y_true.
    check_distribution : bool, optional
        If True, check if y_true follows a normal distribution.
    check_correlation : bool, optional
        If True, check correlation between y_true and y_pred.
    check_missing_large : bool, optional
        If True, check for missing values using sampling for large datasets.
    sample_size : int, optional
        Sample size for missing value checks in large data.
    epsilon : float, optional
        Small constant for numerical stability (default: 1e-10).
    method : {'standard', 'ratio'}, optional
        Calculation method: 'standard' (1 - Var(e)/Var(y)) or 'ratio' (Var(y_pred)/Var(y_true)).

    Returns:
    --------
    float or array-like
        The explained variance score (between -∞ and 1).

    Notes:
    ------
    - Score of 1 indicates perfect prediction
    - Score of 0 means model performs as well as predicting the mean
    - Negative scores indicate worse performance than predicting the mean
    - Variance-weighted option accounts for scale differences in multi-output
    """
    # Convert inputs to numpy arrays with float64 dtype
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Initialize and run validator
    validator = DataValidator()
    validator.validate_all(
        y_true,
        y_pred,
        check_outliers=check_outliers,
        check_distribution=check_distribution,
        check_correlation=check_correlation,
        check_missing_large=check_missing_large,
        sample_size=sample_size
    )

    # Handle invalid values if requested
    if force_finite:
        y_true = np.nan_to_num(y_true,
                             nan=np.nanmean(y_true),
                             posinf=np.nanmax(y_true),
                             neginf=np.nanmin(y_true))
        y_pred = np.nan_to_num(y_pred,
                             nan=np.nanmean(y_pred),
                             posinf=np.nanmax(y_pred),
                             neginf=np.nanmin(y_pred))

    # Calculate variance of y_true with weights if provided
    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights, dtype=np.float64)
        if len(sample_weights) != len(y_true):
            raise ValueError("sample_weights must have same length as y_true and y_pred")

        weighted_mean = np.average(y_true, weights=sample_weights, axis=0)
        variance_y_true = np.average((y_true - weighted_mean)**2,
                                   weights=sample_weights, axis=0)
    else:
        variance_y_true = np.var(y_true, axis=0)

    # Handle zero variance cases
    zero_variance_mask = variance_y_true < epsilon
    if np.any(zero_variance_mask):
        if multioutput == 'uniform_average':
            if np.all(zero_variance_mask):
                return 0.0
        else:
            result = np.zeros_like(variance_y_true)
            if multioutput == 'variance_weighted':
                return 0.0 if np.all(zero_variance_mask) else result
            return result

    # Calculate explained variance using specified method
    if method == 'standard':
        # Standard method: 1 - Var(y_true - y_pred)/Var(y_true)
        if sample_weights is not None:
            errors = y_true - y_pred
            weighted_mean_error = np.average(errors, weights=sample_weights, axis=0)
            variance_error = np.average((errors - weighted_mean_error)**2,
                                      weights=sample_weights, axis=0)
        else:
            variance_error = np.var(y_true - y_pred, axis=0)

        explained_var = 1 - (variance_error / (variance_y_true + epsilon))

    elif method == 'ratio':
        # Variance ratio method: Var(y_pred)/Var(y_true)
        if sample_weights is not None:
            weighted_mean_pred = np.average(y_pred, weights=sample_weights, axis=0)
            variance_pred = np.average((y_pred - weighted_mean_pred)**2,
                                     weights=sample_weights, axis=0)
        else:
            variance_pred = np.var(y_pred, axis=0)

        explained_var = variance_pred / (variance_y_true + epsilon)
    else:
        raise ValueError("method must be 'standard' or 'ratio'")

    # Aggregate results based on multioutput
    if multioutput == 'raw_values':
        return explained_var
    elif multioutput == 'uniform_average':
        return np.mean(explained_var)
    elif multioutput == 'variance_weighted':
        return np.average(explained_var, weights=variance_y_true)
    else:
        raise ValueError(
            "Invalid multioutput option. Choose from 'uniform_average', "
            "'raw_values', or 'variance_weighted'"
        )

"""# buber_loss"""

def huber_loss(
    y_true,
    y_pred,
    delta=1.0,
    sample_weights=None,
    method='mean',
    multioutput='uniform_average',
    force_finite=False,
    check_outliers=False,
    check_distribution=False,
    check_correlation=False,
    check_missing_large=False,
    sample_size=1000,
    epsilon=1e-10
):
    """
    Compute Huber Loss, a robust regression metric that combines MSE and MAE.

    Huber loss is less sensitive to outliers than MSE while remaining differentiable at zero.
    It behaves like MSE for small errors (|error| ≤ δ) and like MAE for large errors (|error| > δ).

    Parameters:
    -----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated target values.
    delta : float, optional
        Threshold that determines the transition point between quadratic and linear loss.
        Must be > 0 (default: 1.0).
    sample_weights : array-like, optional
        Sample weights for weighted loss calculation.
    method : str, optional
        Aggregation method: 'mean' (default), 'sum', or 'none'.
    multioutput : str, optional
        Defines aggregation for multi-output targets:
        'uniform_average' : average of scores (default)
        'raw_values' : return full scores array
    force_finite : bool, optional
        If True, replaces invalid values with finite alternatives.
    check_outliers : bool, optional
        If True, check for outliers in y_true.
    check_distribution : bool, optional
        If True, check if y_true follows a normal distribution.
    check_correlation : bool, optional
        If True, check correlation between y_true and y_pred.
    check_missing_large : bool, optional
        If True, check for missing values using sampling for large datasets.
    sample_size : int, optional
        Sample size for missing value checks in large data.
    epsilon : float, optional
        Small constant for numerical stability (default: 1e-10).

    Returns:
    --------
    float or array-like
        The computed Huber loss. Returns:
        - scalar if method='mean'/'sum' and multioutput='uniform_average'
        - array if method='none' or multioutput='raw_values'

    Raises:
    -------
    ValueError
        If delta ≤ 0 or if sample_weights length doesn't match inputs.

    Examples:
    --------
    >>> y_true = [1, 2, 3, 4, 5]
    >>> y_pred = [1.1, 1.9, 3.2, 3.8, 5.5]
    >>> huber_loss(y_true, y_pred, delta=1.0)
    0.057
    """
    # Validate delta parameter
    if delta <= 0:
        raise ValueError(f"delta must be > 0, got {delta}")

    # Convert inputs to numpy arrays with float64 dtype
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Initialize and run validator
    validator = DataValidator()
    validator.validate_all(
        y_true,
        y_pred,
        check_outliers=check_outliers,
        check_distribution=check_distribution,
        check_correlation=check_correlation,
        check_missing_large=check_missing_large,
        sample_size=sample_size
    )

    # Handle invalid values if requested
    if force_finite:
        y_true = np.nan_to_num(y_true,
                             nan=np.nanmean(y_true),
                             posinf=np.nanmax(y_true),
                             neginf=np.nanmin(y_true))
        y_pred = np.nan_to_num(y_pred,
                             nan=np.nanmean(y_pred),
                             posinf=np.nanmax(y_pred),
                             neginf=np.nanmin(y_pred))

    # Calculate errors
    errors = y_true - y_pred
    abs_errors = np.abs(errors)

    # Calculate Huber loss
    quadratic = (0.5 * errors**2)
    linear = (delta * abs_errors - 0.5 * delta**2)
    loss = np.where(abs_errors <= delta, quadratic, linear)

    # Apply sample weights if provided
    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights, dtype=np.float64)
        if len(sample_weights) != len(y_true):
            raise ValueError("sample_weights must have same length as y_true and y_pred")
        loss *= sample_weights
        if method == 'mean':
            loss /= np.mean(sample_weights)

    # Aggregate results
    if multioutput == 'raw_values':
        if method == 'mean':
            return np.mean(loss, axis=0)
        elif method == 'sum':
            return np.sum(loss, axis=0)
        elif method == 'none':
            return loss
        else:
            raise ValueError("Invalid method. Choose from 'mean', 'sum', or 'none'")
    elif multioutput == 'uniform_average':
        if method == 'mean':
            return np.mean(loss)
        elif method == 'sum':
            return np.sum(loss)
        elif method == 'none':
            return loss
        else:
            raise ValueError("Invalid method. Choose from 'mean', 'sum', or 'none'")
    else:
        raise ValueError("Invalid multioutput option. Choose from 'uniform_average' or 'raw_values'")

"""# log_cosh_loss"""

def log_cosh_loss(
    y_true,
    y_pred,
    sample_weights=None,
    method='mean',
    multioutput='uniform_average',
    force_finite=False,
    check_outliers=False,
    check_distribution=False,
    check_correlation=False,
    check_missing_large=False,
    sample_size=1000,
    epsilon=1e-10,
    approximate=False
):
    """
    Compute Log-Cosh Loss, a smooth approximation of Huber loss.

    Log-Cosh loss behaves like MSE for small errors and like MAE for large errors,
    while being twice differentiable everywhere. Useful for robust regression.

    Parameters:
    -----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated target values.
    sample_weights : array-like, optional
        Sample weights for weighted loss calculation.
    method : str, optional
        Aggregation method: 'mean' (default), 'sum', or 'none'.
    multioutput : str, optional
        Defines aggregation for multi-output targets:
        'uniform_average' : average of scores (default)
        'raw_values' : return full scores array
    force_finite : bool, optional
        If True, replaces invalid values with finite alternatives.
    check_outliers : bool, optional
        If True, check for outliers in y_true.
    check_distribution : bool, optional
        If True, check if y_true follows a normal distribution.
    check_correlation : bool, optional
        If True, check correlation between y_true and y_pred.
    check_missing_large : bool, optional
        If True, check for missing values using sampling for large datasets.
    sample_size : int, optional
        Sample size for missing value checks in large data.
    epsilon : float, optional
        Small constant for numerical stability (default: 1e-10).
    approximate : bool, optional
        If True, uses faster approximation for large errors (default: False).

    Returns:
    --------
    float or array-like
        The computed Log-Cosh loss. Returns:
        - scalar if method='mean'/'sum' and multioutput='uniform_average'
        - array if method='none' or multioutput='raw_values'

    Notes:
    ------
    - For small errors (|x| < 1): ≈ x²/2
    - For large errors (|x| ≥ 1): ≈ |x| - log(2)
    - Twice differentiable everywhere (unlike Huber loss)
    - Less sensitive to outliers than MSE
    """
    # Convert inputs to numpy arrays with float64 dtype
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Initialize and run validator
    validator = DataValidator()
    validator.validate_all(
        y_true,
        y_pred,
        check_outliers=check_outliers,
        check_distribution=check_distribution,
        check_correlation=check_correlation,
        check_missing_large=check_missing_large,
        sample_size=sample_size
    )

    # Handle invalid values if requested
    if force_finite:
        y_true = np.nan_to_num(y_true,
                            nan=np.nanmean(y_true),
                            posinf=np.nanmax(y_true),
                            neginf=np.nanmin(y_true))
        y_pred = np.nan_to_num(y_pred,
                            nan=np.nanmean(y_pred),
                            posinf=np.nanmax(y_pred),
                            neginf=np.nanmin(y_pred))

    # Calculate errors
    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    # Calculate log-cosh loss (with approximation option for large errors)
    if approximate:
        # Faster approximation that avoids overflow for large errors
        loss = np.where(
            abs_errors < 1,
            0.5 * errors**2,
            abs_errors - np.log(2)
        )
    else:
        # Exact calculation (slower but more precise)
        loss = np.log(np.cosh(errors + epsilon))

    # Apply sample weights if provided
    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights, dtype=np.float64)
        if len(sample_weights) != len(y_true):
            raise ValueError("sample_weights must have same length as y_true and y_pred")
        loss *= sample_weights
        if method == 'mean':
            loss /= np.mean(sample_weights)

    # Aggregate results
    if multioutput == 'raw_values':
        if method == 'mean':
            return np.mean(loss, axis=0)
        elif method == 'sum':
            return np.sum(loss, axis=0)
        elif method == 'none':
            return loss
        else:
            raise ValueError("Invalid method. Choose from 'mean', 'sum', or 'none'")
    elif multioutput == 'uniform_average':
        if method == 'mean':
            return np.mean(loss)
        elif method == 'sum':
            return np.sum(loss)
        elif method == 'none':
            return loss
        else:
            raise ValueError("Invalid method. Choose from 'mean', 'sum', or 'none'")
    else:
        raise ValueError("Invalid multioutput option. Choose from 'uniform_average' or 'raw_values'")

"""# max_error"""

def max_error(
    y_true,
    y_pred,
    sample_weights=None,
    force_finite=False,
    check_outliers=False,
    check_distribution=False,
    check_correlation=False,
    check_missing_large=False,
    sample_size=1000,
    return_index=False
):
    """
    Compute the maximum residual error (worst prediction error).

    This metric calculates the maximum absolute difference between true and predicted values,
    identifying the single worst prediction in the dataset.

    Parameters:
    -----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated target values.
    sample_weights : array-like, optional
        Sample weights. If provided, the errors are weighted by these values.
    force_finite : bool, optional
        If True, replaces infinite values with finite alternatives.
    check_outliers : bool, optional
        If True, check for outliers in y_true.
    check_distribution : bool, optional
        If True, check if y_true follows a normal distribution.
    check_correlation : bool, optional
        If True, check for high correlation between y_true and y_pred.
    check_missing_large : bool, optional
        If True, check for missing values using sampling for large datasets.
    sample_size : int, optional
        Sample size for missing value checks in large data.
    return_index : bool, optional
        If True, returns a tuple of (max_error, index) where index is the position of max error.

    Returns:
    --------
    float or tuple
        The maximum absolute error. If return_index=True, returns (max_error, index).

    Examples:
    --------
    >>> y_true = [3, 2, 7, 1]
    >>> y_pred = [4, 2, 7, 1.5]
    >>> max_error(y_true, y_pred)
    1.0
    >>> max_error(y_true, y_pred, return_index=True)
    (1.0, 0)
    """
    # Convert inputs to numpy arrays with float64 dtype
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Initialize and run validator
    validator = DataValidator()
    validator.validate_all(
        y_true,
        y_pred,
        check_outliers=check_outliers,
        check_distribution=check_distribution,
        check_correlation=check_correlation,
        check_missing_large=check_missing_large,
        sample_size=sample_size
    )

    # Handle invalid values if requested
    if force_finite:
        y_true = np.nan_to_num(y_true,
                             nan=np.nanmean(y_true),
                             posinf=np.nanmax(y_true),
                             neginf=np.nanmin(y_true))
        y_pred = np.nan_to_num(y_pred,
                             nan=np.nanmean(y_pred),
                             posinf=np.nanmax(y_pred),
                             neginf=np.nanmin(y_pred))

    # Calculate absolute errors
    abs_errors = np.abs(y_true - y_pred)

    # Apply sample weights if provided
    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights, dtype=np.float64)
        if len(sample_weights) != len(y_true):
            raise ValueError("sample_weights must have same length as y_true and y_pred")
        abs_errors *= sample_weights

    # Find maximum error
    max_err = np.max(abs_errors)

    if return_index:
        max_idx = np.argmax(abs_errors)
        return (max_err, max_idx)
    return max_err

"""# mean_tweedie_deviance"""

def mean_tweedie_deviance(
    y_true,
    y_pred,
    p=0,
    sample_weights=None,
    method='mean',
    multioutput='uniform_average',
    force_finite=False,
    check_outliers=False,
    check_distribution=False,
    check_correlation=False,
    check_missing_large=False,
    sample_size=1000,
    epsilon=1e-10,
    safe_log=True
):
    """
    Compute Mean Tweedie Deviance for various exponential dispersion distributions.

    Tweedie deviance generalizes several common loss functions for regression:
    - p=0: Normal distribution (equivalent to MSE)
    - p=1: Poisson distribution
    - p=2: Gamma distribution
    - 1<p<2: Compound Poisson-Gamma distributions

    Parameters:
    -----------
    y_true : array-like
        Ground truth (correct) target values. Must be >0 for p≥1.
    y_pred : array-like
        Estimated target values. Must be >0 for p≠0.
    p : float, optional
        Tweedie power parameter (default=0). Common values:
        0: Normal, 1: Poisson, 2: Gamma, (1,2): Compound Poisson-Gamma
    sample_weights : array-like, optional
        Sample weights for weighted deviance calculation.
    method : str, optional
        Aggregation method: 'mean' (default), 'sum', or 'none'.
    multioutput : str, optional
        Defines aggregation for multi-output targets:
        'uniform_average' : average of scores (default)
        'raw_values' : return full scores array
    force_finite : bool, optional
        If True, replaces invalid values with finite alternatives.
    check_outliers : bool, optional
        If True, check for outliers in y_true.
    check_distribution : bool, optional
        If True, check if y_true follows expected distribution.
    check_correlation : bool, optional
        If True, check correlation between y_true and y_pred.
    check_missing_large : bool, optional
        If True, check for missing values using sampling for large datasets.
    sample_size : int, optional
        Sample size for missing value checks in large data.
    epsilon : float, optional
        Small constant for numerical stability (default: 1e-10).
    safe_log : bool, optional
        If True, uses safe log calculations with clipping (default: True).

    Returns:
    --------
    float or array-like
        The computed Tweedie deviance. Returns:
        - scalar if method='mean'/'sum' and multioutput='uniform_average'
        - array if method='none' or multioutput='raw_values'

    Raises:
    -------
    ValueError
        - For invalid p values
        - For negative values when p≥1
        - For zero values when using log with p≠0
    """
    # Validate power parameter
    if p < 0:
        raise ValueError(f"Power parameter p must be >=0, got {p}")
    if p > 2:
        raise ValueError("Power parameter p>2 is not currently supported")

    # Convert inputs to numpy arrays with float64 dtype
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Initialize and run validator
    validator = DataValidator()
    validator.validate_all(
        y_true,
        y_pred,
        check_outliers=check_outliers,
        check_distribution=check_distribution,
        check_correlation=check_correlation,
        check_missing_large=check_missing_large,
        sample_size=sample_size
    )

    # Handle invalid values if requested
    if force_finite:
        y_true = np.nan_to_num(y_true,
                             nan=np.nanmean(y_true),
                             posinf=np.nanmax(y_true),
                             neginf=np.nanmin(y_true))
        y_pred = np.nan_to_num(y_pred,
                             nan=np.nanmean(y_pred),
                             posinf=np.nanmax(y_pred),
                             neginf=np.nanmin(y_pred))

    # Check for invalid values based on p
    if p >= 1:
        if np.any(y_true < 0):
            raise ValueError("y_true contains negative values (invalid for p≥1)")
        if np.any(y_pred <= 0):
            raise ValueError("y_pred contains non-positive values (invalid for p≥1)")
    elif p != 0:
        if np.any(y_pred <= 0):
            raise ValueError("y_pred contains non-positive values (invalid for p≠0)")

    # Calculate Tweedie deviance based on power parameter
    if p == 0:
        # Normal distribution (MSE)
        deviance = (y_true - y_pred) ** 2
    elif p == 1:
        # Poisson distribution
        if safe_log:
            ratio = np.clip(y_true / (y_pred + epsilon), epsilon, 1/epsilon)
            deviance = 2 * (y_true * np.log(ratio) - (y_true - y_pred))
        else:
            deviance = 2 * (y_true * np.log(y_true / y_pred) - (y_true - y_pred))
    elif p == 2:
        # Gamma distribution
        if safe_log:
            ratio = np.clip(y_true / (y_pred + epsilon), epsilon, 1/epsilon)
            deviance = 2 * (ratio - np.log(ratio) - 1)
        else:
            deviance = 2 * (y_true / y_pred - np.log(y_true / y_pred) - 1)
    elif 1 < p < 2:
        # Compound Poisson-Gamma distribution
        if safe_log:
            y_pred = np.clip(y_pred, epsilon, None)
            term1 = np.maximum(y_true, epsilon) ** (2 - p) / ((1 - p) * (2 - p))
            term2 = y_true * y_pred ** (1 - p) / (1 - p)
            term3 = y_pred ** (2 - p) / (2 - p)
            deviance = 2 * (term1 - term2 + term3)
        else:
            deviance = 2 * (
                (y_true ** (2 - p)) / ((1 - p) * (2 - p)) -
                y_true * y_pred ** (1 - p) / (1 - p) +
                y_pred ** (2 - p) / (2 - p)
            )

    # Apply sample weights if provided
    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights, dtype=np.float64)
        if len(sample_weights) != len(y_true):
            raise ValueError("sample_weights must have same length as y_true and y_pred")
        deviance *= sample_weights
        if method == 'mean':
            deviance /= np.mean(sample_weights)

    # Aggregate results
    if multioutput == 'raw_values':
        if method == 'mean':
            return np.mean(deviance, axis=0)
        elif method == 'sum':
            return np.sum(deviance, axis=0)
        elif method == 'none':
            return deviance
        else:
            raise ValueError("Invalid method. Choose from 'mean', 'sum', or 'none'")
    elif multioutput == 'uniform_average':
        if method == 'mean':
            return np.mean(deviance)
        elif method == 'sum':
            return np.sum(deviance)
        elif method == 'none':
            return deviance
        else:
            raise ValueError("Invalid method. Choose from 'mean', 'sum', or 'none'")
    else:
        raise ValueError("Invalid multioutput option. Choose from 'uniform_average' or 'raw_values'")

"""# mean_pinball_loss"""

def mean_pinball_loss(
    y_true,
    y_pred,
    tau=0.5,
    sample_weights=None,
    method='mean',
    multioutput='uniform_average',
    force_finite=False,
    check_outliers=False,
    check_distribution=False,
    check_correlation=False,
    check_missing_large=False,
    sample_size=1000,
    epsilon=1e-10
):
    """
    Compute Mean Pinball Loss (Quantile Loss) for quantile regression evaluation.

    The pinball loss evaluates the accuracy of a quantile prediction, where tau is the target quantile.
    For tau=0.5, this reduces to mean absolute error (MAE) scaled by 0.5.

    Parameters:
    -----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated quantile values for the specified tau.
    tau : float or array-like, optional
        Target quantile(s) between 0 and 1. If array, must match y_pred shape (default: 0.5).
    sample_weights : array-like, optional
        Sample weights for weighted loss calculation.
    method : str, optional
        Aggregation method: 'mean' (default), 'sum', or 'none'.
    multioutput : str, optional
        Defines aggregation for multi-output targets:
        'uniform_average' : average of scores (default)
        'raw_values' : return full scores array
    force_finite : bool, optional
        If True, replaces invalid values with finite alternatives.
    check_outliers : bool, optional
        If True, check for outliers in y_true.
    check_distribution : bool, optional
        If True, check if y_true follows a normal distribution.
    check_correlation : bool, optional
        If True, check correlation between y_true and y_pred.
    check_missing_large : bool, optional
        If True, check for missing values using sampling for large datasets.
    sample_size : int, optional
        Sample size for missing value checks in large data.
    epsilon : float, optional
        Small constant for numerical stability (default: 1e-10).

    Returns:
    --------
    float or array-like
        The computed pinball loss. Returns:
        - scalar if method='mean'/'sum' and multioutput='uniform_average'
        - array if method='none' or multioutput='raw_values'

    Raises:
    -------
    ValueError
        - If tau is not between 0 and 1
        - If tau array doesn't match y_pred shape
        - If sample_weights length doesn't match inputs

    Examples:
    --------
    >>> y_true = [1, 2, 3, 4, 5]
    >>> y_pred = [1.1, 1.9, 3.2, 3.8, 5.5]  # Predictions for 50th percentile
    >>> mean_pinball_loss(y_true, y_pred, tau=0.5)
    0.15
    """
    # Convert inputs to numpy arrays with float64 dtype
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    tau_arr = np.asarray(tau, dtype=np.float64)

    # Validate tau parameter(s)
    if np.any((tau_arr <= 0) | (tau_arr >= 1)):
        raise ValueError("All tau values must be strictly between 0 and 1")

    # Check if tau array matches y_pred shape for multiple quantiles
    if tau_arr.ndim > 0 and tau_arr.shape != y_pred.shape:
        raise ValueError("When tau is an array, its shape must match y_pred")

    # Initialize and run validator
    validator = DataValidator()
    validator.validate_all(
        y_true,
        y_pred,
        check_outliers=check_outliers,
        check_distribution=check_distribution,
        check_correlation=check_correlation,
        check_missing_large=check_missing_large,
        sample_size=sample_size
    )

    # Handle invalid values if requested
    if force_finite:
        y_true = np.nan_to_num(y_true,
                             nan=np.nanmedian(y_true),
                             posinf=np.nanmax(y_true),
                             neginf=np.nanmin(y_true))
        y_pred = np.nan_to_num(y_pred,
                             nan=np.nanmedian(y_pred),
                             posinf=np.nanmax(y_pred),
                             neginf=np.nanmin(y_pred))

    # Calculate errors
    errors = y_true - y_pred

    # Calculate pinball loss
    loss = np.where(
        errors >= 0,
        tau_arr * errors,
        (tau_arr - 1) * errors
    )

    # Apply sample weights if provided
    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights, dtype=np.float64)
        if len(sample_weights) != len(y_true):
            raise ValueError("sample_weights must have same length as y_true and y_pred")
        loss *= sample_weights
        if method == 'mean':
            loss /= np.mean(sample_weights)

    # Aggregate results
    if multioutput == 'raw_values':
        if method == 'mean':
            return np.mean(loss, axis=0)
        elif method == 'sum':
            return np.sum(loss, axis=0)
        elif method == 'none':
            return loss
        else:
            raise ValueError("Invalid method. Choose from 'mean', 'sum', or 'none'")
    elif multioutput == 'uniform_average':
        if method == 'mean':
            return np.mean(loss)
        elif method == 'sum':
            return np.sum(loss)
        elif method == 'none':
            return loss
        else:
            raise ValueError("Invalid method. Choose from 'mean', 'sum', or 'none'")
    else:
        raise ValueError("Invalid multioutput option. Choose from 'uniform_average' or 'raw_values'")