

# RegressionDataValidator
"""

import numpy as np
import pandas as pd
from scipy.stats import kstest
from scipy.spatial.distance import cdist
from typing import Union, Optional
from scipy.ndimage import label
import torch
import warnings

class RegressionDataValidator:
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

    def validate_all(self, y_true, y_pred, log_based=False, check_missing_large= False, check_outliers=False, check_distribution=False, check_correlation=False, sample_size=1000):
        """Run all validation checks"""
        self.check_data_type(y_true, y_pred)
        self.check_shapes(y_true, y_pred)
        self.check_missing_values(y_true, y_pred)
        self.check_inf_values(y_true, y_pred)
        self.check_lengths(y_true, y_pred)
        self.check_numeric_values(y_true, y_pred)
        self.check_variance(y_true, y_pred)
        if check_missing_large:
            self.check_missing_values_large_data(y_true, y_pred, sample_size)
        else:
            self.check_missing_values(y_true, y_pred)
        if log_based:
            self.check_non_negative(y_true, y_pred)
        if check_outliers:
            self.check_outliers(y_true, y_pred)
        if check_distribution:
            self.check_distribution(y_true, y_pred)
        if check_correlation:
            self.check_correlation(y_true, y_pred)
        return True  # Return True if all checks pass


# Metadata for regression metrics
REGRESSION_METRIC_DETAILS = {
    "mean_absolute_error": {
        "y_true": "Ground truth (correct) target values. Expected as a NumPy array, list, or PyTorch tensor.",
        "y_pred": "Predicted target values. Expected as a NumPy array, list, or PyTorch tensor.",
        "sample_weights": "Optional array of weights for each sample. If None, all samples are weighted equally.",
        "method": "Method to compute the error ('mean', 'median', etc.). Default is 'mean'.",
        "multioutput": "Defines aggregation for multi-output ('uniform_average', 'raw_values'). Default is 'uniform_average'.",
        "force_finite": "If True, forces the result to be finite even if there are issues like division by zero.",
        "check_outliers": "If True, checks for outliers in y_true using Z-score.",
        "check_distribution": "If True, checks if y_true follows a specific distribution (e.g., normal).",
        "check_correlation": "If True, checks for high correlation between y_true and y_pred.",
        "check_missing_large": "If True, checks for missing values in large datasets using sampling.",
        "sample_size": "Number of samples to use for large dataset checks."
    },
    "mean_squared_error": {
        "y_true": "Ground truth (correct) target values. Expected as a NumPy array, list, or PyTorch tensor.",
        "y_pred": "Predicted target values. Expected as a NumPy array, list, or PyTorch tensor.",
        "sample_weights": "Optional array of weights for each sample. If None, all samples are weighted equally.",
        "squared": "If True, returns MSE; if False, returns RMSE.",
        "method": "Method to compute the error ('mean', 'median', etc.). Default is 'mean'.",
        "multioutput": "Defines aggregation for multi-output ('uniform_average', 'raw_values'). Default is 'uniform_average'.",
        "force_finite": "If True, forces the result to be finite even if there are issues like division by zero.",
        "check_outliers": "If True, checks for outliers in y_true using Z-score.",
        "check_distribution": "If True, checks if y_true follows a specific distribution (e.g., normal).",
        "check_correlation": "If True, checks for high correlation between y_true and y_pred.",
        "check_missing_large": "If True, checks for missing values in large datasets using sampling.",
        "sample_size": "Number of samples to use for large dataset checks."
    },
    "mean_absolute_percentage_error": {
        "y_true": "Ground truth (correct) target values. Expected as a NumPy array, list, or PyTorch tensor.",
        "y_pred": "Predicted target values. Expected as a NumPy array, list, or PyTorch tensor.",
        "sample_weights": "Optional array of weights for each sample. If None, all samples are weighted equally.",
        "method": "Method to compute the error ('mean', 'median', etc.). Default is 'mean'.",
        "multioutput": "Defines aggregation for multi-output ('uniform_average', 'raw_values'). Default is 'uniform_average'.",
        "force_finite": "If True, forces the result to be finite even if there are issues like division by zero.",
        "check_outliers": "If True, checks for outliers in y_true using Z-score.",
        "check_distribution": "If True, checks if y_true follows a specific distribution (e.g., normal).",
        "check_correlation": "If True, checks for high correlation between y_true and y_pred.",
        "check_missing_large": "If True, checks for missing values in large datasets using sampling.",
        "sample_size": "Number of samples to use for large dataset checks.",
        "epsilon": "Small value added to avoid division by zero. Default is 1e-10."
    },
    "mean_squared_log_error": {
        "y_true": "Ground truth (correct) target values. Expected as a NumPy array, list, or PyTorch tensor.",
        "y_pred": "Predicted target values. Expected as a NumPy array, list, or PyTorch tensor.",
        "sample_weights": "Optional array of weights for each sample. If None, all samples are weighted equally.",
        "squared": "If True, returns MSLE; if False, returns RMSLE.",
        "method": "Method to compute the error ('mean', 'median', etc.). Default is 'mean'.",
        "multioutput": "Defines aggregation for multi-output ('uniform_average', 'raw_values'). Default is 'uniform_average'.",
        "force_finite": "If True, forces the result to be finite even if there are issues like division by zero.",
        "check_outliers": "If True, checks for outliers in y_true using Z-score.",
        "check_distribution": "If True, checks if y_true follows a specific distribution (e.g., normal).",
        "check_correlation": "If True, checks for high correlation between y_true and y_pred.",
        "check_missing_large": "If True, checks for missing values in large datasets using sampling.",
        "sample_size": "Number of samples to use for large dataset checks.",
        "epsilon": "Small value added to avoid division by zero. Default is 1e-15."
    },
    "mean_bias_deviation": {
        "y_true": "Ground truth (correct) target values. Expected as a NumPy array, list, or PyTorch tensor.",
        "y_pred": "Predicted target values. Expected as a NumPy array, list, or PyTorch tensor.",
        "sample_weights": "Optional array of weights for each sample. If None, all samples are weighted equally.",
        "method": "Method to compute the error ('mean', 'median', etc.). Default is 'mean'.",
        "multioutput": "Defines aggregation for multi-output ('uniform_average', 'raw_values'). Default is 'uniform_average'.",
        "force_finite": "If True, forces the result to be finite even if there are issues like division by zero.",
        "check_outliers": "If True, checks for outliers in y_true using Z-score.",
        "check_distribution": "If True, checks if y_true follows a specific distribution (e.g., normal).",
        "check_correlation": "If True, checks for high correlation between y_true and y_pred.",
        "check_missing_large": "If True, checks for missing values in large datasets using sampling.",
        "sample_size": "Number of samples to use for large dataset checks.",
        "relative": "If True, computes relative bias deviation. Default is False."
    },
    "median_absolute_error": {
        "y_true": "Ground truth (correct) target values. Expected as a NumPy array, list, or PyTorch tensor.",
        "y_pred": "Predicted target values. Expected as a NumPy array, list, or PyTorch tensor.",
        "sample_weights": "Optional array of weights for each sample. If None, all samples are weighted equally.",
        "force_finite": "If True, forces the result to be finite even if there are issues like division by zero.",
        "check_outliers": "If True, checks for outliers in y_true using Z-score.",
        "check_distribution": "If True, checks if y_true follows a specific distribution (e.g., normal).",
        "check_correlation": "If True, checks for high correlation between y_true and y_pred.",
        "check_missing_large": "If True, checks for missing values in large datasets using sampling.",
        "sample_size": "Number of samples to use for large dataset checks.",
        "scale": "Optional scaling factor for the error metric."
    },
    "symmetric_mean_absolute_percentage_error": {
        "y_true": "Ground truth (correct) target values. Expected as a NumPy array, list, or PyTorch tensor.",
        "y_pred": "Predicted target values. Expected as a NumPy array, list, or PyTorch tensor.",
        "sample_weights": "Optional array of weights for each sample. If None, all samples are weighted equally.",
        "method": "Method to compute the error ('mean', 'median', etc.). Default is 'mean'.",
        "multioutput": "Defines aggregation for multi-output ('uniform_average', 'raw_values'). Default is 'uniform_average'.",
        "force_finite": "If True, forces the result to be finite even if there are issues like division by zero.",
        "check_outliers": "If True, checks for outliers in y_true using Z-score.",
        "check_distribution": "If True, checks if y_true follows a specific distribution (e.g., normal).",
        "check_correlation": "If True, checks for high correlation between y_true and y_pred.",
        "check_missing_large": "If True, checks for missing values in large datasets using sampling.",
        "sample_size": "Number of samples to use for large dataset checks.",
        "epsilon": "Small value added to avoid division by zero. Default is 1e-10."
    },
    "relative_squared_error": {
        "y_true": "Ground truth (correct) target values. Expected as a NumPy array, list, or PyTorch tensor.",
        "y_pred": "Predicted target values. Expected as a NumPy array, list, or PyTorch tensor.",
        "sample_weights": "Optional array of weights for each sample. If None, all samples are weighted equally.",
        "force_finite": "If True, forces the result to be finite even if there are issues like division by zero.",
        "check_outliers": "If True, checks for outliers in y_true using Z-score.",
        "check_distribution": "If True, checks if y_true follows a specific distribution (e.g., normal).",
        "check_correlation": "If True, checks for high correlation between y_true and y_pred.",
        "check_missing_large": "If True, checks for missing values in large datasets using sampling.",
        "sample_size": "Number of samples to use for large dataset checks.",
        "epsilon": "Small value added to avoid division by zero. Default is 1e-10.",
        "baseline": "Baseline method for comparison ('mean', 'median', etc.). Default is 'mean'."
    },
    "r_squared": {
        "y_true": "Ground truth (correct) target values. Expected as a NumPy array, list, or PyTorch tensor.",
        "y_pred": "Predicted target values. Expected as a NumPy array, list, or PyTorch tensor.",
        "sample_weights": "Optional array of weights for each sample. If None, all samples are weighted equally.",
        "adjusted": "If True, computes adjusted R-squared. Requires n_features.",
        "n_features": "Number of features used in the model (required for adjusted R-squared).",
        "force_finite": "If True, forces the result to be finite even if there are issues like division by zero.",
        "check_outliers": "If True, checks for outliers in y_true using Z-score.",
        "check_distribution": "If True, checks if y_true follows a specific distribution (e.g., normal).",
        "check_correlation": "If True, checks for high correlation between y_true and y_pred.",
        "check_missing_large": "If True, checks for missing values in large datasets using sampling.",
        "sample_size": "Number of samples to use for large dataset checks.",
        "epsilon": "Small value added to avoid division by zero. Default is 1e-10.",
        "method": "Method to compute R-squared ('standard', etc.). Default is 'standard'."
    },
    "explained_variance":{},
    "buber_loss":{},
    "log_cosh_loss":{},
    "max_error":{},
    "mean_tweedie_deviance":{},
    "mean_pinball_loss":{}
}