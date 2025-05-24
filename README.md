# AllMetrics-A-Unified-Python-Library-for-Standardized-Metric-Evaluation-in-Machine-Learning
ArXiv: https://arxiv.org/abs/2505.15931

Python Version 3.11.12

AllMetrics is a comprehensive Python library designed to standardize performance metric evaluation across diverse machine learning tasks while ensuring robust data validation. It addresses critical challenges of:

* **Implementation Differences (ID)**: Variations in how metrics are computed across libraries

* **Reporting Differences (RD)**: Inconsistent formats for presenting results

* **Data Integrity**: Automated validation of input quality

  **Key Features**
  
✔ Unified API for regression, classification, clustering, segmentation, and image-to-image translation

✔ Standardized implementations resolving inconsistencies across Python/R/Matlab ecosystems

✔ Robust validation for edge cases (empty masks, class imbalance, etc.)

✔ Configurable reporting with class-specific outputs

✔ Optimized performance with efficient computations


**Installation**

### pip install **allmetrics**
---
Quick Start (Classification Example):

#### from allmetrics.classification import precision_score
#### y_true = [0, 1, 1, 0, 1]
##### y_pred = [0, 1, 0, 0, 1]
#### print(precision_score(y_true, y_pred))  # {0: 1.0, 1: 0.666...}
#### print(precision_score(y_true, y_pred, average='macro'))

Segmentation Example

#### from allmetrics.segmentation import dice_score
#### y_true = [[1, 0], [1, 1]]  Ground truth mask
#### y_pred = [[1, 0], [0, 1]]  Predicted mask
#### print(dice_score(y_true, y_pred))
----

### Supported Metrics

* **Regression**:	mean_absolute_error, mean_squared_error, mean_bias_deviation, r_squared, r_squared(adjusted), mean_absolute_percentage_error, symmetric_mean_absolute_percentage_error, huber_loss, relative_squared_error, mean_squared_log_error, log_cosh_loss, explained_variance, median_absolute_error, max_error, mean_tweedie_deviance, mean_pinball_loss.
* **Classification**:	accuracy_score, precision_score, recall_score, balanced_accuracy, matthews_correlation_coefficient, cohens_kappa, f1_score, confusion_matrix, fbeta_score, jaccard_score, log_loss, hamming_loss, top_k_accuracy.
* **Clustering**:	adjusted_rand_index, normalized_mutual_info_score, silhouette_score, calinski_harabasz_index, homogeneity_score, completeness_score, davies_bouldin_index, mutual_information, v_measure_score, rand_score, adjusted_mutual_info_score, fowlkes_mallows_score.
* **Segmentation**:	dice_score, iou_score, sensitivity, specificity, precision, hausdorff_distance
* **Image-to-Image Translation**:	ssim, psnr
---
**Documentation**

Full documentation available at:
