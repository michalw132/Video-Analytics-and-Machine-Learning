# CSC3067: Video Analytics and Machine Learning Project

A computer vision project for detecting pedestrians in images using classical machine learning techniques. The system extracts Histogram of Oriented Gradients (HOG) features, applies preprocessing and dimensionality reduction, trains an SVM classifier, and performs pedestrian detection using a sliding-window pipeline with non-maxima suppression.

## Overview

This project investigates different feature descriptors, classifiers, preprocessing methods, and evaluation strategies for pedestrian detection. The final approach uses:

- **HOG feature extraction** for robust shape and gradient-based pedestrian representation
- **Z-score standardisation** to reduce feature-scale dominance
- **L2 normalisation** to reduce illumination-related feature magnitude differences
- **PCA dimensionality reduction** to reduce computational cost
- **Support Vector Machine (SVM)** classification
- **Sliding-window detection** across multiple image scales
- **Non-maxima suppression (NMS)** to remove duplicate overlapping detections


## Key Results

The project compared multiple feature extraction and classification approaches.

| Approach | Mean Accuracy | Notes |
|---|---:|---|
| SVM with raw full-image features | 79.1% | Slowest and least effective feature approach |
| SVM with HOG features | 87.0% | Faster and more accurate than raw pixels |
| SVM with PCA and HOG-based pipeline | 91.4% | Large reduction in training/testing time |
| Final tuned SVM pipeline | ~91% | Best overall balance of accuracy and detection performance |

The final tuned classifier used the following SVM-related parameters:

```text
kernel      = gaussian
lambda      = 1e-20
C           = 40
sigmakernel = 5
k           = 10
ndim        = 70
threshold   = 0.1
```

## Training and Evaluation

The model was evaluated using cross-validation rather than a single 50/50 split. Cross-validation was chosen because the dataset size was moderate, with approximately equal numbers of positive and negative samples:

```text
Positive images: 3633
Negative images: 3632
```

Cross-validation gave a more reliable estimate of performance than a single split, where results could depend heavily on the random partition.

The evaluation script records:

- Training time
- Testing time
- True positives
- False positives
- True negatives
- False negatives
- Accuracy
- Error rate
- Precision
- Specificity
- Sensitivity
- F-measure
- False alarm rate


## Project Structure

The report describes the following implementation structure:

```text
.
├── SVMTraining
├── SVMTesting
├── SVMClassifierFunction
├── SVMClassifierScript
├── NN / kNN classifier code
├── feature extraction utilities
├── preprocessing utilities
├── sliding window detector
├── non-maxima suppression
└── evaluation scripts
```

## Contributors

- Aaron McGee
- John Scott
- Michal Wisniewski