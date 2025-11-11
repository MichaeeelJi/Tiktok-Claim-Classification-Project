# TikTok-Claims-Project

## Project Title
Classifying TikTok videos' claim status with Support Vector Machines

## Project Overview
The goal of this project is to develop a predictive model that determines whether a TikTok video contains a claim or offers an opinion. In this iteration, the final model is a Support Vector Machine (SVM). We train an SVM within a scikit-learn pipeline that scales features and tunes the regularization parameter via cross-validation. The model achieves strong, balanced precision and recall on a held-out test set. Consistent with exploratory analysis, user engagement signals such as a video's views, likes, shares, downloads, and comments contribute most to accurate classification, as assessed via permutation importance.

## Business Understanding 
TikTok users can submit reports identifying videos and comments that contain user claims. These reports identify content that needs review by moderators but can accumulate quickly. A high-performing classification model enables triage by prioritizing videos likely to contain claims, supporting efficient moderation and reducing backlog.

## Data Understanding
The dataset is a synthetic sample provided by the TikTok team for the Google Advanced Data Analytics Professional Certificate. A copy is included in this repository. It contains 19,382 rows and 12 columns, with each row representing a video and columns covering engagement metrics and metadata. We removed 298 rows with missing values (a small fraction of the data), found no duplicates, and observed roughly balanced classes, so no resampling was required. Non-informative identifiers (e.g., video ID, number) were dropped. Categorical variables (e.g., `claim_status`, `author_ban_status`, `verified_status`) were encoded numerically. We also engineered `text_length` from `video_transcription_text` to capture basic signal from the transcript as a numeric feature.

## Modeling and Evaluation
- Pipeline: `StandardScaler` + linear SVM (`LinearSVC`) for efficient training and interpretability on numeric features.
- Tuning: Hyperparameter `C` selected via cross-validation on the training set; class weight evaluated to address any minor imbalance.
- Metrics: Accuracy, precision, recall, and F1 evaluated on a hold-out test split; confusion matrix inspected to diagnose error types.
- Insights: Permutation importance indicates engagement-related features (views, likes, shares, downloads, comments) are most predictive for claim detection.

This approach provides a good balance of performance and interpretability. Where deeper interpretability is needed, coefficients from the linear SVM and permutation importance complement each other.

## Conclusion 
An SVM-based classifier effectively distinguishes videos by claim status, with engagement features driving most of the predictive power. Initial data inspection and exploratory analysis align with these findings. Additional statistical and regression analyses are documented in the included notebooks. While `video_transcription_text` is not directly modeled as free text here, future work could apply natural language processing (e.g., CountVectorizer or TF–IDF) to incorporate richer linguistic signal and potentially improve performance.

## Confusion Matrix
The following image shows an example confusion matrix from the model evaluation.

