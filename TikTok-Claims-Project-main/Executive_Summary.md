# Machine Learning Model Outcomes

Executive summary report for TikTok prepared by the TikTok data team

## Overview
The goal is to develop a machine learning model that classifies TikTok videos as containing a claim or expressing an opinion. Using the provided synthetic dataset (19,382 rows, 12 columns), we engineered features, trained models, and selected a Support Vector Machine (SVM) as the final classifier for its strong, balanced precision and recall and straightforward interpretability via linear coefficients.

## Problem
User reports that flag potential claims can quickly accumulate, overwhelming moderators. Without automated triage, high-priority content can be delayed in review, increasing operational backlog and risk.

## Solution
- Model: Support Vector Machine (LinearSVC) in a scikit‑learn pipeline with `StandardScaler`.
- Tuning: Cross‑validation over `C` and `class_weight`, refitting on recall to prioritize catching true claims.
- Data prep: Dropped 298 rows with missing values; removed identifiers; one‑hot encoded categorical fields; engineered `text_length` from transcripts; no resampling required due to near‑balanced classes.
- Split: 60/20/20 train/validation/test.

## Details
- Performance: The SVM achieved high, balanced precision and recall on validation and test splits. Consistent with EDA, engagement features (views, likes, shares, downloads, comments) were the strongest predictors.
- Interpretability: Linear SVM coefficients and permutation importance highlight engagement as the dominant signal; transcript length adds modest signal.
- Confusion matrix: Few false positives/negatives on the held‑out test set (see figure).

![Confusion Matrix](images/confusion_matrix.png)

## Next Steps
- Add richer NLP features from `video_transcription_text` (e.g., TF‑IDF n‑grams) to capture linguistic signal.
- Consider probability calibration (e.g., `CalibratedClassifierCV`) for threshold tuning aligned to moderation SLAs.
- Conduct fairness checks across author attributes; monitor model drift and recalibrate periodically.
- Explore alternative linear baselines (logistic regression) and kernel SVM/RBF as benchmarks.
- Build lightweight inference pipeline and batch scoring job for priority queues in moderation tooling.

## References
- Modeling workflow and results: `TikTok Project Machine Learning Models.ipynb`
- Data and project notes: `README.md`
