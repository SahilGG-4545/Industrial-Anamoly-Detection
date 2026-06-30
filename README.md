# Industrial Anomaly Detection using PatchCore

An unsupervised industrial anomaly detection system built using **PatchCore** and **Anomalib** on the **MVTec AD Leather** dataset.

The model is trained exclusively on **normal images** and learns the distribution of defect-free samples. During inference, deviations from these learned patterns are identified as anomalies and localized using anomaly heatmaps.

## Features

* Unsupervised anomaly detection using only normal images
* Defect localization using pixel-level anomaly heatmaps
* Feature extraction using a pretrained ResNet backbone
* FAISS-based nearest neighbor search for anomaly scoring
* Supports both image-level and pixel-level evaluation

## Dataset

* **Dataset:** MVTec AD
* **Category:** Leather
* **Defect Types:** Color, Cut, Fold, Glue, Poke

## Tech Stack

* Python
* PyTorch
* Anomalib
* PatchCore
* FAISS
* OpenCV
* Matplotlib

## Model Pipeline

1. Train the model using only normal leather images.
2. Extract feature embeddings using a pretrained ResNet backbone.
3. Build a memory bank of normal feature representations.
4. Compare test image patches against the memory bank using nearest neighbor search.
5. Generate anomaly scores and localization heatmaps.

## Results

| Metric         | Score  |
| -------------- | ------ |
| Image AUROC    | 0.9956 |
| Image F1 Score | 0.9834 |
| Pixel AUROC    | 0.9893 |

## Sample Output

* Original Image
* Anomaly Heatmap
* Heatmap Overlay
* Image-level Anomaly Score
* Predicted Class (Normal/Anomaly)

## Future Improvements

* Evaluate on additional MVTec categories.
* Compare performance with PaDiM and EfficientAD.
* Deploy the model using FastAPI.
* Add support for real-time industrial inspection.

## References

* MVTec AD Dataset
* PatchCore: Towards Total Recall in Industrial Anomaly Detection
* Anomalib Documentation
