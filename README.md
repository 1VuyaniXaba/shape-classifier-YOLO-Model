# shape-classifier-YOLO-Model
YOLO-based shape detection model trained via Roboflow RF-DETR (Medium) on 192 images of circles, squares, and triangles. Achieves 99.2% mAP@50, 97.4% precision, 97.2% recall. Dataset split 80/10/10 with auto-orient and stretch-resize to 512x512. Includes inference script for real-time webcam detection.
# Shape Classifier (YOLO)

A YOLO-based shape classification model trained using [Roboflow](https://roboflow.com) for dataset labeling/augmentation and trained on Google Colab.

## Model

- **File:** `models/best.pt`
- **Framework:** Ultralytics YOLO
- **Task:** Shape classification/detection
- **Classes:** `circle`, `square`, `triangle`

> **Note:** This model was trained on shapes with a **fixed color per class**:
> - 🔵 Triangle — Blue
> - 🟡 Circle — Yellow
> - 🔴 Square — Red
>
> The model learned shape and color together as a combined feature, so it has **not** been trained to recognize these shapes in other colors (e.g. a red triangle or a blue square may not be detected reliably). For best results, use shapes matching the original color scheme.

## Results

<img width="472" height="167" alt="image" src="https://github.com/user-attachments/assets/a3b9166c-1b10-4ef0-8798-31f549472548" />

