# EyeDR: Diabetic Retinopathy Detection using YOLOv8

Welcome to **EyeDR**, a comprehensive repository for **Diabetic Retinopathy (DR) detection** using the **YOLOv8 object detection framework**. This repository contains everything needed to prepare datasets, train models, evaluate performance, and visualize predictions.

---

## Repository Overview
EyeDR is structured to guide you step-by-step, from dataset preparation to inference, making it beginner-friendly and easy to navigate.

### 1️⃣ `DR_dataset/idrid-yolo/`
This folder contains the dataset formatted for YOLOv8 training.

- **images/**
  - `train/` → All images used for training the model.
  - `val/` → Images reserved for validation to evaluate model performance.

- **labels/**
  - `train/` → YOLOv8 label files corresponding to training images. Each label contains class and bounding box coordinates.
  - `val/` → YOLOv8 label files for validation images.
  - `.cache` files → Used internally by YOLOv8 to speed up training.

- **data.yaml** → Central configuration file that specifies dataset paths, class names, and number of classes. **Tip:** Open this to verify paths and class labels before training.

**Step to follow:** Ensure the dataset paths in `data.yaml` match your local or Google Drive paths.

---

### 2️⃣ `DR_results/`
All outputs from training, validation, and inference are stored here. Each subfolder corresponds to a specific training run or evaluation.

- **DR_yolov8m_1024_aug_weighted/**
  - `BoxP_curve.png` → Precision-Recall curve to visualize classification performance.
  - `args.yaml` → Contains all training arguments used during the run.
  - `DR_aug_weights/` → Folder containing trained model weights (may be large).

- **DR_yolov8m_1024_aug_weighted_pred_val/**
  - Predicted images with bounding boxes on validation images. Example: `IDRiD_81.jpg`.
  **Step to follow:** Open these images to verify if lesions are correctly detected.

- **DR_yolov8m_1024_aug_weighted_val/**
  - `BoxF1_curve.png` → F1 score curve for evaluating model accuracy.

- **dr_hyp.yaml** → Hyperparameter configuration used for the training run.

**Tip:** Use this folder to track different experiments, compare performance, and store trained weights safely.

---

### 3️⃣ `DR-YOLOv8/`
Contains results from multiple experiments for comparative analysis.

- **idrid_results*/**
  Each folder corresponds to a different training run with variations in hyperparameters, image size, augmentation, or model type. Contains:
  - Plots (PR curves, F1 curves, confusion matrices)
  - Predicted images
  - Saved weights

**Step to follow:** Review each experiment folder to choose the best-performing model for inference.

---

### 4️⃣ Root Folder
- `DR_YOLOv8_Training.ipynb` → The main notebook for training, evaluating, and performing inference.

**Step to follow:**
1. Open the notebook in **Google Colab** or your local Jupyter environment.
2. Mount your dataset if using Google Drive.
3. Adjust hyperparameters such as batch size, image size, and epochs based on GPU capacity.
4. Execute each cell to reproduce training, evaluation, and prediction workflows.

---

## ⚙️ Notes & Best Practices

- **Large weights:** Files over 100 MB need **Git LFS** if you plan to push them to GitHub.
- **GPU memory:** Adjust batch size and image resolution to prevent memory errors.
- **Experiment tracking:** Keep each run in a separate folder for clarity.
- **Predictions & plots:** Always check predicted images and evaluation plots for quality assurance.

---

## 📚 References
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [IDRiD Dataset](https://idrid.grand-challenge.org/)

---

## 🔖 License
Made by Amogha Sri Kommera.

