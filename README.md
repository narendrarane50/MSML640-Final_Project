# AU-Conditioned MAE + Pose-Normalized FER  
**Course:** MSML640 – Computer Vision  
**Team Members:**  
- **Prakhar P. Tiwari** – AU-MAE pretraining, backbone models, GitHub pipeline  
- **Narendra R. Rane** – Pose normalization, dataset preparation, evaluation, ablation studies  

---

## 🧩 Project Overview
This project develops a **robust Facial Expression Recognition (FER)** system capable of performing well in “in-the-wild” conditions.  
We integrate **Action Unit (AU)-conditioned Masked Autoencoder (MAE)** pretraining with a **Pose Normalization** module to improve resilience to pose variation, occlusions, and data imbalance.

---

## 🎯 Objective
1. Learn **pose-invariant features** via a Spatial Transformer-based Pose Normalizer.  
2. Use **AU-conditioned MAE pretraining** to capture fine-grained facial cues.  
3. Evaluate effectiveness through ablations and cross-dataset generalization.

---

## 🧠 Datasets
| Dataset | Usage | Description |
|----------|--------|-------------|
| **RAF-DB** | Main training & validation | 7-class facial expression dataset |
| **FER-2013** | Benchmark testing | In-the-wild grayscale dataset |
| **AffectNet (partial)** | AU-MAE pretraining | Labeled + unlabeled large-scale face data |
| **Custom Wild Data** | Bonus analysis | Self-collected occluded and angled faces |

---

## 🧪 Ablation Experiments

| Experiment | Description | Accuracy | Balanced Accuracy | Macro F1 |
|-------------|--------------|-----------|-------------------|-----------|
| **Ablation A** | FER baseline (no pose normalization) | 81.45% | 69.20% | 0.717 |
| **Ablation B** | FER + Pose Normalizer | 79.95% | 68.26% | 0.707 |

---

## 📊 Evaluation Metrics
- Accuracy, Balanced Accuracy, Macro-F1  
- Confusion Matrix (per-class insights)  
- Robustness analysis under pose and occlusion variation  

---


## ⚙️ Environment Setup (Kaggle Only)

You can run this project end-to-end on **Kaggle Notebooks** with GPU acceleration.

### Steps

1. **Create a New Kaggle Notebook**
   - Go to https://www.kaggle.com/code
   - Click **New Notebook → GPU (T4/P100)**
   - Turn **Internet = ON** in notebook settings

2. **Clone the Repository**
   ```bash
   !git clone https://github.com/narendrarane50/MSML640-Final_Project.git
   %cd MSML640-Final_Project

3. **Install Dependencies**  
   (Most required libraries are preinstalled on Kaggle; install extras if needed)
   ```bash
   !pip install -r requirements.txt

4. **Attach Dataset**
   - In the right sidebar of the Kaggle Notebook, click **Add Data → Your Datasets → RAF-DB / FER-2013**
   - Ensure your dataset folder name matches what is referenced in `config.yaml`
   - Example configuration:
     ```yaml
     dataset_root: "/kaggle/input/rafdb"
     ```
5. **Run Training / Evaluation**
   - Once the environment and dataset are ready, run the training command below:
     ```bash
     !python main.py --config config.yaml --dataset rafdb
     ```
   - This will automatically:
     - Load the configuration from `config.yaml`
     - Train the FER model (baseline or Pose-Normalized version)
     - Print epoch-wise loss, accuracy, and F1 metrics
     - Save checkpoints and logs in `/kaggle/working/results/`

