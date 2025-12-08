# AU-Conditioned MAE + Pose-Normalized FER  
**Course:** MSML640 – Computer Vision  

**Team Members:**  
- **Narendra Rane**   
- **Prakhar P. Tiwari** 

---

## Project Overview
This project explores improving **Facial Expression Recognition (FER)** robustness using two complementary ideas:
1. **Pose Normalization** to reduce sensitivity to head pose variations.
2. **AU‑Conditioned Masked Autoencoder (AU‑MAE)** pretraining using Action Unit supervision.

The system is evaluated through controlled ablation studies on the **RAF‑DB** dataset.

---

## Datasets
- **RAF-DB**: Primary dataset for FER training and evaluation (7 emotion classes).

---

## Ablations Implemented
- **Ablation A**: FER baseline (ResNet‑50).
- **Ablation B**: FER + Pose Normalizer.
- **Ablation C**: AU‑MAE pretrained → FER.

---

## Setup Instructions (Kaggle – Required)

### Step 1: Download RAF‑DB
- Download the **"RAF‑DB"** dataset locally.
- Then rename the downloaded folder as **"RAF-DB"**.


### Step 2: Clone the project
- Create a folder named **"MSML 640"** and open it in VS Code
- Then run the below command in the VS Code terminal
   ```bash
   git clone https://github.com/narendrarane50/MSML640-Final_Project.git
   ```
- Then create the **"datasets"** folder inside the **"MSML640-Final_Project"** folder and put the **"RAF-DB"** folder inside the **"datasets"** folder such that **"datasets/RAF-DB"**.
- Then compress the **"MSML640-Final_Project"** folder.

### Step 3: Create Kaggle Notebook
- Go to Kaggle → Create → Notebook
- On the right panel: Click Upload → Click New Dataset → Upload the compressed **"MSML640-Final_Project.zip"** folder → Dataset title: **"msml640project"** → Click Create

### Step 4: Enable GPU
- Open Notebook Settings
- Set Accelerator → GPU (P100)
- Start the session

### Step 5: Move Project Files into Working Directory
- Then in Notebook cell run
   ```bash
   !mkdir -p /kaggle/working/MSML640-Final_Project
   !cp -r /kaggle/input/msml640project/MSML640-Final_Project/* /kaggle/working/MSML640-Final_Project/
   %cd /kaggle/working/MSML640-Final_Project
   !ls
   ```

---

## Running the Experiments

### AU‑MAE Pretraining
- Run this first to generate **"au_mae_pretrained.pth"** model:
   ```bash
   !python main.py --config config.yaml --mode pretrain_au_mae
   ```

### Ablation A, B, and C (FER Training + Evaluation)
- After AU‑MAE pretraining completes run the below command for Ablation A (FER baseline), B (FER + Pose Normalizer) and C (AU‑MAE fine‑tuned FER) and for saving the evaluation metrics to logs:
   ```bash
   !python main.py --config config.yaml --dataset rafdb
   ```

### Generating Confusion Matrix Images
- After all ablations finish, run:
   ```bash
   !python scripts/plot_confusion_matrices.py
   ```

 ### Outputs
 - Training logs
 - Saved metrics (JSON / NPY)
 - Confusion matrix images
 - AU‑MAE pretrained weights (au_mae_pretrained.pth)
 
