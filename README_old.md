# EnsDTI: Mixture-of-Experts Approach for Enhanced Drug-Target Interaction Prediction and Confidence Assessment


This repository contains code for running multiple expert models for drug–target interaction (DTI) prediction, as well as their ensemble via **Mixture-of-Experts (MoE)** and **Inductive Conformal Prediction (ICP)**.



## 📦 0. Project Structure

```text
.
├── main.py # 1. Run expert models (DeepConvDTI, DeepPurpose, etc.)
├── moe_predict.py # 2. Run Mixture-of-Experts ensemble
├── icp_predict.py # 3. Run Inductive Conformal Prediction

├── data/ # Data loading, splitting, and preprocessing
│   ├── dataset # CustomCPIDataset and related
├── models/ # Model architectures
├── engine/ # Trainer, tester
├── tmp/ # Temporary storage for intermediate results
└── output/ # Final prediction results
```


## ✅ 1. Running Individual Expert Models

Each model (e.g., `dcdti`, `dpdta`, `mdprd`, etc.) can be trained or evaluated via `main.py`.

### ⚙️ Example

```bash
python main.py \
  --model dcdti \
  --task classification \
  --exp_mode train_and_test \
  --train_data davis \
  --test_data davis \
  --custom default \
  --batch 128 \
  --epoch 100
```

### 🔧 Common Arguments

| Argument       | Description                                                       |
| -------------- | ----------------------------------------------------------------- |
| `--model`      | Model name (e.g., dcdti, dpdta, mdprd, etc.)                      |
| `--exp_mode`   | One of: `train_and_test`, `pure_train`, `pure_test`, `5_fold_val` |
| `--train_data` | Dataset name for training (e.g., `davis`)                         |
| `--test_data`  | Dataset name for testing (for `train_and_test`)                   |
| `--batch`      | Batch size                                                        |
| `--epoch`      | Number of epochs                                                  |
| `--custom`     | Custom tag for result folders                                     |


All results will be stored under:
```
dataset/{data}/{model}/{exp_mode}/{custom}/
```


## 🤝 2. Mixture-of-Experts (MoE)

This approach combines predictions from multiple expert models using a shallow MLP ensemble.


### 🧩 Requirements

Ensure expert model outputs are saved as:

```bash
tmp/{dataset}/{model}_result.csv
```

and the train/test sample indices are saved as:

```bash
tmp/{dataset}/train_index.csv
tmp/{dataset}/test_index.csv
```

### ⚙️ Run MoE

```bash
python moe_predict.py --dataset davis
```

This will:

- Load prediction outputs of all expert models
- Train a shallow MLP via GridSearchCV using training predictions
- Evaluate the ensemble on test set
- Save final predictions to:
  ```bash
  output/{dataset}/result.csv
  ```

## 🔍 3. Inductive Conformal Prediction (ICP)

This module evaluates the confidence of predictions via ICP.

### 🧩 Input Format

Use prediction output in the format:

```bash
results/new_ensdti_internal_validation_on_{dataset}.csv
```

with structure:
  | drug | target | dcdti | dpdta | ... | label |
  | ---- | ------ | ----- | ----- | --- | ----- |


### ⚙️ Run ICP

```bash
python icp_predict.py --d davis
```

This will:
- Sample calibration set from full data
- Estimate confidence (p-value) for each prediction
- Evaluate prediction quality under varying confidence thresholds

### 📁 Output

Results saved to:

```
results/ICP_on_{dataset}.csv
```

## ✨ Citation
If you find this work useful, please cite our paper (coming soon).