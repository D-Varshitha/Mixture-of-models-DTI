# Mixture-of-Experts (MoE) Drug-Target Interaction (DTI) Framework

Welcome to the MoE DTI Framework! This repository provides a state-of-the-art Sparse Mixture-of-Experts pipeline designed to predict Drug-Target Interactions (DTI).

By using a "gating network" powered by chemical (ChemBERTa) and protein (ESM2) language models, this framework dynamically routes every unique drug-protein pair to the best subset of specialized "Expert" models (like DeepConvDTI, DeepDTA, DeepPurpose, etc.).

---

## 1. Installation

Before running the code, you need to set up your environment and install the required dependencies. It is highly recommended to use a virtual environment (like `conda` or `venv`).

**Install Requirements:**
```bash
pip install -r requirements.txt
```
*Note: Make sure you have PyTorch installed with CUDA support if you intend to run this on a GPU (highly recommended).*

---

## 2. Data Preparation

Your datasets should be placed inside the `dataset/` directory. Each dataset must have its own folder containing a `data.csv` file and its precomputed expert feature files.

**Expected Folder Structure:**
```text
DTI-Moe-final/
│
├── dataset/
│   └── davis/                 <-- Name of your dataset
│       ├── data.csv           <-- Contains columns: lig, pro, smi, seq, lab (or affinity)
│       ├── fps_1024.npy       <-- Precomputed 1024-bit Morgan Fingerprints
│       ├── fps_2048.npy       <-- Precomputed 2048-bit Morgan Fingerprints
│       ├── dp_mpnn.npy        <-- MPNN features for DeepPurpose
│       ├── mdprd_pro.pth      <-- Protein features for MDeePred
│       └── dp_pro.npy         <-- Protein features for DeepPurpose
│
├── models/                    <-- Contains Expert implementations
├── engine/                    <-- Contains Training loops and metrics
└── main_MoE.py                <-- Main execution script
```

---

## 3. Running the Code

The main entry point for running the framework is `main_MoE.py`. The script is highly configurable via command-line arguments.

### Debug Mode (Recommended for your first run)
If you just want to verify that the environment is set up correctly and the code doesn't crash, run it in **debug mode**. 

Debug mode automatically:
- Only loads 100 samples from your dataset.
- Restricts training to exactly 1 cross-validation fold.
- Forces the batch size to 16.
- Stops after 3 epochs.

**Command:**
```bash
python main_MoE.py --mode debug --data davis --task classification
```

### Full Training Mode
Once you've verified everything works in debug mode, you can launch a full training experiment. A full run performs rigorous **5-fold cross-validation**, logging results at every step and tracking the best performing model globally.

**Example Command (Classification):**
```bash
python main_MoE.py --mode full --data davis --task classification --batch 64 --top_k 2 --epoch 100
```

**Example Command (Regression):**
```bash
python main_MoE.py --mode full --data davis --task regression --batch 64 --top_k 3 --epoch 100
```

---

## 4. Important Command-Line Arguments

Here is a breakdown of the most useful arguments you can tweak:

### Core Settings
* `--mode`: Set to `full` for real training or `debug` for rapid testing (Default: `full`).
* `--data`: The name of the dataset folder inside `dataset/` (e.g., `davis`, `human`, `kiba`).
* `--task`: The type of machine learning task. Use `classification` or `regression` (Default: `classification`).

### Training Hyperparameters
* `--epoch`: Maximum number of epochs to train per fold (Default: `100`).
* `--batch`: Batch size. *Tip: If you are using a 16GB GPU (like Kaggle T4), a batch size of `64` is recommended to prevent Out-Of-Memory errors* (Default: `128`).
* `--lr`: Learning rate for the optimizer (Default: `1e-4`).
* `--seed`: The global random seed for ensuring reproducible data splits and initialization (Default: `42`).

### Mixture-of-Experts (MoE) Settings
* `--top_k`: The number of experts each sample gets routed to. Options are `2`, `3`, `4`, `5`, `6` (Default: `2`).
* `--lambda-aux`: The weight of the auxiliary load-balancing loss, which ensures the gating network doesn't become biased and ignore certain experts (Default: `0.1`).

### Pretrained Routing Models
* `--chembert-model-name`: HuggingFace model used for chemical SMILES embeddings (Default: `DeepChem/ChemBERTa-77M-MTR`).
* `--esm-model-name`: HuggingFace model used for Protein sequence embeddings (Default: `facebook/esm2_t30_150M_UR50D`).

---

## 5. Viewing Your Results

After your experiment finishes, the framework will automatically organize your outputs:

1. **Results & Metrics:** 
   * A summary of every fold and the final `mean ± std` will be saved to:
     `results/<dataset_name>/<task>/fold_summary.csv`
   * A master history file (in JSON format) logging all your experiments is saved as:
     `experiment_results.json`
2. **Saved Models:** 
   * Only the **single best epoch of the best fold** is saved to conserve disk space. It will be located in the `saved_models/` folder (e.g., `saved_models/best_model_human_classification.pt`).
3. **Training Curves:** 
   * Per-epoch history (loss, accuracy, etc.) is saved as a CSV for each fold inside your `results/` folder, making it easy to plot training curves later.

Happy Training!