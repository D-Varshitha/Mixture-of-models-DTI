# DTI-MoE Implementation Plan

This document outlines the implementation plan for the Sparse Mixture-of-Experts (MoE) model for Drug-Target Interaction prediction, strictly adhering to the provided constraints.

> [!CAUTION]
> **User Review Required**: Please review the **Open Questions** section regarding the exact unified format for DataLoader outputs and the missing `PerceiverCPI` implementation before I proceed with the coding.

---

## 1. Identified Code Inconsistencies & Required Refactoring

Based on the analysis of the existing codebase (`main.py`, `config.py`, `engine/trainer.py`, `data/dataset.py`, `models/`), several updates are needed to support the MoE architecture cleanly:

*   **Missing Models**: `perceivercpi` is commented out in `models/__init__.py` and the file is missing from the directory. We will need to create `models/perceivercpi.py`.
*   **Hardcoded Forward Logic**: `engine/trainer.py` currently explicitly dispatches logic based on `model_name == 'dp'`, `model_name == 'cpi'`, etc. This will be replaced or extended in a new `engine/trainer_MoE.py` where the MoE model encapsulates how to pass specific features to experts.
*   **Monolithic Data Processing**: `data/dataset.py` contains monolithic preprocessing. The new data loaders (`data/davis_loader.py`, etc.) will decouple dataset-specific parsing from feature extraction to yield the requested unified format.
*   **Argument Parsing**: `config.py` needs to support the new `--mode debug/full` and `--subset_size` flags, as well as the MoE loss hyperparameters (e.g., `lambda`).

---

## 2. Proposed Architecture Breakdown

### Shared Gating Embedding (Strictly For Gating)
*   **Drug Encoder**: A Lightweight Transformer Encoder to process drug sequences at the token level using padding masks (no truncation).
*   **Protein Encoder with Chunking**: 
    *   Proteins that exceed maximum lengths will be split into non-overlapping chunks.
    *   Each chunk is passed through the Transformer Encoder independently.
    *   Chunk representations are coalesced into a single vector using an **Attention Layer** (e.g., soft alignment attention), strictly avoiding max/mean pooling.
*   **Gating Network**: 
    *   Outputs of the drug and protein encoders are concatenated (or cross-attended).
    *   Passed to a multi-layer perceptron (MLP) mapping to `Softmax(6 dimensions)`.

### Expert Implementation
*   **The 6 Experts**: DeepDTA, DeepConv-DTI, DeepPurpose, MDeePred, GFDTI, PerceiverCPI.
*   The exact internal implementations of existing models (`models/dp.py`, `models/dpdta.py`, etc.) will **not** be modified. 
*   A new wrapper `models/model_MoE.py` will encapsulate these architectures as sub-modules. 

### Execution Pipeline & Routing
The pipeline will STRICTLY execute as follows:
**Input (drug, protein) → Shared embedding → Gate → Top-2 → Run selected experts → Combine → Loss → Backprop**

Key Properties of this Pipeline:
1.  **Shared Embedding ONLY for gating:** The shared embedding generated from the inputs is consumed exclusively by the gating network to produce the routing probabilities. The experts do NOT use the shared embedding; instead, they use their own respective input encoders.
2.  **Only selected top 2 experts predict:** Softmax is applied by the gate to evaluate all experts, but ONLY the top 2 experts are executed during the forward pass computations. The other 4 experts are completely bypassed for that sample to skip unnecessary computations. 
3.  **Weighted Combination:** The final output is formed by combining the predictions of the selected top 2 experts using their specific gate weights.
4.  **Loss & Backprop:** The task loss and auxiliary load balancing loss are summed. Backpropagation seamlessly routes gradients only to the executed experts and the shared gating network.

### Loss Formulation
*   $Total Loss = Task Loss + \lambda \times Aux Loss$
*   **Aux Loss**: The standard Load Balancing Loss (Square of the coefficient of variation of the routing proportions) ensuring experts receive an equal proportion of assignments across batches.

---

## 3. Data Pipelines & Loaders

We will create standalone loaders ensuring unified output:
*   `data/davis_loader.py`
*   `data/kiba_loader.py`
*   `data/human_loader.py`
*   `data/kinome_loader.py`

**Unified Signature**: `{"drug": ..., "protein": ..., "label": ..., "mask": ...}`
Because experts require totally different feature sets (e.g., DeepPurpose needs MPNN features, MDeePred needs 5-channel 2D matrices), the `drug` and `protein` dictionary values will themselves be sub-dictionaries carrying the specific features required by the 6 architectures alongside the token sequences for the gating network.

---

## 4. Training Modes (Debug vs. Full)

We will introduce a new entry script `main_MoE.py` to preserve the original pipeline. It will support:
*   `--mode debug`: Overrides training sequence to use only a random subset of data (`subset_size`, default 100). Model skips expensive dataset initializations entirely to speed up testing loops.
*   `--mode full`: Iterates via K-Fold cross validation (default 5-folds) on the selected dataset over epochs. **Note:** K-Fold cross validation is already heavily supported in your existing codebase (`engine/trainer.py` and `data/dataset.py` via `split_dataset_by_fold`). We will directly re-use this perfectly intact existing cross-validation logic.

---

## 5. File Modifications & Additions

#### [NEW] `IMPLEMENTATION.md`
This document.

#### [NEW] `main_MoE.py`
New entry point handling `--mode` parsing, debug bootstrapping, and initializing `MoE` architectures instead of singles.

#### [NEW] `engine/trainer_MoE.py`
New training loop responsible for tracking `main_loss` and `aux_loss`, reporting both to wandb, and executing the MoE forward pass following the exact pipeline defined above.

#### [NEW] `models/model_MoE.py`
The Sparse MoE logic. Contains:
*   `class ProteinChunkingEncoder(nn.Module)`
*   `class DrugEncoder(nn.Module)`
*   `class SharedGatingNetwork(nn.Module)`
*   `class DTI_Sparse_MoE(nn.Module)`

#### [NEW] `models/perceivercpi.py`
Implementation for the 6th expert.

#### [NEW] `data/davis_loader.py`, `data/kiba_loader.py`, `data/human_loader.py`, `data/kinome_loader.py`
Extracting specific logic from `dataset.py` into standalone modular files.

#### [MODIFY] `models/__init__.py`
Import the new `DTI_Sparse_MoE` and `PerceiverCPI`.

#### [MODIFY] `config.py`
Add arguments for `--mode`, `--subset_size`, `--lambda_aux`.

---

## Open Questions

> [!IMPORTANT]
> Please clarify the following before I start coding:
> 1. **Data Format Unified Output**: Because experts require vastly different matrices/graphs, is it acceptable for the unified `"drug"` and `"protein"` arrays to act as python *dictionaries* mapping to the respective expert inputs (e.g., `batch["drug"]["mpnn"]` and `batch["drug"]["shared_tokens"]`), or must they literally be a single tensor representation?
> 2. **PerceiverCPI**: The model `perceivercpi.py` is absent from the existing files. Should I construct a standard PerceiverIO implementation for this expert, or will you provide it?
> 3. **Shared Embedding Mask**: You mentioned `{"mask": ...}`. Does this refer to the padding mask of the shared transformer embedding, or cross-validation mask?

---

## Verification Plan

### Automated Tests
*   Run `python main_MoE.py --dataset davis --mode debug` to ensure:
    *   Batch routes to exactly 2 experts.
    *   Auxiliary load balancing loss evaluates > 0 initially and decreases over time.
    *   Gradients are *only* propagated to selected experts via `tensor.grad` checking.
    *   Sequence lengths > threshold gracefully chunk via the attention mechanism computationally smoothly.

### Manual Verification
*   Validation against exact GPU memory capacity locally ensuring graph decoupling between unselected experts.
