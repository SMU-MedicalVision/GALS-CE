# GALS-CE: AI-based LMs Screening Model with Contrast Agent Knowledge

This repository contains the official implementation of our paper:

**“Generative AI enables origin identification of liver metastases using non-contrast CT with contrast agents knowledge.”**

<img src="https://github.com/SMU-MedicalVision/GALS-CE/blob/main/sample_png/Schematic%20illustration.png" width="400px">

## 1. Setup Environment

We recommend creating a Conda environment:

```bash
conda create -n GALS-CE_env python=3.8
conda activate GALS-CE_env
```

Download and extract the repository, and then enter the project directory:

```bash
cd <root>/GALS-CE
```

Here, `<root>` denotes the directory where the repository is stored.

Install PyTorch and the required dependencies:

```bash
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

> **Environment note:** The experiments reported in the manuscript were conducted using PyTorch 2.0.0 on NVIDIA RTX 2080Ti GPUs. During preparation of this repository, the inference pipeline was additionally tested with PyTorch 2.4.1 for compatibility with newer environments. The default requirements reproduce the original experimental environment.

## 2. Quick Start: End-to-End Demo

### 2.1 Download the Dataset
A ready-to-use demo dataset containing 18 sample cases is available for quick verification:

[Download the GALS-CE Demo Dataset](https://www.kaggle.com/datasets/zhengkaiyi/gals-ce-demo-dataset)

The demo dataset contains both the original NIfTI images and the preprocessed data, allowing users to verify the complete preprocessing, multiphase CECT synthesis, and LMs origin-identification pipeline.

### 2.2 Download and Extract the Code

Download and extract the GALS-CE repository, and then enter the project directory:

```bash
cd <root>/GALS-CE
```
### 2.3 Run the End-to-End Demo

After downloading and extracting the demo dataset, run:

```bash
bash ./run_GALS_CE_demo.sh
```

On a single NVIDIA RTX 2080Ti GPU, the script completes the entire GALS-CE quick-test workflow, including synthesis training and inference followed by identification training and inference, in approximately five minutes.

The generated multiphase CECT images are saved in: `./RAW_DATA/Inference_GALS-CE_syn/`

The identification results are saved in:`./Processed_DATA/GALS-CE_cla_pred_Inference.xlsx`

The `--quick_test` option is used for rapid pipeline verification. Remove `--quick_test` from both commands in `run_GALS_CE_demo.sh` to perform standard training.


---

# Train GALS-CE on Your Own Dataset

The following sections describe how to prepare your own dataset, train GALS-CE, and perform inference using your trained models.

## 3. Prepare the Dataset

Organize the NIfTI images according to the following structure:

```text
./RAW_DATA
├── Train
│   ├── ID_0001
│   │   ├── NC.nii.gz
│   │   ├── AP.nii.gz
│   │   ├── PVP.nii.gz
│   │   ├── DP.nii.gz
│   │   ├── Body_mask.nii.gz
│   │   ├── Tumor_mask.nii.gz
│   │   └── Liver_mask.nii.gz
│   ├── ID_0002
│   ├── ...
│   └── ID_0008
│
├── Val
│   ├── ID_0009
│   │   ├── NC.nii.gz
│   │   ├── AP.nii.gz
│   │   ├── PVP.nii.gz
│   │   ├── DP.nii.gz
│   │   ├── Body_mask.nii.gz
│   │   ├── Tumor_mask.nii.gz
│   │   └── Liver_mask.nii.gz
│   ├── ...
│   └── ID_0016
│
├── Inference
│   ├── ID_0017
│   │   ├── NC.nii.gz
│   │   ├── Body_mask.nii.gz
│   │   ├── Tumor_mask.nii.gz
│   │   └── Liver_mask.nii.gz
│   └── ID_0018
│
├── metadata.xlsx
└── metadata_Inference.xlsx
```

### 3.1 Training and Validation Metadata

The class labels of our GALS-CE are defined as follows:

```python
{
    "ICLMs": 1,
    "RCLMs": 2,
    "BCLMs": 3,
    "ECLMs": 4,
    "PCLMs": 5,
    "GCLMs": 6,
    "HCC": 7,
    "ICC": 8
}
```

The file `./RAW_DATA/metadata.xlsx` should contain:

| ID | age | sex | label |
|---|---:|---|---:|
| `ID_0001` | `<age>` | `male/female` | 1 |
| `ID_0002` | `<age>` | `male/female` | 2 |

### 3.2 Inference Metadata

The file `./RAW_DATA/metadata_Inference.xlsx` should contain:

| ID | age | sex |
|---|---:|---|
| `ID_0017` | `<age>` | `male/female` |
| `ID_0018` | `<age>` | `male/female` |

### 3.3 Data Preprocessing

Before training, preprocess the images using grayscale normalization:

```bash
python ./main/data/DATA_prepare_cla.py
```
After preprocessing, the processed data will be saved in `./Processed_DATA`

## 4. Training

### 4.1 Quick Test

The quick-test mode can be used to verify whether the training pipeline is configured correctly.

#### Stage I: Synthesis Quick Test

```bash
python ./main/train_GALS-CE_gen.py --gpu 0 --quick_test
```

#### Stage II: Identification Quick Test

```bash
python ./main/train_GALS-CE_cla.py --gpu 0 --quick_test
```

After training, inference is performed automatically. To run quick-test inference separately:

```bash
python ./main/train_GALS-CE_gen.py \
    --gpu 0 \
    --quick_test \
    --inference_only \
    --inf_dataset <INFERENCE_DATA_DIR> \
    --override \
    --train_model_path_AP <AP_CHECKPOINT> \
    --train_model_path_PVP <PVP_CHECKPOINT> \
    --train_model_path_DP <DP_CHECKPOINT>
```

```bash
python ./main/train_GALS-CE_cla.py \
    --gpu 0 \
    --quick_test \
    --inference_only \
    --inf_dataset <SYNTHESIS_RESULT_DIR> \
    --train_model_pth <CLASSIFICATION_CHECKPOINT>
```

Replace the placeholders with the corresponding data and checkpoint paths.

### 4.2 Comprehensive Training

#### Stage I: Train the CECT Synthesis Model

```bash
python ./main/train_GALS-CE_gen.py --gpu 0
```

#### Stage II: Train the LMs Identification Model

```bash
python ./main/train_GALS-CE_cla.py --gpu 0
```

The synthesized multiphase CECT results generated in Stage I should be provided as input for Stage II.

### 4.3 Visualize the Training Process

TensorBoard can be used to visualize training losses and generated samples:

```bash
tensorboard --logdir ./main/trained_models/
```

## 5. Inference

After training, inference is performed automatically. To perform inference separately, run the following commands.

### 5.1 Multiphase CECT Synthesis

```bash
python ./main/train_GALS-CE_gen.py \
    --gpu 0 \
    --inference_only \
    --inf_dataset <INFERENCE_DATA_DIR> \
    --override \
    --train_model_path_AP <AP_CHECKPOINT> \
    --train_model_path_PVP <PVP_CHECKPOINT> \
    --train_model_path_DP <DP_CHECKPOINT>
```

### 5.2 LMs Origin Identification

```bash
python ./main/train_GALS-CE_cla.py \
    --gpu 0 \
    --inference_only \
    --inf_dataset <SYNTHESIS_RESULT_DIR> \
    --train_model_pth <CLASSIFICATION_CHECKPOINT>
```

Here:

- `<INFERENCE_DATA_DIR>` is the directory containing the preprocessed NCCT inference data.
- `<SYNTHESIS_RESULT_DIR>` is the directory containing the synthetic AP, PVP, and DP images generated in Stage I.
- `<AP_CHECKPOINT>`, `<PVP_CHECKPOINT>`, and `<DP_CHECKPOINT>` are the checkpoints for the three synthesis models.
- `<CLASSIFICATION_CHECKPOINT>` is the checkpoint for the identification model.

## 6. Troubleshooting

Common installation and execution problems are documented in:

```text
Error_troubleshooting.txt
```