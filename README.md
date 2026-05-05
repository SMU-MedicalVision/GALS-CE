# GALS-CE : AI-based LMs screening model with contrast agent knowledge

This repository contains the code of our paper "Generative AI enables origin identification of liver metastases using non-contrast CT with contrast agents knowledge".

<img src="https://github.com/SMU-MedicalVision/GALS-CE/blob/main/sample_png/Schematic%20illustration.png" width="400px">


# 1. Setup Environment
In order to run our model, we suggest you create a virtual environment
```
conda create -n GALS-CE_env python=3.8
```
and activate it with
```
conda activate GALS-CE_env
```
Subsequently, download and install the required libraries by running:
```
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```
# 2. Prepare the Dataset
To simplify the dataloading for your own dataset, we provide a default dataset that simply requires the path to the folder with your NifTI images inside, i.e.
```
./RAW_DATA  
├──Train
│    ├── ID_001                       
│    │        ├── NC.nii.gz             
│    │        ├── AP.nii.gz        
│    │        ├── PVP.nii.gz       
│    │        ├── DP.nii.gz        
│    │        ├── (Body_mask.nii.gz)  
│    │        ├── (Tumor_mask.nii.gz) 
│    │        └── (Liver_mask.nii.gz) 
│    ├── ID_002
│    ├── ... 
│    └── ID_N 
│
├──Val
│    ├── ID_111                       
│    │        ├── NC.nii.gz             
│    │        ├── AP.nii.gz        
│    │        ├── PVP.nii.gz       
│    │        ├── DP.nii.gz        
│    │        ├── (Body_mask.nii.gz)  
│    │        ├── (Tumor_mask.nii.gz) 
│    │        └── (Liver_mask.nii.gz) 
│    └── ...
│
└──Inference
     ├── ID_211                       
     │        ├── NC.nii.gz                   
     │        └── (Body_mask.nii.gz)  
     └── ...
```
Before training, the data needs to be preprocessed by **'Grayscale Normalization'** by executing the following command.
```
python ./main/data/DATA_prepare_cla.py
```


# 3. Training
- ## Quick Test (optional)
**Stage I**: Synthesis quick test
```
python ./main/train_GALS-CE_gen.py --gpu 0 --quick_test
```
**Stage II**: Identification quick test 
```
python ./main/train_GALS-CE_cla.py --gpu 0 --quick_test
```

**Inference**(optional): quick test. After the training is completed, the inference will be automatically carried out. If you want to perform the inference separately, please run:
```
python ./main/train_GALS-CE_gen.py --gpu 0 --quick_test --inference_only --save_dir ./main/trained_models/GALS-CE_gen/{pred_*_...class_seg_time}/
python ./main/train_GALS-CE_cla.py --gpu 0 --quick_test --inference_only --gen_save_dir ./main/trained_models/GALS-CE_gen/{pred_*_...class_seg_time}/ --save_dir ./main/trained_models/GALS-CE_cla/{bs*_ImageSize*_epoch*_seed*_time}/
```
>{} should be changed to the actual path for saving the synthesis result.  

- ## Comprehensive Training

**Stage I**: First, you need to train the generation model. To do so in a prepared dataset, you can run the following command:
```
python ./main/train_GALS-CE_gen.py --gpu 0
```

**Stage II**: Second, you need to train the classification model by running the following command. 
```
python ./main/train_GALS-CE_cla.py --gpu 0 --gen_save_dir ./main/trained_models/GALS-CE_gen/{pred_*_...class_seg_time}/
```
>Note that you need to provide the path to the synthesis result to successfully run the command.


- ## Visualize the Training Process (optional)
You can use the following command to observe the loss curve of the training process, visualize the sample image, etc.
```
tensorboard --logdir ./main/trained_models/
```


[Supplement] Problem troubleshooting can be found in Error_troubleshooting.txt
# 4. Inference (optional)
After the training is completed, the inference will be automatically carried out. If you want to perform the inference separately, please run:
```
python ./main/train_GALS-CE_gen.py --gpu 0 --inference_only --inf_dataset {}  --override --train_model_path_AP {./main/trained_models/GALS-CE_gen/train/Pre_Swin_ADN_trainD_modal-AP/**.pth} --train_model_path_PVP {./main/trained_models/GALS-CE_gen/train/Pre_Swin_ADN_trainD_modal-PVP/**.pth} --train_model_path_DP {./main/trained_models/GALS-CE_gen/train/Pre_Swin_ADN_trainD_modal-DP/**.pth}
python ./main/train_GALS-CE_cla.py --gpu 0 --inference_only --gen_save_dir ./main/trained_models/GALS-CE_gen/{pred_*_...class_seg_time}/ --save_dir ./main/trained_models/GALS-CE_cla/{bs*_ImageSize*_epoch*_seed*_time}/
```

# Citation

To cite our work, please use
```
(To be updated)
```

