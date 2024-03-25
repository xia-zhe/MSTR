# MSTR
## Multi-Modal Spiking Tensor Regression Network for Audio-Visual Zero-Shot Learning
Zhe Yang, Wenrui Li, Jinxiu Hou, Guanghui Cheng. 
The code is based on [AVCA](https://github.com/ExplainableML/AVCA-GZSL) and tested on Ubuntu 20.04 with torch 2.0.1.

### Installing tensorly
Simply run, in your terminal:
```
pip install -U tensorly
```

### Inportant
The version of [spikingjelly](https://spikingjelly.readthedocs.io/zh_CN/latest/index.html) we used is 0.0.0.0.14.

Installing different versions can cause performance differences.
### Downloading features
The features and dataset structure could download and placed the same as [AVCA](https://github.com/ExplainableML/AVCA-GZSL).


## Evaluation
### Dowloading pre-trained models
[Here](https://drive.google.com/drive/folders/1A691fo9_DnKoTZJku4xTBfpgqw1Nn6Re?usp=sharing), you can download our trained MSTR models and baselines which are located in `pretrain_model.zip`
Put the content of `pretrain_model.zip` in the `runs/` folder.
### Test on three benchmark datasets
Here is an example for evaluating MSTR on Vggsound-GZSL using SeLaVi features.
``` 
python get_evaluation.py --load_path_stage_A runs/attention_ucf_vggsound_main --load_path_stage_B runs/attention_vggsound_all_main  --dataset_name VGGSound --MSTR 
```

## Results 
### GZSL performance on VGGSound-GZSL, UCF-GZSL, ActivityNet-GZSL

| Method             | VGGSound-GZSL          | UCF-GZSL        | ActivityNet-GZSL |
|--------------------|------------------------|-----------------|------------------|
| APN                |    5.11                |    20.61        |   7.27           |
| VAEGAN             |    1.77                |    11.37        |   2.87           |
| SJE                |    2.15                |    26.50        |   5.57           |
| DEVISE             |    2.08                |    23.56        |   4.91           |
| CJME               |    6.17                |    12.48        |   5.12           |
| AVGZSLNET          |    5.83                |    18.05        |   6.44           |
| AVCA               |    6.31                |    27.15        |   12.13          |
| TCaF               |    7.33                |    31.72        |   10.71          |
| Hyper_alignment    |    7.27                |    27.19        |   12.46          |
| MSTR               |  **7.83**              |  **32.43**      | **13.21**        |


## Project Structure
```audioset_vggish_tensorflow_to_pytorch``` - Contains the code which is used to obtain the audio features using VGGish.

```c3d``` - Folder contains the code for the C3D network.

```selavi_feature_extraction``` - Contains the code used to extract the SeLaVi features.

```src``` - Contains the code used throughout the project for dataloaders/models/training/testing.

```cls_feature_extraction``` - Contains the code used to extract the C3D/VGGish features from all 3 datasets.

```avgzsl_benchmark_datasets``` - Contains the class splits and the video splits for each dataset for both features from SeLaVi and features from C3D/VGGish.

```splitting_scripts``` - Contains files from spltting our dataset into the required structure. 

```w2v_features``` - Contains the w2v embeddings for each dataset.
```run_scripts``` - Contains the scripts for training/evaluation for all models for each dataset.

## Acknowledgement
We appreciate the code provided by [AVCA](https://github.com/ExplainableML/AVCA-GZSL), which is very helpful to our research.
