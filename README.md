# DeepDrRVO
Code for deep learning on color fundus photographs for early recognition and differential diagnosis of retinal vascular occlusion.  

>**Notification**
>+ Here we provide the interface of the DeepDrRVO framework and its submodules (DeepDrVAN, DeepDrVBC, DeepDrABC) for early recognition and differential diagnosis of retinal vascular occlusion.
>+ Code for the training of the individual modules and the Few-Sample Generator (FSG) is available from the corresponding author upon reasonable request. 
 
## Requirements
`pip install -r requirements.txt`
## Data  
WMUEH is available from the corresponding author upon reasonable request.  
[RFMiD](https://www.kaggle.com/datasets/andrewmvd/retinal-disease-classification) Retinal fundus multi-disease image dataset  
[ODIR](https://github.com/nkicsl/OIA-ODIR) Ocular Disease Intelligent Recognition  
[JSIEC](https://www.kaggle.com/datasets/linchundan/fundusimage1000)  
## Module Zoo  
We provide the `.pth` file of the optimal model for each module at [Baidu pan]().  
## Inference  
Through the `main.py`, you can start using the modules of DeepDrRVO to realize early recognition and differential diagnosis of retinal vascular occlusion.  
+ Setting work mode at `--module`  
+ Setting CFPs path at `--image_root_dir`  
+ Setting batch size at `--batch_size`  
+ Run `main.py`  

When the inference is completed, the results will be stored in `./results.csv`
## Acknowledgments
+ The implementation of baseline SwinTransformer was based on [timm](https://github.com/rwightman/pytorch-image-models#introduction).  
+ One of the data augmentation methods used comes from [ietk](https://github.com/adgaudio/ietk-ret).
