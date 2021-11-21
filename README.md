# IQAAD
Official implementation for the paper "***Unsupervised Anomaly Detection Based on Full-Reference Image Quality Assessment***"

## Descriptions
This project is a [Pytorch](https://pytorch.org/) implementation of IQAAD. This paper improves two classical full-reference image quality assessment(FR-IQA), the gradient magnitude similarity deviation(GMSD) and the feature similarity induced metric(FSIM), which can better haddle the intrinsic difficulties of anomaly detection tasks.

## How To Use This Code
You will need:
  - Python 3.7
  - [Pytorch](https://pytorch.org/), version 1.7.0
  - torchvision, version 0.8.0
  - numpy, opencv-python

The default parameters for CelebA-HQ faces at 128x128, 256x256, 512x512 and 1024x1024 resolutions are provided in the file 'run_128.sh', 'run_256.sh', 'run_512.sh' and 'run_1024.sh', respectively. 

 To train 128x128 celebA-HQ, put your dataset into a folder and run:
```
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --dataroot='/dataset/celebAHQ/celeba-128/' --noise_dim=256 --batch_size=128 --test_batch_size=64 --nEpochs=500 --sample-step=1000 --save_step=10 --channels='32, 64, 128, 256, 512' --trainsize=29000 --input_height=128 --output_height=128 --m_plus=140 --weight_neg=0.5 --weight_rec=0.2 --weight_kl=1. --weight_logit=20 --num_vae=0 --num_gan=10 > main.log 2>&1 &
```

 To train 256x256 celebA-HQ, put your dataset into a folder and run:
```
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --dataroot='/dataset/celebAHQ/celeba-256/' --noise_dim=512 --batch_size=32 --test_batch_size=16 --nEpochs=500 --sample-step=1000 --save_step=10 --channels='32, 64, 128, 256, 512, 512' --trainsize=29000 --input_height=256 --output_height=256 --m_plus=160 --weight_neg=0.5 --weight_rec=0.05 --weight_kl=1. --weight_logit=1000 --num_vae=0 --num_gan=10 > main.log 2>&1 &
```

 To train 512x512 celebA-HQ, put your dataset into a folder and run:
```
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --dataroot='/dataset/celebAHQ/celeba-512/' --noise_dim=512 --batch_size=16 --test_batch_size=16 --nEpochs=500 --sample-step=1000 --save_step=10 --channels='16, 32, 64, 128, 256, 512, 512' --trainsize=29000 --input_height=512 --output_height=512 --m_plus=90 --weight_neg=0.25 --weight_rec=0.01 --weight_kl=1. --weight_logit=100 --num_vae=0 --num_gan=10 > main.log 2>&1 &
```

 To train 1024x1024 celebA-HQ, put your dataset into a folder and run:
```
CUDA_VISIBLE_DEVICES=0,1,2 python3 main.py --dataroot='/dataset/celebAHQ/celeba-1024/' --noise_dim=512 --batch_size=8 --test_batch_size=8 --nEpochs=500 --sample-step=1000 --save_step=10 --channels='16, 32, 64, 128, 256, 512, 512, 512' --trainsize=29000 --input_height=1024 --output_height=1024 --m_plus=130 --weight_neg=0.5 --weight_rec=0.0025 --weight_kl=1. --weight_logit=50 --num_vae=10 --num_gan=10 > main.log 2>&1 &
```

## Results
[![logo](https://github.com/openAIRoom/IQAAD/blob/main/Samples/mvtec.jpg)](https://github.com/openAIRoom/IQAAD/blob/main/Samples/mvtec.jpg) 
[![logo](https://github.com/openAIRoom/IQAAD/blob/main/Samples/mnist.png)](https://github.com/openAIRoom/IQAAD/blob/main/Samples/mnist.png) 

## Citation
If you find our code helpful in your research or work please cite our paper.
```
@inproceedings{AEGI,
  title={Unsupervised Anomaly Detection Based on Full-Reference Image Quality Assessment},
  author={*},
  booktitle={*},
  pages={*},    
  year={2022}
}
```

**The released codes are only allowed for non-commercial use.**
