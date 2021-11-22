# IQAAD
Official implementation for the paper "***Unsupervised Anomaly Detection Based on Full-Reference Image Quality Assessment***"

## Descriptions
This project is a [Pytorch](https://pytorch.org/) implementation of IQAAD. This paper introduce the full-reference image quality assessment(FR-IQA) metric into the reconstruction error, taking the input image as reference. Besides, IQAAD improves two classical FR-IQA metrics, the gradient magnitude similarity deviation(GMSD) and the feature similarity induced metric(FSIM), so they can better haddle the intrinsic difficulties of anomaly detection tasks. In addition, in order to maintain high reconstruction quality for normal samples while suppressing that for abnormal samples, IQAAD proposes a memory bank with a new updating stratery.

## How To Use This Code
You will need:
  - Python 3.7
  - [Pytorch](https://pytorch.org/), version 1.7.0
  - torchvision, version 0.8.0
  - numpy, opencv-python

## Results
[![logo](https://github.com/openAIRoom/IQAAD/blob/main/Samples/mvtec.jpg)](https://github.com/openAIRoom/IQAAD/blob/main/Samples/mvtec.jpg) 
[![logo](https://github.com/openAIRoom/IQAAD/blob/main/Samples/mvtec_more.jpg)](https://github.com/openAIRoom/IQAAD/blob/main/Samples/mvtec_more.jpg) 

## Citation
If you find our code helpful in your research or work please cite our paper.
```
@inproceedings{IQAAD,
  title={Unsupervised Anomaly Detection Based on Full-Reference Image Quality Assessment},
  author={*},
  booktitle={*},
  pages={*},    
  year={2022}
}
```

**The released codes are only allowed for non-commercial use.**
