## RLD

## Prerequisites

- Python 2.7
- GPU Memory >= 6G
- Numpy
- Pytorch 0.4.1


## Getting started
### Installation
- Install Pytorch from http://pytorch.org/
- Install Torchvision from the source
```
git clone https://github.com/pytorch/vision
cd vision
python setup.py install
```
Because pytorch and torchvision are ongoing projects.

## Dataset & Preparation
Download [Market1501 Dataset](http://www.liangzheng.org/Project/project_reid.html),[CUHK03](https://github.com/zhunzhong07/person-re-ranking/tree/master/CUHK03-NP),[DukeMTMC](https://github.com/layumi/DukeMTMC-reID_evaluation)

Preparation: Put the images with the same id in one folder. You may use 
```bash
python prepare.py
```
Remember to change the dataset path to your own path.


## Train
Train a model by
```bash
python train.py --gpu_ids 0 --name model1 --train_all --batchsize 32  --data_dir your_data_path --weight 0.2
```
`--gpu_ids` which gpu to run.

`--name` the name of model.

`--data_dir` the path of the training data.

`--baseline` without using the RLD.

`--train_all` using all images to train. 

`--batchsize` batch size.

`--erasing_p` random erasing probability.

`--weight` for two training loss weighting.

Train a model with random erasing by
```bash
python train.py --gpu_ids 0 --name model1 --train_all --batchsize 32  --data_dir your_data_path --weight 0.2 --erasing_p 0.5
```

## Test
Use trained model to extract feature by
```bash
python test.py --gpu_ids 0 --name model1 --test_dir your_data_path  --which_epoch 59 --dataset_name market
```
`--gpu_ids` which gpu to run.

`--name` the dir name of trained model.

`--which_epoch` select the i-th model.

`--test_dir` the path of the testing data.

`--dataset_name` the name of the testing dataset.

## Evaluation
```bash
python evaluate.py --mat-path model1
```
`--mat-path` the dir name of trained model.

It will output Rank@1, Rank@5, Rank@10 and mAP results.

For mAP calculation, you also can refer to the [C++ code for Oxford Building](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/compute_ap.cpp). We use the triangle mAP calculation (consistent with the Market1501 original code).

## Note

The baseline has been well-trained in  [repository](https://github.com/layumi/Person_reID_baseline_pytorch).  


## Citation
Our codes are mainly based on this [repository](https://github.com/layumi/Person_reID_baseline_pytorch) 

If you use this code, please kindly cite it in your paper

```
@article{guangcong2019RLD,
  title={Discovering Underlying Person Structure Pattern with Relative Local Distance for Person Re-identification},
  author={Wang, Guangcong and Lai, Jianhuang and Xie, Zhenyu and Xie, Xiaohua},
  journal={arXiv preprint arXiv:1901.10100},
  year={2019}
}
```

 