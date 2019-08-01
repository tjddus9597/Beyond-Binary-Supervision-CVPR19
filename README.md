# Deep Metric Learning Beyond Binary Supervision
Official pytorch Implementation of [Deep Metric Learning Beyond Binary Supervision](https://arxiv.org/abs/1904.09626), CVPR 2019

## Citing this work
If you find this work useful in your research, please consider citing:

    @inproceedings{kim2019deep,
      title={Deep Metric Learning Beyond Binary Supervision},
      author={Kim, Sungyeon and Seo, Minkyo and Laptev, Ivan and Cho, Minsu and Kwak, Suha},
      booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
      pages={2288--2297},
      year={2019}
    }
    
##  Dependency
* Python 3.6
* Pytorch >=0.4.1
* tqdm (pip install tqdm)
* scipy
* tensorboardX

## Prerequisites 
1. Download pretrained *human pose dataset* with labels from [here](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet)
2. Extract the zip file into `./data/`

## Human Pose Retrieval Quick Start

```bash
python main.py --help

# Train a embedding network of resnet34 (d=128)
# using logratio loss with **dense triplet sampling**.

python main.py --loss logratio \
               --model resnet34 \ 
               --result-name dense_Logratio \
               --optimizer sgd \
               --lr 0.01 \ 
               --lr-decay 1e-4 \ 
               --batch-size 150 \
               --num-NN 5 \
               --embedding-size 128 \
               --sampling dense \
               
# Train a embedding network of resnet34 (d=128)
# using triplet loss (margin=0.03) with **dense triplet sampling**.

python main.py --loss triplet \
               --is-norm True \
               --model resnet34 \ 
               --result-name dense_Triplet \
               --optimizer sgd \
               --lr 0.01 \ 
               --lr-decay 1e-4 \ 
               --batch-size 150 \
               --num-NN 5 \
               --embedding-size 128 \
               --sampling dense \               
               
# Train a embedding network of resnet34 (d=128)
# using triplet loss (margin=0.03) with **binary triplet sampling**.

python main.py --loss triplet \
               --is-norm True \
               --model resnet34 \ 
               --result-name naive_Triplet \
               --optimizer sgd \
               --lr 0.01 \ 
               --lr-decay 1e-4 \ 
               --batch-size 150 \
               --embedding-size 128 \
               --sampling naive \     

