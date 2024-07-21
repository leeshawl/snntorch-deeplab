# crea_augmented_fcns
This project proposes the efficient attention augmented convolution module which comprises convolution, efficient attention and column-row attention operations. We replace some convolutional layers with this augmented module in fully convolutional networks to obtain the augmented convolutional networks, and test them on semantic segmentation task.

# Requirements
- tqdm==4.47.0
- torchvision==0.8.2
- numpy==1.18.5
- torch==1.7.1
- scikit_image==0.16.2
- scipy==1.4.1
- pytz==2020.1
- Pillow==8.1.0
- PyYAML==5.4.1
- skimage==0.0

# Prepare Dataset
The dataset in this project is Augmented PASCAL VOC, which is made of PASCAL VOC 2012 and Semantic Boundaries Dataset. 

## Make the dataset by yourself
If you want to make the augmented dataset by yourself, you can make it following the guidance below.

Download the PASCAL VOC 2012 from this url:    
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

Download the Semantic Boundaries Dataset from this url:
http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz

The directory structures of these two datasets are as below:

PASCAL VOC 2012
```
+ VOCdevkit
  + VOC2012
    + Annotations
    + ImageSets
    + JPEGImages
    + SegmentationClass
    + SegmentationObject 
```
Semantic Boundaries Dataset
```
+ benchmark_RELEASE
  + cls
  + img
  + inst  
  train.txt
  val.txt
```
The annotations in PASCAL VOC 2012 are colorful images of .png type while those in Semantic Boundaries Dataset are of .mat type. So we transform all the annotations into grayscale images before combining these two datasets. We first create a folder in benchmark_RELEASE and name it as tools, then put three python files into this folder which are mat2png.py, convert_labels.py and utils.py respectively.

Use mat2png.py to transform the annotation datas in Semantic Boundaries Dataset. Enter the directory /tools and run mat2png.py. The transformed annotations is in the directory /cls_aug.
```
python mat2png.py cls cls_aug
```
Use convert_labels.py to transform the annotation datas in PASCAL VOC 2012. Copy convert_labels.py and utils.py to the directory /VOCdevkit/VOC2012, and run convert_labels.py.  The transformed annotations is in the directory /SegmentationClass_1D.
```
python convert_labels.py SegmentationClass ImageSets/Segmentation/trainval.txt SegmentationClass_1D
```
Finally, we combine the images and annotations in both datasets to make Augmented PASCAL VOC. And put the indexed files train_aug.txt and val.txt into the directory /ImageSets/Segmentation of this augmented dataset. The directory structure of Augmented PASCAL VOC is as below:
```
+ VOC2012    
    + ImageSets
      +Segmentation
        train_aug.txt
        val.txt
    + JPEGImages
    + SegmentationClass    
```
## Download the dataset directly
If you want to download the dataset directly, you can get it [here](https://pan.baidu.com/s/1Ux2FOkUlwrWnMOEqq3Thow), and the code is 'etli'.

# Training
Enter the directory /AAFCNS/trainaugfcn, there are several python files to train different models. The strings after 'train' in the names of these python files are the names of models(e.g. 'train_crea_fcn32sc3' represents the model that replaces the last convolution layer wiht efficient attention augmented convoluton module in the third, fourth and fifth convolutional blocks of the baseline fcn32s.)

You should first change the root to your own path of the dataset in this code of the python files:
```python
root = osp.expanduser('~/')
```
Then you can start training. For example, you can train crea_fcn32sc3 by running train_crea_fcn32sc3.py in the directory /AAFCNS/trainaugfcn:
```
python train_crea_fcn32sc3.py
```
