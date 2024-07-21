# snn-deeplab
Youngeun Kim děng rén 2022 Neuromorph. Comput. Eng. 2 044015 Lùnwén de fù xiàn,(chāoyuè fēnlèi: Zhíjiē xùnliàn màichōng shénjīng wǎngluò jìnxíng yǔyì fēngē)
84 / 5,000
A reproduction of Youngeun Kim et al. 2022 Neuromorph. Comput. Eng. 2 044015 (Beyond classification: Directly training spiking neural networks for semantic segmentation)
## Description“
This project includes the implementation of ann, snn-deeplab, and snn-fcn codes, using the snntorch library to implement the snn neural network framework
Using the ANN-SNN conversion method, first, ANN-DeepLab is trained with the traditional 2D cross entropy loss. We convert the pre-trained ANN to an SNN with two conversion methods, layer-wise conversion and channel-wise conversion, and use the DeepLab architecture trained on PASCAL VOC2012. The spikes are accumulated in the last layer of all time steps, and the class corresponding to the maximum number of spikes at each pixel location is selected.
## Step
### Prepare Dataset
In augdataset.py under the dataset folder.The dataset in this project is Augmented PASCAL VOC, which is made of PASCAL VOC 2012 and Semantic Boundaries Dataset.
You can view the dataset creation tutorial on this website https://github.com/LGsmile/crea_augmented_fcns.git
Then in the code project, change the dataset address to the dataset address you created.
### Initialization Parameters
You can set visualize = True and compute_weight = True
