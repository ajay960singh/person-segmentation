# Person-segmentation

### Description

The repository provides a training script, `train_maskrcnn.ipynb`, that can be run on Google Colab and contains cells for downloading the dataset and 
training the model. The trained model can be downloaded [here](https://drive.google.com/file/d/1yHgibFxDNkXlHO3aI4OtLdyTSJgjYI3S/view?usp=sharing) and 
some of the sample outputs of the model are saved in the `sample_results/` folder.


### Pre-requisites

The code uses `Python 3.6`. The dependencies required for the repository can be found in `requirements.txt`, which is called from `setup.sh` so run `setup.sh` before running the code.

### Dataset 

The dataset used is Penn-Fudan dataset, which contains images for peson detection and segmentation and can be downloaded [here](https://www.cis.upenn.edu/~jshi/ped_html/).
The dataset contains a total of 170 images, out of which 120 are used for training the model and the remaining 50 are used as validation set for the model.

### Model

MaskRCNN is based on FasterRCNN model. 

![Faster R-CNN](https://raw.githubusercontent.com/pytorch/vision/temp-tutorial/tutorials/tv_image03.png)

Mask R-CNN adds an extra branch into Faster R-CNN, which also predicts segmentation masks for each instance.

![Mask R-CNN](https://raw.githubusercontent.com/pytorch/vision/temp-tutorial/tutorials/tv_image04.png)

The backbone used in the model is `resnet50`. The model was optimized for providing the highest IoU on the validation set.

### Steps to run the code

1. Clone the repository and go inside it.
2. Install the dependendies by running `setup.sh`.
3. Open `train_maskrcnn.ipynb` in Google Collab to view the training script and play with it.

### Performance

* The highest IoU for validation set was `0.65` . IoU was only counted for objects, for which confidence score>0.50.
* Numer of trainable parameters : 171,862,875
* Google Colab GPU training time for 1 epoch : ~112 seconds
* Inference time on CPU : 5.767 seconds

