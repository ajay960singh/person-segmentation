# Person Segmentaion with MaskRCNN

### Description

This is a Pytorch implementation of retraining a person segmentation model, specifically the MaskRCNN Model provided by Torchvision. This reposity provides 
two folders:

* `train/` : This contains the training script, that can be run on Google Colab and contains cells for downloading.
* `api/` : This provides an API to perform inference on your image.

`Note` : Refer to respective `README.md` inside these folders for a more detailed description.

### Model

MaskRCNN is based on FasterRCNN model. 

![Faster R-CNN](https://raw.githubusercontent.com/pytorch/vision/temp-tutorial/tutorials/tv_image03.png)

Mask R-CNN adds an extra branch into Faster R-CNN, which also predicts segmentation masks for each instance.

![Mask R-CNN](https://raw.githubusercontent.com/pytorch/vision/temp-tutorial/tutorials/tv_image04.png)

The backbone used in the model is `resnet50`. The model was optimized for providing the highest IoU on the validation set.
The model can be downloaded [here](https://drive.google.com/file/d/1yHgibFxDNkXlHO3aI4OtLdyTSJgjYI3S/view?usp=sharing).




