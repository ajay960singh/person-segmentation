import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import MaskRCNN

# helper function for model
def get_backbone(num_classes):

    # get backbone
    backbone = torchvision.models.resnet50(pretrained=True)

    # remove the fc layers
    new_backbone = torch.nn.Sequential(*(list(backbone.children())[:-2]))
    new_backbone.out_channels = 2048
    model = MaskRCNN(new_backbone, num_classes)
    
    return model

# Define model
def get_instance_segmentation_model(num_classes):

    # get the maskrcnn model
    model = get_backbone(num_classes)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model
