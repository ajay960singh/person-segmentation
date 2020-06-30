import cv2
import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image
import random
import time
from models import get_instance_segmentation_model
from utils import *
import argparse

class Inference(object):
    def __init__(self, model_path):
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        model = get_instance_segmentation_model(num_classes = 2) # bakground and foreground num_classes
        # Load the weights (assumes that infernece is run on CPU)
        model.load_state_dict(torch.load('maskrcnn_resnet_50.pt', map_location=torch.device('cpu'))['model'], strict=True)

        return model

    def get_pred(self,img_path):

        #read the img and convert to a tensor
        img = Image.open(img_path).convert("RGB")
        img = F.to_tensor(img)

        # Set the model to evaluation and get predictions
        self.model.eval()
        start = time.time()
        pred = self.model([img])
        print('time_taken: ', time.time()-start)

        # get the masks and only consider masks where the confidence for score is above a threshold
        pred_masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
        pred_score = list(pred[0]['scores'].detach().cpu().numpy())
        pred_class = ['Person '+str(np.round(pred_score[i],3))
                        for i,_ in enumerate(list(pred[0]['labels'].detach().cpu().numpy()))]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                        for i in list(pred[0]['boxes'].detach().cpu().numpy())]
        pred_t = [pred_score.index(x) for x in pred_score if x>0.7][-1]
        pred_masks = pred_masks[:pred_t+1]
        pred_class = pred_class[:pred_t+1]
        pred_boxes = pred_boxes[:pred_t+1]

        # draw masks on img
        img = draw_predictions(pred_masks, pred_class, pred_boxes, img_path)

        return img


if __name__ == '__main__':
    inference = Inference('maskrcnn_resnet_50_tf.pt')
    # img_path = '../PennFudanPed/PNGImages/PennPed00011.png'

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', required=True, type=str)
    parser.add_argument('-o','--output_path', type=str, default="")

    args = parser.parse_args()

    img_path = args.input_path
    masked_img =inference.get_pred(img_path)

    # out_path = 'output_images/' + img_path.split('/')[-1]
    out_path = args.output_path + 'output_'+ img_path.split('/')[-1]
    # save the masked img
    cv2.imwrite(out_path,cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR))
