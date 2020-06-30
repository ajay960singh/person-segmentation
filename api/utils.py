import numpy as np
import cv2
import random
import string


def _random_colour_masks(image):

    colours = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
                (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
                (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0),
                (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128)]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    rand_color = colours[random.randrange(0,len(colours))]

    r[image == 1], g[image == 1], b[image == 1] = rand_color

    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask, rand_color

def _get_area_mask(pred_masks):
    summed_masks = np.sum(pred_masks, axis=0)
    mask_num_pix = np.sum(summed_masks!=0)
    area_mask = mask_num_pix / (summed_masks.shape[0]*summed_masks.shape[1])
    area_mask_pt = area_mask*100
    return area_mask_pt

def draw_predictions(pred_masks, pred_class, pred_boxes, img_path):

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_img = img.copy()
    # Draw the predictions one-by-one
    for i in range(len(pred_masks)):
        rgb_mask, rand_color = _random_colour_masks(pred_masks[i])
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        cv2.rectangle(img, pred_boxes[i][0], pred_boxes[i][1],color=rand_color, thickness=1)
        cv2.putText(img,pred_class[i], pred_boxes[i][0],
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, rand_color,thickness=1)

    area_mask = _get_area_mask(pred_masks)
    cv2.putText(img,'{"percent_masked" : '+str(np.round(area_mask,3)) + '}',
                (5,15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),thickness=2)

    final_img = np.concatenate((input_img, img), axis=1)

    return final_img

def randomString(stringLength=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))
