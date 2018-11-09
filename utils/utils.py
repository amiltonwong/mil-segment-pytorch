# camera-ready

import torch
import torch.nn as nn

import numpy as np

def add_weight_decay(net, l2_value, skip_list=()):
    # https://raberrytv.wordpress.com/2017/10/29/pytorch-weight-decay-made-easy/

    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay, 'weight_decay': l2_value}]

# function for colorizing a label image:
def label_img_to_color(img):
    label_to_color = {
        0: [128, 64,128], # road
        1: [244, 35,232], # sidewalk
        2: [ 70, 70, 70], # building
        3: [102,102,156], # wall
        4: [190,153,153], # fence
        5: [153,153,153], # pole
        6: [250,170, 30], # traffic light
        7: [220,220,  0], # traffic sign
        8: [107,142, 35], # vegetation
        9: [152,251,152], # terrain
        10: [ 70,130,180],# sky
        11: [220, 20, 60],# person
        12: [255,  0,  0],# rider
        13: [  0,  0,142],# car
        14: [  0,  0, 70],# truck
        15: [  0, 60,100],# bus
        16: [  0, 80,100],# train
        17: [  0,  0,230],# motorcycle
        18: [119, 11, 32],# bicycle
        19: [81,  0, 81]  # ground & other classes
        }

    img_height, img_width = img.shape

    img_color = np.zeros((img_height, img_width, 3))
    for row in range(img_height):
        for col in range(img_width):
            label = img[row, col]

            img_color[row, col] = np.array(label_to_color[label])

    return img_color
