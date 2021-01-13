# Copyright (c) 2020 FSMLP Authors. All Rights Reserved.

import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.data.datasets import register_coco_instances

import glob
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='Checkbox Detection')
parser.add_argument('--config', default='weights/cb_detection/config.yaml', type=str, help='config_file')
parser.add_argument('--weights', default='weights/cb_detection/model_final.pth', type=str, help='pretrained model')
parser.add_argument('--threshold', default=0.8, type=float, help='confidence threshold')
parser.add_argument('--square_size', default=1024, type=int, help='image input resize')
parser.add_argument('--input_folder', default='CBinput/', type=str, help='folder path to input images')
parser.add_argument('--output_folder', default='CBoutput/', type=str, help='folder path to results')
parser.add_argument('--cpu', default=False, type=str2bool, help='Use cpu for inference')

args = parser.parse_args()

files = glob.glob(args.input_folder + "/*.jpg")
result_folder = args.output_folder
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

def resize_aspect_ratio(img, filename, square_size):
    h,w,c = img.shape
    ratio = square_size/max(h,w)
    th, tw = int(h*ratio), int(w*ratio)
    proc = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)

    extra_l, extra_u = 0,0
    targeth,targetw = th,tw
    if targetw < square_size:
        extra = square_size - targetw
        extra_l = int(extra/2)

    if targeth < square_size:
        extra = square_size - targeth
        extra_u = int(extra/2)

    resized = np.zeros((square_size, square_size, c), dtype=np.float32) + 255
    resized[extra_u:targeth+extra_u, extra_l:targetw+extra_l, :] = proc

    return resized, ratio, extra_l, extra_u

def adjust_box_size(boxes, ratio, extral, extrau, ratio_net = 1):
        
    if len(boxes) > 0:
        boxes = np.array(boxes)
        ratio_h = ratio_w = 1 / ratio
        for k in range(len(boxes)):
            if boxes[k] is not None:
                boxes[k] -= (extral, extrau)  
                boxes[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)             
    return boxes


if __name__ == "__main__":
    cfg = get_cfg()
    cfg.merge_from_file(args.config)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.threshold
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    if args.cpu:
        cfg.MODEL.DEVICE='cpu'
    predictor = DefaultPredictor(cfg)

    t = time.time()
    for f in files:
        img = cv2.imread(f)
        filename = f.split('/')[-1]
        resized, ratio, extral, extrau = resize_aspect_ratio(img, filename, args.square_size)
        outputs = predictor(resized)

        ### Use this when visualizing raw images
        v = Visualizer(resized[:, :, ::-1],scale=1)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(result_folder+'label_'+filename, out.get_image()[:,:,::-1])

        instances = outputs["instances"].to("cpu")
        print(instances)
        l1 = []
        for i in (instances.pred_boxes):
            l1.append(np.split(np.asarray(i), 2))

        boxes = adjust_box_size(l1, ratio, extral, extrau)

        txtfile = open(result_folder+filename[:-4]+'.csv', 'w')
        txtfile.write('startX,startY,endX,endY\r\n')
        for i,box in enumerate(boxes):
            poly = np.array(box).astype(np.int32).reshape((-1))
            poly.reshape(-1,2)
            strResult = ','.join([str(p) for p in poly]) + '\r\n'
            txtfile.write(strResult)
            cv2.rectangle(img, (poly[0],poly[1]), (poly[2],poly[3]), (0,255,0), 6)

        txtfile.close()
        cv2.imwrite(result_folder+filename, img)

    print()
    print("Total time: {} s".format(time.time() - t))