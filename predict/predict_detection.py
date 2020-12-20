# Copyright (c) 2020 FSMLP Authors. All Rights Reserved.

import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

import predict.predict_utility as utility
logger = utility.initial_logger()

from src.preprocess.craft_preprocess import CRAFTPreprocessTest
from src.postprocess.craft_postprocess import CRAFTPostprocessTest
from src.utils.input_utility import get_image_file_list, check_and_read_gif

import onnxruntime
import cv2
import numpy as np
import math
import time


class TextDetector(object):

    def __init__(self, args):
        
        self.det_algorithm = args.det_algorithm
        self.det_render = args.det_render
        self.use_gpu = args.use_gpu
        preprocess_params = {}
        postprocess_params = {}
        if self.det_algorithm == 'CRAFT':
            preprocess_params["canvas_size"] = args.canvas_size
            preprocess_params["mag_ratio"] = args.mag_ratio
            preprocess_params["interpolation"] = cv2.INTER_LINEAR
            self.preprocess = CRAFTPreprocessTest(preprocess_params)
            postprocess_params["text_threshold"] = args.text_threshold
            postprocess_params["link_threshold"] = args.link_threshold
            postprocess_params["low_text"] = args.low_text
            postprocess_params["is_dilate"] = args.is_dilate
            postprocess_params["x_dilate"] = args.x_dilate
            postprocess_params["y_dilate"] = args.y_dilate
            postprocess_params["rotated_box"] = args.rotated_box
            self.postprocess = CRAFTPostprocessTest(postprocess_params)
            self.predictor = utility.create_predictor_onnx(args, mode="det")
        else:
            logger.info("unknown det_algorithm:{}".format(self.det_algorithm))
            sys.exit(0)


    def order_points_clockwise(self, pts):
        """
        reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        # sort the points based on their x-coordinates
        """
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype="float32")
        # print(rect)
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def __call__(self, img):
        ori_im = img.copy()
        if self.det_algorithm == 'CRAFT':
            im, ratio, size_heatmap = self.preprocess(img)
            if im is None:
                return None, 0

            start = time.time()
            
            inputs = {self.predictor.get_inputs()[0].name: np.asarray(im)}
            outputs = self.predictor.run(None, inputs)[0]

            score_text = outputs[0,:,:,0]
            score_link = outputs[0,:,:,1]
            boxes = self.postprocess(score_text, score_link, ratio)
            dt_boxes = self.filter_tag_det_res(boxes, ori_im.shape)

            # *TODO: COMPLETE (if self.det_render:)

        elapse = time.time() - start
        return dt_boxes, elapse

if __name__ == "__main__":
    args = utility.parse_args()
    image_file_list = get_image_file_list(args.image_dir)
    text_detector = TextDetector(args)
    count = 0
    total_time = 0
    draw_img_save = "./inference_results"
    if not os.path.exists(draw_img_save):
        os.makedirs(draw_img_save)
    for image_file in image_file_list:
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        dt_boxes, elapse = text_detector(img)
        if count > 0:
            total_time += elapse
        count += 1
        # print("Predict time of %s:" % image_file, elapse)
        src_im = utility.draw_text_det_res(dt_boxes, image_file)
        img_name_pure = image_file.split("/")[-1]
        cv2.imwrite(
            os.path.join(draw_img_save, "det_res_%s" % img_name_pure), src_im)
    if count > 1:
        print("Avg Time:", total_time / (count - 1))
