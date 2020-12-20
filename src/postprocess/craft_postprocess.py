# Copyright (c) 2020 FSMLP Authors. All Rights Reserved.

import numpy as np
import cv2
import math

class CRAFTPostprocessTest(object):

    def __init__(self, params):
        self.text_threshold = params["text_threshold"]
        self.link_threshold = params["link_threshold"]
        self.low_text = params["low_text"]
        self.is_dilate = params["is_dilate"]
        self.x_dilate = params["x_dilate"]
        self.y_dilate = params["y_dilate"]
        self.rotated_box = params["rotated_box"]

    def get_boxes(self, textmap, linkmap):

        text_threshold = self.text_threshold
        link_threshold = self.link_threshold
        low_text = self.low_text

        is_dilate = self.is_dilate
        x_dilate = self.x_dilate
        y_dilate = self.y_dilate
        rotated_box = self.rotated_box

        linkmap = linkmap.copy()
        textmap = textmap.copy()
        img_h, img_w = textmap.shape

        # perform thresholding on greyscale map of link_score
        ret, text_score = cv2.threshold(textmap, low_text, 1, 0)

        if is_dilate:
            # custom kernel defined to pad the textmap strictly in the upper left region of each blob
            center_x = int((x_dilate + 1) / 2)
            center_y = int((y_dilate + 1) / 2)

            inner = np.ones(center_x * center_y).reshape(center_y, center_x).astype(np.uint8)
            outer_r = np.zeros((x_dilate - center_x) * center_y).reshape(center_y, (x_dilate - center_x)).astype(np.uint8)
            outer_d = np.zeros((x_dilate) * -1 * (center_y - y_dilate)).reshape(y_dilate - center_y, x_dilate).astype(np.uint8)

            final = np.append(outer_r, inner, 1)
            Vkernel = np.append(outer_d, final, 0)

            # dilation is performed here
            text_score = cv2.dilate(text_score, Vkernel, 1)

        # perform thresholding on greyscale map of link_score
        ret, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)
        text_score_comb = np.clip(text_score + link_score, 0, 1)
        nLabels, labels, stats, _ = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8), connectivity=4)

        det = []

        for k in range(1,nLabels):
            # size filtering
            size = stats[k, cv2.CC_STAT_AREA]
            if size < 10: continue

            # thresholding (commented by me as thresholding didn't produce any significant difference in our case. Could be useful in very noisy images)
            if np.max(textmap[labels==k]) < text_threshold: continue

            # make segmentation map
            segmap = np.zeros(textmap.shape, dtype=np.uint8)
            segmap[labels==k] = 255
            segmap[np.logical_and(link_score==1, text_score==0)] = 0   # remove link area
            x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
            w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
            if is_dilate:
                niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2.4)
            else:
                niter = int(math.sqrt(size * min(w, h) / (w * h)) * 1.8)
            sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
            # boundary check
            if sx < 0 : sx = 0
            if sy < 0 : sy = 0
            if ex >= img_w: ex = img_w
            if ey >= img_h: ey = img_h
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1 + niter, 1 + niter))
            segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

            # make box
            np_contours = np.roll(np.array(np.where(segmap!=0)),1,axis=0).transpose().reshape(-1,2)

            if rotated_box:
                rectangle = cv2.minAreaRect(np_contours)
                points = cv2.boxPoints(rectangle)

                # align diamond-shape
                # w, h = np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[1] - points[2])
                # box_ratio = max(w, h) / (min(w, h) + 1e-5)
                # if abs(1 - box_ratio) <= 0.1:
                #     l, r = min(np_contours[:,0]), max(np_contours[:,0])
                #     t, b = min(np_contours[:,1]), max(np_contours[:,1])
                #     box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)
                
                #alternative approach to align diamond-shape
                index_1, index_2, index_3, index_4 = 0, 1, 2, 3
                if points[1][1] > points[0][1]:
                    index_1 = 0
                    index_4 = 1
                else:
                    index_1 = 1
                    index_4 = 0
                if points[3][1] > points[2][1]:
                    index_2 = 2
                    index_3 = 3
                else:
                    index_2 = 3
                    index_3 = 2

                box = [
                    points[index_1], points[index_2], points[index_3], points[index_4]
                ]

            # *TODO: Fix else condition to align the output style with if part!
            else:
                l, t, w, h = cv2.boundingRect(np_contours)
                box = np.array([l, t, l+w, t+h])

            det.append(box)

        """ Simpler Approach """
        # if not rotated_box:
        #     box_list = []
        #     text_score_comb = (np.clip(text_score_comb, 0, 1) * 255).astype(np.uint8)
        #     Vcnts = cv2.findContours(text_score_comb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #     Vcnts = Vcnts[0] if len(Vcnts) == 2 else Vcnts[1]

        #     for c in reversed(Vcnts):
        #         l,t,w,h = cv2.boundingRect(c)
        #         box_list.append(np.array([l, t, l+w, t+h]))
        # return box_list

        return det


    def adjust_box_size(self, boxes, ratio, ratio_net = 2):
        
        if len(boxes) > 0:
            boxes = np.array(boxes)
            ratio_h = ratio_w = 1 / ratio
            for k in range(len(boxes)):
                if boxes[k] is not None:
                    boxes[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)             
        return boxes


    def __call__(self, textmap, linkmap, ratio):

        boxes = self.get_boxes(textmap, linkmap)
        adj_boxes = self.adjust_box_size(boxes, ratio)
        return adj_boxes