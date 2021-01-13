# Copyright (c) 2020 FSMLP Authors. All Rights Reserved.

import os
import sys
import math
import time
import copy
import tqdm

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

import predict.predict_utility as utility
logger = utility.initial_logger()

import xlsxwriter as xw
import cv2
import numpy as np
import pandas as pd
from PIL import Image

import predict.predict_box_orientation as predict_bor
import predict.predict_recognition as predict_rec
import predict.predict_detection as predict_det
from predict.predict_utility import draw_ocr_box_txt
from src.utils.input_utility import get_image_file_list, check_and_read_gif
import shutil


class TextSystem(object):
    def __init__(self, args):
        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_box_orientation = args.use_box_orientation
        
        if self.use_box_orientation:
            self.text_classifier = predict_bor.TextClassifier(args) 
        
        self.merge_boxes = args.merge_boxes

        if self.merge_boxes:
            self.merge_slope_thresh = args.merge_slope_thresh
            self.merge_ycenter_thresh = args.merge_ycenter_thresh
            self.merge_height_thresh = args.merge_height_thresh
            self.merge_width_thresh = args.merge_width_thresh
            self.merge_add_margin = args.merge_add_margin

    def get_rotate_crop_image(self, img, points):
        '''
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        '''
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def print_draw_crop_rec_res(self, img_crop_list, rec_res):
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite("./output/img_crop_%d.jpg" % bno, img_crop_list[bno])
            # print(bno, rec_res[bno])

    @staticmethod
    def sorted_boxes(dt_boxes):
        """
        Sort text boxes in order from top to bottom, left to right
        args:
            dt_boxes(array):detected text boxes with shape [4, 2]
        return:
            sorted boxes(array) with shape [4, 2]
        """
        num_boxes = dt_boxes.shape[0]
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)

        for i in range(num_boxes - 1):
            if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                    (_boxes[i + 1][0][0] < _boxes[i][0][0]):
                tmp = _boxes[i]
                _boxes[i] = _boxes[i + 1]
                _boxes[i + 1] = tmp
        return _boxes

    @staticmethod
    def merge_text_boxes(dt_boxes, rec_res, **params):
        dt_boxes = np.asarray(dt_boxes)
        polys = np.empty((len(dt_boxes), 8))
        polys[:,0] = dt_boxes[:,0,0]
        polys[:,1] = dt_boxes[:,0,1]
        polys[:,2] = dt_boxes[:,1,0]
        polys[:,3] = dt_boxes[:,1,1]
        polys[:,4] = dt_boxes[:,2,0]
        polys[:,5] = dt_boxes[:,2,1]
        polys[:,6] = dt_boxes[:,3,0]
        polys[:,7] = dt_boxes[:,3,1]
        slope_ths = params["slope_thresh"]
        ycenter_ths = params["ycenter_thresh"]
        height_ths = params["height_thresh"]
        width_ths = params["width_thresh"]
        add_margin = params["add_margin"]

        horizontal_list, free_list_box, free_list_text, combined_list, merged_list_box, merged_list_text = [],[],[],[],[],[]

        for i, poly in enumerate(polys):
            slope_up = (poly[3]-poly[1])/np.maximum(10, (poly[2]-poly[0]))
            slope_down = (poly[5]-poly[7])/np.maximum(10, (poly[4]-poly[6]))
            if max(abs(slope_up), abs(slope_down)) < slope_ths:
                x_max = max([poly[0],poly[2],poly[4],poly[6]])
                x_min = min([poly[0],poly[2],poly[4],poly[6]])
                y_max = max([poly[1],poly[3],poly[5],poly[7]])
                y_min = min([poly[1],poly[3],poly[5],poly[7]])
                horizontal_list.append([x_min, x_max, y_min, y_max, 0.5*(y_min+y_max), y_max-y_min, rec_res[i][0], rec_res[i][1],str(poly)])
            else:
                height = np.linalg.norm( [poly[6]-poly[0],poly[7]-poly[1]])
                margin = int(1.44*add_margin*height)
                theta13 = abs(np.arctan( (poly[1]-poly[5])/np.maximum(10, (poly[0]-poly[4]))))
                theta24 = abs(np.arctan( (poly[3]-poly[7])/np.maximum(10, (poly[2]-poly[6]))))
                # do I need to clip minimum, maximum value here?
                x1 = poly[0] - np.cos(theta13)*margin
                y1 = poly[1] - np.sin(theta13)*margin
                x2 = poly[2] + np.cos(theta24)*margin
                y2 = poly[3] - np.sin(theta24)*margin
                x3 = poly[4] + np.cos(theta13)*margin
                y3 = poly[5] + np.sin(theta13)*margin
                x4 = poly[6] - np.cos(theta24)*margin
                y4 = poly[7] + np.sin(theta24)*margin

                free_list_box.append(np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]]))
                free_list_text.append([rec_res[i][0], rec_res[i][1],str(poly), rec_res[i][0]])

        horizontal_list = sorted(horizontal_list, key=lambda item: item[4])

        # combine box
        new_box = []
        for poly in horizontal_list:

            if len(new_box) == 0:
                b_height = [poly[5]]
                b_ycenter = [poly[4]]
                new_box.append(poly)
            else:
                # comparable height and comparable y_center level up to ths*height
                if (abs(np.mean(b_height) - poly[5]) < height_ths*np.mean(b_height)) and (abs(np.mean(b_ycenter) - poly[4]) < ycenter_ths*np.mean(b_height)):
                    b_height.append(poly[5])
                    b_ycenter.append(poly[4])
                    new_box.append(poly)
                else:
                    b_height = [poly[5]]
                    b_ycenter = [poly[4]]
                    combined_list.append(new_box)
                    new_box = [poly]
        combined_list.append(new_box)

        # merge list use sort again
        for boxes in combined_list:
            if len(boxes) == 1: # one box per line
                box = boxes[0]
                margin = int(add_margin*box[5])
                _x0 = _x3 = box[0]-margin
                _y0 = _y1 = box[2]-margin
                _x1 = _x2 = box[1]+margin
                _y2 = _y3 = box[3]+margin
                merged_list_box.append(np.array([[_x0,_y0],[_x1,_y1],[_x2,_y2],[_x3,_y3]]))
                merged_list_text.append([box[6], box[7], box[8], box[6]])
            else: # multiple boxes per line
                boxes = sorted(boxes, key=lambda item: item[0])

                merged_box, new_box = [],[]
                for box in boxes:
                    if len(new_box) == 0:
                        x_max = box[1]
                        new_box.append(box)
                    else:
                        if abs(box[0]-x_max) < width_ths *(box[3]-box[2]): # merge boxes
                            x_max = box[1]
                            new_box.append(box)
                        else:
                            x_max = box[1]
                            merged_box.append(new_box)
                            new_box = [box]
                if len(new_box) >0: merged_box.append(new_box)

                for mbox in merged_box:
                    if len(mbox) != 1: # adjacent box in same line
                        # do I need to add margin here?
                        x_min = min(mbox, key=lambda x: x[0])[0]
                        x_max = max(mbox, key=lambda x: x[1])[1]
                        y_min = min(mbox, key=lambda x: x[2])[2]
                        y_max = max(mbox, key=lambda x: x[3])[3]
                        text_comb = str(mbox[0][6]) if isinstance(mbox[0][6], str) else ''
                        sum_score = mbox[0][7]
                        box_id = str(mbox[0][8])
                        text_id = str(mbox[0][6]) if isinstance(mbox[0][6], str) else ''
                        for val in range(len(mbox)-1):
                            if isinstance(mbox[val+1][6], str):
                                strin = mbox[val+1][6]
                            else:
                                strin = ''
                            text_comb += ' ' + strin
                            sum_score += mbox[val+1][7]
                            box_id += '|||' + str(mbox[val+1][8])
                            text_id += '|||' + strin 
                        avg_score = sum_score / len(mbox)
                        margin = int(add_margin*(y_max - y_min))

                        # merged_list.append([x_min-margin, x_max+margin, y_min-margin, y_max+margin, text_comb, avg_score])
                        _x0 = _x3 = x_min-margin
                        _y0 = _y1 = y_min-margin
                        _x1 = _x2 = x_max+margin
                        _y2 = _y3 = y_max+margin
                        merged_list_box.append(np.array([[_x0,_y0],[_x1,_y1],[_x2,_y2],[_x3,_y3]]))
                        merged_list_text.append([text_comb, avg_score, box_id, text_id])

                    else: # non adjacent box in same line
                        box = mbox[0]

                        margin = int(add_margin*(box[3] - box[2]))
                        # merged_list.append([box[0]-margin,box[1]+margin,box[2]-margin,box[3]+margin, box[6], box[7]])
                        _x0 = _x3 = box[0]-margin
                        _y0 = _y1 = box[2]-margin
                        _x1 = _x2 = box[1]+margin
                        _y2 = _y3 = box[3]+margin
                        merged_list_box.append(np.array([[_x0,_y0],[_x1,_y1],[_x2,_y2],[_x3,_y3]]))
                        merged_list_text.append([box[6], box[7], box[8], box[6]])

        # may need to check if box is really in image
        return free_list_box, free_list_text, merged_list_box, merged_list_text

    def __call__(self, img):
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)
        # print("dt_boxes num : {}, elapse : {}".format(len(dt_boxes), elapse))
        if dt_boxes is None:
            return None, None
        img_crop_list = []

        dt_boxes = self.sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)

        if self.use_box_orientation:
            img_crop_list, angle_list, elapse = self.text_classifier(
                img_crop_list)
            # print("bor num  : {}, elapse : {}".format(
            #     len(img_crop_list), elapse, angle_list))

        rec_res, elapse = self.text_recognizer(img_crop_list)

        # print("rec_res num  : {}, elapse : {}".format(len(rec_res), elapse))

        if self.merge_boxes:
            free_box, free_text, merged_box, merged_text = self.merge_text_boxes(
                dt_boxes, rec_res,
                slope_thresh = self.merge_slope_thresh,
                ycenter_thresh = self.merge_ycenter_thresh,
                height_thresh = self.merge_height_thresh,
                width_thresh = self.merge_width_thresh,
                add_margin = self.merge_add_margin)
            dt_boxes = free_box + merged_box
            rec_res = free_text + merged_text

        # self.print_draw_crop_rec_res(img_crop_list, rec_res)

        return dt_boxes, rec_res

def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    text_sys = TextSystem(args)
    is_visualize = True
    font_path = args.vis_font_path
    
    outer = tqdm.tqdm(total=len(image_file_list), desc=f'Folder: {args.image_dir}', unit='file', position=0, colour='green')
    
    file_log = tqdm.tqdm(total=0, position=1, bar_format='{desc}')
    
    for image_file in image_file_list:
        
        file_name = os.path.basename(image_file)
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        starttime = time.time()
        dt_boxes, rec_res = text_sys(img)
        elapse = time.time() - starttime

        file_log.set_description_str(f'Current File:\t{file_name}\t\t||\tTotal Bounding Boxes:\t{len(rec_res)}')
        
        drop_score = 0.5
        if is_visualize:
            image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            boxes = dt_boxes
            txts = [rec_res[i][0] for i in range(len(rec_res))]
            scores = [rec_res[i][1] for i in range(len(rec_res))]
            draw_img = draw_ocr_box_txt(
                image,
                boxes,
                txts,
                scores,
                drop_score=drop_score,
                font_path=font_path)
            draw_img_save = args.output_dir
            if not os.path.exists(draw_img_save):
                os.makedirs(draw_img_save)
            cv2.imwrite(
                os.path.join(draw_img_save, os.path.basename(image_file)),
                draw_img[:, :, ::-1])

            np_boxes = (np.rint(np.asarray(boxes))).astype(int)
            
            if args.merge_boxes:
                box_ids = [rec_res[i][2] for i in range(len(rec_res))]
                txt_ids = [rec_res[i][3] for i in range(len(rec_res))]
                stacked = np.stack((np_boxes[:,0,0], np_boxes[:,0,1], np_boxes[:,2,0], np_boxes[:,2,1], np.asarray(txts), np.asarray(box_ids), np.asarray(txt_ids)))
                df = pd.DataFrame(np.transpose(stacked), columns=["startX", "startY", "endX", "endY", "OCR", "BoxIDs", "TextIDs"])
                
            else:
                stacked = np.stack((np_boxes[:,0,0], np_boxes[:,0,1], np_boxes[:,2,0], np_boxes[:,2,1], np.asarray(txts)))
                df = pd.DataFrame(np.transpose(stacked), columns=["startX", "startY", "endX", "endY", "OCR"])
            
            save_fold = args.output_dir
            if not os.path.exists(save_fold):
                os.makedirs(save_fold)
            csv_file = save_fold + os.path.basename(image_file)[:-4] + '.csv'
            df.to_csv(csv_file, index=False)
            
            if args.print_to_excel:
                excel_file = save_fold + os.path.basename(image_file)[:-4] + '.xlsx' 
                excelbook = xw.Workbook(excel_file)
                excelsheet = excelbook.add_worksheet('Sheet1')
                excelsheet.set_column(4,4,31)
                excelsheet.set_column(5,5,25)
                excelsheet.set_column(6,7,20)
                excelsheet.set_default_row(15)
                
                bold = excelbook.add_format({'bold':True})
                excelsheet.write(0,0,'start_X',bold)
                excelsheet.write(0,1,'start_Y',bold)
                excelsheet.write(0,2,'end_X',bold)
                excelsheet.write(0,3,'end_Y',bold)
                excelsheet.write(0,4,'Image',bold)
                excelsheet.write(0,5,'Text',bold)
                excelsheet.write(0,6,'Ground_Truth',bold)
                excelsheet.write(0,7,'Label',bold)
                
                box_fold = save_fold + os.path.basename(image_file)[:-4]
                if not os.path.exists(box_fold):
                    os.makedirs(box_fold)
                else:
                    shutil.rmtree(box_fold)
                    os.makedirs(box_fold)
                for i in range(np_boxes.shape[0]):
                    excelsheet.write(i+1,0,np_boxes[i,0,0])
                    excelsheet.write(i+1,1,np_boxes[i,0,1])
                    excelsheet.write(i+1,2,np_boxes[i,2,0])
                    excelsheet.write(i+1,3,np_boxes[i,2,1])
                    excelsheet.write(i+1,5,txts[i])
                    excelsheet.write(i+1,6,'---')
                    excelsheet.write(i+1,7,'O')
                    
                    #extract roi from the image
                    roi = img[np_boxes[i,0,1]:np_boxes[i,2,1], np_boxes[i,0,0]:np_boxes[i,2,0], :]
                    
                    try:
                        h,w,_=roi.shape
                        height = 20
                        ratio = height/h
                        width = int(ratio*w)
                        resized = cv2.resize(roi, (width, height), interpolation=cv2.INTER_LINEAR)
                    except:
                        resized = roi
                    
                    try: 
                        bbox_file = os.path.join(box_fold,str(i+1)+'.jpg')
                        cv2.imwrite(bbox_file, resized)
                        excelsheet.insert_image(i+1, 4, bbox_file, {'x_offset':3, 'y_offset':2, 'object_position':1})
                    except:
                        print('Unable to write image!')
                
                excelbook.close()   
            
        outer.update(1)
            

if __name__ == "__main__":
    start = time.time()
    main(utility.parse_args())