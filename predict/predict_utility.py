# Copyright (c) 2020 FSMLP Authors. All Rights Reserved.

import argparse
import os, sys

import logging
import cv2
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont
import math
import unidecode

import onnxruntime
from paddle.fluid.core import PaddleTensor
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor


def initial_logger():
    FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT)
    logger = logging.getLogger(__name__)
    return logger

logger = initial_logger()


def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")
    
    parser = argparse.ArgumentParser()
    # params for prediction engine
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--gpu_mem", type=int, default=8000)
    parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
    parser.add_argument("--use_zero_copy_run", type=str2bool, default=False)    
    parser.add_argument("--use_pdserving", type=str2bool, default=False)
    
    # params for text detector
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--det_algorithm", type=str, default='CRAFT')
    parser.add_argument("--det_model_dir", type=str)
    parser.add_argument("--det_max_side_len", type=float, default=960)
    parser.add_argument("--det_render", type=str2bool, default=True)

    # CRAFT params
    """ Data PreProcessing """
    parser.add_argument(
        '--canvas_size', default=1920, 
        type=int, help='image size for inference')
    parser.add_argument(
        '--mag_ratio', default=1.5, 
        type=float, help='image magnification ratio')
    """ Detection Model Specifications """
    parser.add_argument(
        '--text_threshold', 
        default=0.7, type=float, 
        help='text confidence threshold') #This threshold is not used in our case.
    parser.add_argument(
        '--low_text', 
        default=0.35, type=float, 
        help='text low-bound score') #0.003 was used with 0.2-tt and 0.1-lt
    parser.add_argument(
        '--link_threshold', 
        default=0.1, type=float, help='link confidence threshold')
    parser.add_argument(
        '--rotated_box', 
        type=str2bool, default=True, 
        help='use this to get rotated rectangles (bounding box)') # Currently not handling for rotated boxes
    parser.add_argument('--x_dilate', default=1, type=int, help='left x-padding during post processing')
    parser.add_argument('--y_dilate', default=3, type=int, help='up y-padding during post processing')


    # params for text classifier
    parser.add_argument("--use_box_orientation", type=str2bool, default=False)
    parser.add_argument("--bor_model_dir", type=str)
    parser.add_argument("--bor_image_shape", type=str, default="3, 48, 192")
    parser.add_argument("--label_list", type=list, default=['0', '180'])
    parser.add_argument("--bor_batch_num", type=int, default=30)
    parser.add_argument("--bor_thresh", type=float, default=0.9)

    # params for text recognizer
    parser.add_argument("--rec_algorithm", type=str, default='CRNN')
    parser.add_argument("--rec_model_dir", type=str)
    parser.add_argument("--rec_render", type=str2bool, default=True)
    parser.add_argument("--rec_image_shape", type=str, default="3, 32, 320")
    parser.add_argument("--rec_char_type", type=str, default='ch')
    parser.add_argument("--rec_whitelist", type=str, default='') #whitelist characters while prediction
    parser.add_argument("--rec_blacklist", type=str, default='') #blacklist characters while prediction
    parser.add_argument("--rec_batch_num", type=int, default=6)
    parser.add_argument("--max_text_length", type=int, default=25)
    parser.add_argument(
        "--rec_char_dict_path",
        type=str,
        default="./src/utils/phonemes/ppocr_keys_v1.txt")
    parser.add_argument("--use_space_char", type=str2bool, default=True)
    parser.add_argument(
        "--vis_font_path", type=str, default="./src/utils/fonts/simfang.ttf")
    
    # params for beam-search decoder in text recognizer
    parser.add_argument("--use_beam_search", type=str2bool, default=False)
    parser.add_argument("--beam_width", type=int, default=100)
    parser.add_argument("--beam_lm_dir", type=str, default='')
    parser.add_argument("--beam_alpha", type=float, default=0)
    parser.add_argument("--beam_beta", type=float, default=0)
    parser.add_argument("--beam_cutoff_top", type=int, default=40)
    parser.add_argument("--beam_cutoff_prob", type=float, default=1.0)

    # params for spell-checker in text recognizer
    parser.add_argument("--use_spell_check", type=str2bool, default=False)
    parser.add_argument("--spell_language", type=str, default="en")
    parser.add_argument("--spell_case_sensitive", type=str2bool, default=False)
    parser.add_argument("--spell_tokenizer", type=str, default='NLTK')
    parser.add_argument("--spell_word_freq", type=str, default='')
    parser.add_argument("--spell_text_corpus", type=str, default='')

    # params for merging resulting values
    parser.add_argument("--merge_boxes", type=str2bool, default=False)
    parser.add_argument("--merge_slope_thresh", type=float, default=0.1)
    parser.add_argument("--merge_ycenter_thresh", type=float, default=0.5)
    parser.add_argument("--merge_height_thresh", type=float, default=0.5)
    parser.add_argument("--merge_width_thresh", type=float, default=1.0)
    parser.add_argument("--merge_add_margin", type=float, default=0.05)
    

    return parser.parse_args()

def create_predictor_onnx(args, mode):
    if mode == 'det':
        model_dir = args.det_model_dir
    if model_dir is None:
        logger.info("not find {} model file path {}".format(mode, model_dir))
        sys.exit(0)  
    model_file_path = model_dir + "/model.onnx"
    if not os.path.exists(model_file_path):
        logger.info("not find model file path {}".format(model_file_path))
        sys.exit(0)
    
    session = onnxruntime.InferenceSession(model_file_path)
    return session


def create_predictor(args, mode):
    if mode == "det":
        model_dir = args.det_model_dir
    elif mode == 'bor':
        model_dir = args.bor_model_dir
    else:
        model_dir = args.rec_model_dir

    if model_dir is None:
        logger.info("not find {} model file path {}".format(mode, model_dir))
        sys.exit(0)
    model_file_path = model_dir + "/model"
    params_file_path = model_dir + "/params"
    if not os.path.exists(model_file_path):
        logger.info("not find model file path {}".format(model_file_path))
        sys.exit(0)
    if not os.path.exists(params_file_path):
        logger.info("not find params file path {}".format(params_file_path))
        sys.exit(0)

    config = AnalysisConfig(model_file_path, params_file_path)

    if args.use_gpu:
        config.enable_use_gpu(args.gpu_mem, 0)
    else:
        config.disable_gpu()
        config.set_cpu_math_library_num_threads(6)
        if args.enable_mkldnn:
            # cache 10 different shapes for mkldnn to avoid memory leak
            config.set_mkldnn_cache_capacity(10)
            config.enable_mkldnn()

    # config.enable_memory_optim()
    config.disable_glog_info()

    if args.use_zero_copy_run:
        config.delete_pass("conv_transpose_eltwiseadd_bn_fuse_pass")
        config.switch_use_feed_fetch_ops(False)
    else:
        config.switch_use_feed_fetch_ops(True)

    predictor = create_paddle_predictor(config)
    input_names = predictor.get_input_names()
    for name in input_names:
        input_tensor = predictor.get_input_tensor(name)
    output_names = predictor.get_output_names()
    output_tensors = []
    for output_name in output_names:
        output_tensor = predictor.get_output_tensor(output_name)
        output_tensors.append(output_tensor)
    return predictor, input_tensor, output_tensors


def draw_text_det_res(dt_boxes, img_path):
    src_im = cv2.imread(img_path)
    for box in dt_boxes:
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
    return src_im

# TODO: Cut this and other sections related to rendering and paste into src/utils/output_utility.py
def draw_ocr_box_txt(image,
                     boxes,
                     txts,
                     scores=None,
                     drop_score=0.5,
                     font_path="./doc/simfang.ttf"):
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = Image.new('RGB', (w, h), (255, 255, 255))

    import random

    random.seed(0)
    draw_left = ImageDraw.Draw(img_left)
    draw_right = ImageDraw.Draw(img_right)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:
            continue
        color = (random.randint(0, 255), random.randint(0, 255),
                 random.randint(0, 255))
        draw_left.polygon(box, fill=color)
        draw_right.polygon(
            [
                box[0][0], box[0][1], box[1][0] - 3, box[1][1], box[2][0] - 3,
                box[2][1], box[3][0], box[3][1]
            ],
            outline=color)
        box_height = math.sqrt((box[0][0] - box[3][0])**2 + (box[0][1] - box[3][
            1])**2)
        box_width = math.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][
            1])**2)
        if box_height > 2 * box_width:
            font_size = max(int(box_width * 0.9), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            cur_y = box[0][1]
            for c in txt:
                char_size = font.getsize(c)
                draw_right.text(
                    (box[0][0] + 3, cur_y), unidecode.unidecode(c), fill=(0, 0, 0), font=font)
                cur_y += char_size[1]
        else:
            img_fraction = 1
            # font_size=1
            font_change=False
            font_size = max(int(box_height * 0.8), 10)
            font = ImageFont.truetype(font_path, font_size)
            while font.getsize(txt)[0] > img_fraction*box_width:
                # iterate until the text size is just larger than the criteria
                font_change=True
                font_size -= 1
                font = ImageFont.truetype(font_path, font_size)
            if font_change:
                font_size +=1
            font = ImageFont.truetype(font_path, font_size)
            draw_right.text(
                [box[0][0]+3, box[0][1]+3], unidecode.unidecode(txt), fill=(0, 0, 0), font=font)

            wid,het = font.getsize(txt)
            draw_right.polygon(
            [
                box[0][0], box[0][1], box[0][0] + wid + 6, box[0][1], box[0][0] + wid + 6,
                box[0][1] + het + 6, box[0][0], box[0][1] + het + 6
            ],
            outline=color)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(img_right, (w, 0, w * 2, h))
    return np.array(img_show)
    