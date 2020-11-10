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


def resize_img(img, input_size=600):
    """
    resize img and limit the longest side of the image to input_size
    """
    img = np.array(img)
    im_shape = img.shape
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(input_size) / float(im_size_max)
    im = cv2.resize(img, None, None, fx=im_scale, fy=im_scale)
    return im


def draw_ocr(image,
             boxes,
             txts=None,
             scores=None,
             drop_score=0.5,
             font_path="./doc/simfang.ttf"):
    """
    Visualize the results of OCR detection and recognition
    args:
        image(Image|array): RGB image
        boxes(list): boxes with shape(N, 4, 2)
        txts(list): the texts
        scores(list): txxs corresponding scores
        drop_score(float): only scores greater than drop_threshold will be visualized
        font_path: the path of font which is used to draw text
    return(array):
        the visualized img
    """
    if scores is None:
        scores = [1] * len(boxes)
    box_num = len(boxes)
    for i in range(box_num):
        if scores is not None and (scores[i] < drop_score or
                                   math.isnan(scores[i])):
            continue
        box = np.reshape(np.array(boxes[i]), [-1, 1, 2]).astype(np.int64)
        image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)
    if txts is not None:
        img = np.array(resize_img(image, input_size=600))
        txt_img = text_visual(
            txts,
            scores,
            img_h=img.shape[0],
            img_w=600,
            threshold=drop_score,
            font_path=font_path)
        img = np.concatenate([np.array(img), np.array(txt_img)], axis=1)
        return img
    return image

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

def str_count(s):
    """
    Count the number of Chinese characters,
    a single English character and a single number
    equal to half the length of Chinese characters.

    args:
        s(string): the input of string
    return(int):
        the number of Chinese characters
    """
    import string
    count_zh = count_pu = 0
    s_len = len(s)
    en_dg_count = 0
    for c in s:
        if c in string.ascii_letters or c.isdigit() or c.isspace():
            en_dg_count += 1
        elif c.isalpha():
            count_zh += 1
        else:
            count_pu += 1
    return s_len - math.ceil(en_dg_count / 2)


def text_visual(texts,
                scores,
                img_h=400,
                img_w=600,
                threshold=0.,
                font_path="./doc/simfang.ttf"):
    """
    create new blank img and draw txt on it
    args:
        texts(list): the text will be draw
        scores(list|None): corresponding score of each txt
        img_h(int): the height of blank img
        img_w(int): the width of blank img
        font_path: the path of font which is used to draw text
    return(array):

    """
    if scores is not None:
        assert len(texts) == len(
            scores), "The number of txts and corresponding scores must match"

    def create_blank_img():
        blank_img = np.ones(shape=[img_h, img_w], dtype=np.int8) * 255
        blank_img[:, img_w - 1:] = 0
        blank_img = Image.fromarray(blank_img).convert("RGB")
        draw_txt = ImageDraw.Draw(blank_img)
        return blank_img, draw_txt

    blank_img, draw_txt = create_blank_img()

    font_size = 20
    txt_color = (0, 0, 0)
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")

    gap = font_size + 5
    txt_img_list = []
    count, index = 1, 0
    for idx, txt in enumerate(texts):
        index += 1
        if scores[idx] < threshold or math.isnan(scores[idx]):
            index -= 1
            continue
        first_line = True
        while str_count(txt) >= img_w // font_size - 4:
            tmp = txt
            txt = tmp[:img_w // font_size - 4]
            if first_line:
                new_txt = str(index) + ': ' + txt
                first_line = False
            else:
                new_txt = '    ' + txt
            draw_txt.text((0, gap * count), new_txt, txt_color, font=font)
            txt = tmp[img_w // font_size - 4:]
            if count >= img_h // gap - 1:
                txt_img_list.append(np.array(blank_img))
                blank_img, draw_txt = create_blank_img()
                count = 0
            count += 1
        if first_line:
            new_txt = str(index) + ': ' + txt + '   ' + '%.3f' % (scores[idx])
        else:
            new_txt = "  " + txt + "  " + '%.3f' % (scores[idx])
        draw_txt.text((0, gap * count), new_txt, txt_color, font=font)
        # whether add new blank img or not
        if count >= img_h // gap - 1 and idx + 1 < len(texts):
            txt_img_list.append(np.array(blank_img))
            blank_img, draw_txt = create_blank_img()
            count = 0
        count += 1
    txt_img_list.append(np.array(blank_img))
    if len(txt_img_list) == 1:
        blank_img = np.array(txt_img_list[0])
    else:
        blank_img = np.concatenate(txt_img_list, axis=1)
    return np.array(blank_img)


def base64_to_cv2(b64str):
    import base64
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.fromstring(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data


def draw_boxes(image, boxes, scores=None, drop_score=0.5):
    if scores is None:
        scores = [1] * len(boxes)
    for (box, score) in zip(boxes, scores):
        if score < drop_score:
            continue
        box = np.reshape(np.array(box), [-1, 1, 2]).astype(np.int64)
        image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)
    return image
