# Copyright (c) 2020 FSMLP Authors. All Rights Reserved.

import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

import cv2
import copy
import numpy as np
import math
import time
import torch
import paddle.fluid as fluid

import predict.predict_utility as utility
logger = utility.initial_logger()

from src.utils.input_utility import get_image_file_list, check_and_read_gif
from src.postprocess.crnn_postprocess import CharacterOps

from ctcdecode import CTCBeamDecoder
from spellchecker import SpellChecker
from nltk.tokenize import word_tokenize

class TextRecognizer(object):
    def __init__(self, args):
        if args.use_pdserving is False:
            self.predictor, self.input_tensor, self.output_tensors =\
                utility.create_predictor(args, mode="rec")
            self.use_zero_copy_run = args.use_zero_copy_run
        self.rec_image_shape = [int(v) for v in args.rec_image_shape.split(",")]
        self.rec_render = args.rec_render
        self.character_type = args.rec_char_type
        self.rec_batch_num = args.rec_batch_num
        self.rec_algorithm = args.rec_algorithm
        self.rec_whitelist = args.rec_whitelist
        self.rec_blacklist = args.rec_blacklist
        self.text_len = args.max_text_length

        char_ops_params = {
            "character_type": args.rec_char_type,
            "character_dict_path": args.rec_char_dict_path,
            "use_space_char": args.use_space_char,
            "max_text_length": args.max_text_length
        }
        if self.rec_algorithm in ["CRNN", "Rosetta", "STAR-Net"]:
            char_ops_params['loss_type'] = 'ctc'
            self.loss_type = 'ctc'
        elif self.rec_algorithm == "RARE":
            char_ops_params['loss_type'] = 'attention'
            self.loss_type = 'attention'
        elif self.rec_algorithm == "SRN":
            char_ops_params['loss_type'] = 'srn'
            self.loss_type = 'srn'
        self.char_ops = CharacterOps(char_ops_params)
        self.len_chars = self.char_ops.len_characters()
        
        # If both blacklist and whitelist are provided, whitelist is only used
        if self.rec_whitelist=='' and self.rec_blacklist!='':
            self.mod_chars = np.arange(start=0, stop=self.len_chars+1, step=1)
            black_list = self.char_ops.encode(self.rec_blacklist)
            self.mod_chars = np.setdiff1d(self.mod_chars, black_list)
        elif self.rec_whitelist!='':
            white_list = self.char_ops.encode(self.rec_whitelist)
            self.mod_chars = np.append(white_list, [self.len_chars])
        elif self.rec_whitelist=='' and self.rec_blacklist=='':
            self.mod_chars = []
            
        self.use_beam_search = args.use_beam_search
        if self.use_beam_search:
            self.beam_width = args.beam_width
            self.beam_lm_dir = args.beam_lm_dir if args.beam_lm_dir != '' else None
            self.beam_alpha = args.beam_alpha
            self.beam_beta = args.beam_beta
            self.beam_cutoff_top = args.beam_cutoff_top
            self.beam_cutoff_prob = args.beam_cutoff_prob
            # self.labels = self.char_ops.decode(np.arange(0,self.len_chars,1)) + '_'
            self.labels = list(self.char_ops.decode(self.mod_chars) + '_')
            self.blank_id = len(self.mod_chars)

            self.decoder = CTCBeamDecoder(
                labels = self.labels,
                model_path=self.beam_lm_dir,
                alpha=self.beam_alpha,
                beta=self.beam_beta,
                cutoff_top_n=self.beam_cutoff_top,
                cutoff_prob=self.beam_cutoff_prob,
                beam_width=self.beam_width,
                num_processes=os.cpu_count(),
                blank_id=self.blank_id,
                log_probs_input=False
            )
        
        self.use_spell_check = args.use_spell_check
        if self.use_spell_check:
            self.spell_case_sensitive = args.spell_case_sensitive
            self.spell_language = "" if self.spell_case_sensitive else args.spell_language
            self.spell_tokenizer = word_tokenize if args.spell_tokenizer == 'NLTK' else None
            self.spell_word_freq = args.spell_word_freq if args.spell_word_freq !='' else None
            self.spell_text_corpus = args.spell_text_corpus

            self.spell = SpellChecker(
                language=self.spell_language,
                local_dictionary=self.spell_word_freq,
                tokenizer=self.spell_tokenizer,
                case_sensitive=self.spell_case_sensitive,
            )

            if self.spell_text_corpus != '':
                self.spell.word_frequency.load_text_file(self.spell_text_corpus)


    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape
        assert imgC == img.shape[2]
        if self.character_type == "ch":
            imgW = int((32 * max_wh_ratio))
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def resize_norm_img_srn(self, img, image_shape):
        imgC, imgH, imgW = image_shape

        img_black = np.zeros((imgH, imgW))
        im_hei = img.shape[0]
        im_wid = img.shape[1]

        if im_wid <= im_hei * 1:
            img_new = cv2.resize(img, (imgH * 1, imgH))
        elif im_wid <= im_hei * 2:
            img_new = cv2.resize(img, (imgH * 2, imgH))
        elif im_wid <= im_hei * 3:
            img_new = cv2.resize(img, (imgH * 3, imgH))
        else:
            img_new = cv2.resize(img, (imgW, imgH))

        img_np = np.asarray(img_new)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        img_black[:, 0:img_np.shape[1]] = img_np
        img_black = img_black[:, :, np.newaxis]

        row, col, c = img_black.shape
        c = 1

        return np.reshape(img_black, (c, row, col)).astype(np.float32)

    def srn_other_inputs(self, image_shape, num_heads, max_text_length,
                         char_num):

        imgC, imgH, imgW = image_shape
        feature_dim = int((imgH / 8) * (imgW / 8))

        encoder_word_pos = np.array(range(0, feature_dim)).reshape(
            (feature_dim, 1)).astype('int64')
        gsrm_word_pos = np.array(range(0, max_text_length)).reshape(
            (max_text_length, 1)).astype('int64')

        gsrm_attn_bias_data = np.ones((1, max_text_length, max_text_length))
        gsrm_slf_attn_bias1 = np.triu(gsrm_attn_bias_data, 1).reshape(
            [-1, 1, max_text_length, max_text_length])
        gsrm_slf_attn_bias1 = np.tile(
            gsrm_slf_attn_bias1,
            [1, num_heads, 1, 1]).astype('float32') * [-1e9]

        gsrm_slf_attn_bias2 = np.tril(gsrm_attn_bias_data, -1).reshape(
            [-1, 1, max_text_length, max_text_length])
        gsrm_slf_attn_bias2 = np.tile(
            gsrm_slf_attn_bias2,
            [1, num_heads, 1, 1]).astype('float32') * [-1e9]

        encoder_word_pos = encoder_word_pos[np.newaxis, :]
        gsrm_word_pos = gsrm_word_pos[np.newaxis, :]

        return [
            encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1,
            gsrm_slf_attn_bias2
        ]

    def process_image_srn(self,
                          img,
                          image_shape,
                          num_heads,
                          max_text_length,
                          char_ops=None):
        norm_img = self.resize_norm_img_srn(img, image_shape)
        norm_img = norm_img[np.newaxis, :]
        char_num = char_ops.get_char_num()

        [encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1, gsrm_slf_attn_bias2] = \
            self.srn_other_inputs(image_shape, num_heads, max_text_length, char_num)

        gsrm_slf_attn_bias1 = gsrm_slf_attn_bias1.astype(np.float32)
        gsrm_slf_attn_bias2 = gsrm_slf_attn_bias2.astype(np.float32)

        return (norm_img, encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1,
                gsrm_slf_attn_bias2)

    def __call__(self, img_list):
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))

        #rec_res = []
        rec_res = [['', 0.0]] * img_num
        batch_num = self.rec_batch_num
        predict_time = 0
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                # h, w = img_list[ino].shape[0:2]
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                if self.loss_type != "srn":
                    norm_img = self.resize_norm_img(img_list[indices[ino]],
                                                    max_wh_ratio)
                    norm_img = norm_img[np.newaxis, :]
                    norm_img_batch.append(norm_img)
                else:
                    norm_img = self.process_image_srn(img_list[indices[ino]],
                                                      self.rec_image_shape, 8,
                                                      25, self.char_ops)
                    encoder_word_pos_list = []
                    gsrm_word_pos_list = []
                    gsrm_slf_attn_bias1_list = []
                    gsrm_slf_attn_bias2_list = []
                    encoder_word_pos_list.append(norm_img[1])
                    gsrm_word_pos_list.append(norm_img[2])
                    gsrm_slf_attn_bias1_list.append(norm_img[3])
                    gsrm_slf_attn_bias2_list.append(norm_img[4])
                    norm_img_batch.append(norm_img[0])

            norm_img_batch = np.concatenate(norm_img_batch, axis=0)
            norm_img_batch = norm_img_batch.copy()

            if self.loss_type == "srn":
                starttime = time.time()
                encoder_word_pos_list = np.concatenate(encoder_word_pos_list)
                gsrm_word_pos_list = np.concatenate(gsrm_word_pos_list)
                gsrm_slf_attn_bias1_list = np.concatenate(
                    gsrm_slf_attn_bias1_list)
                gsrm_slf_attn_bias2_list = np.concatenate(
                    gsrm_slf_attn_bias2_list)
                starttime = time.time()

                norm_img_batch = fluid.core.PaddleTensor(norm_img_batch)
                encoder_word_pos_list = fluid.core.PaddleTensor(
                    encoder_word_pos_list)
                gsrm_word_pos_list = fluid.core.PaddleTensor(gsrm_word_pos_list)
                gsrm_slf_attn_bias1_list = fluid.core.PaddleTensor(
                    gsrm_slf_attn_bias1_list)
                gsrm_slf_attn_bias2_list = fluid.core.PaddleTensor(
                    gsrm_slf_attn_bias2_list)

                inputs = [
                    norm_img_batch, encoder_word_pos_list,
                    gsrm_slf_attn_bias1_list, gsrm_slf_attn_bias2_list,
                    gsrm_word_pos_list
                ]

                self.predictor.run(inputs)
            else:
                starttime = time.time()
                if self.use_zero_copy_run:
                    self.input_tensor.copy_from_cpu(norm_img_batch)
                    self.predictor.zero_copy_run()
                else:
                    norm_img_batch = fluid.core.PaddleTensor(norm_img_batch)
                    self.predictor.run([norm_img_batch])

            if len(self.mod_chars)!=0:
                mod_onehot = np.zeros((self.len_chars + 1))
                mod_onehot[self.mod_chars] = 1

            if self.loss_type == "ctc":
                rec_idx_batch = self.output_tensors[0].copy_to_cpu()
                rec_idx_lod = self.output_tensors[0].lod()[0]
                predict_batch = self.output_tensors[1].copy_to_cpu()
                predict_lod = self.output_tensors[1].lod()[0]

                if len(self.mod_chars)!=0:
                    predict_batch = np.multiply(predict_batch, mod_onehot) #* Implemented blacklist and whitelist here!

                for rno in range(len(rec_idx_lod) - 1):

                    beg = predict_lod[rno]
                    end = predict_lod[rno + 1]
                    probs = predict_batch[beg:end, :]
                    ind = np.argmax(probs, axis=1)
                    valid_ind = range(ind.shape[0])
                    preds_text = self.char_ops.decode(ind[valid_ind], is_remove_duplicate=True)
                    if len(valid_ind) == 0:
                        continue
                    score = np.mean(probs[valid_ind, ind[valid_ind]])
                    

                    # use_spell_check results are the final results if both beam search and spell check is true!
                    if self.use_beam_search:
                        mod_probs = probs[:,self.mod_chars]
                        mod_probs = torch.Tensor(mod_probs).unsqueeze(0)
                        beams,scores,_,out_lens = self.decoder.decode(mod_probs)
                        res_beam = beams[0][0][:out_lens[0][0]]
                        res_list = [self.mod_chars[i] for i in res_beam]
                        res_text = self.char_ops.decode(res_list)
                        score_beam = 1/np.exp(scores[0][0])
                        if preds_text != res_text:
                            print(f'original: {preds_text} || beam_search_corrected: {res_text}')
                        rec_res[indices[beg_img_no + rno]] = [res_text, score_beam]

                        if self.use_spell_check:
                            corrected = self.spell.correction(res_text)
                            if preds_text != corrected:
                                print(f'original: {preds_text} || spell_check_corrected: {corrected}')
                            rec_res[indices[beg_img_no + rno]] = [corrected, score_beam]
                    
                    elif self.use_spell_check:
                        corrected = self.spell.correction(preds_text)
                        if preds_text != corrected:
                            print(f'original: {preds_text} || spell_check_corrected: {corrected}')
                        rec_res[indices[beg_img_no + rno]] = [corrected, score]
                        
                    else:
                        rec_res[indices[beg_img_no + rno]] = [preds_text, score]
                            

            elif self.loss_type == 'srn':
                rec_idx_batch = self.output_tensors[0].copy_to_cpu()
                probs = self.output_tensors[1].copy_to_cpu()

                # TODO: implement whitelist and blacklist for srn loss

                char_num = self.char_ops.get_char_num()
                preds = rec_idx_batch.reshape(-1)
                elapse = time.time() - starttime
                predict_time += elapse
                total_preds = preds.copy()
                for ino in range(int(len(rec_idx_batch) / self.text_len)):
                    preds = total_preds[ino * self.text_len:(ino + 1) *
                                        self.text_len]
                    ind = np.argmax(probs, axis=1)
                    valid_ind = np.where(preds != int(char_num - 1))[0]
                    if len(valid_ind) == 0:
                        continue
                    score = np.mean(probs[valid_ind, ind[valid_ind]])
                    preds = preds[:valid_ind[-1] + 1]
                    preds_text = self.char_ops.decode(preds)

                    rec_res[indices[beg_img_no + ino]] = [preds_text, score]
            else:
                rec_idx_batch = self.output_tensors[0].copy_to_cpu()
                predict_batch = self.output_tensors[1].copy_to_cpu()

                # TODO: implement whitelist and blacklist for srn loss

                elapse = time.time() - starttime
                predict_time += elapse
                for rno in range(len(rec_idx_batch)):
                    end_pos = np.where(rec_idx_batch[rno, :] == 1)[0]
                    if len(end_pos) <= 1:
                        preds = rec_idx_batch[rno, 1:]
                        score = np.mean(predict_batch[rno, 1:])
                    else:
                        preds = rec_idx_batch[rno, 1:end_pos[1]]
                        score = np.mean(predict_batch[rno, 1:end_pos[1]])
                    preds_text = self.char_ops.decode(preds)
                    # rec_res.append([preds_text, score])
                    rec_res[indices[beg_img_no + rno]] = [preds_text, score]

        # *TODO: COMPLETE (if self.rec_render:)

        return rec_res, predict_time


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    text_recognizer = TextRecognizer(args)
    valid_image_file_list = []
    img_list = []
    for image_file in image_file_list:
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        valid_image_file_list.append(image_file)
        img_list.append(img)

    try:
        rec_res, predict_time = text_recognizer(img_list)
    except Exception as e:
        print(e)
        logger.info(
            "ERROR!!!! \n"
            "Please read the FAQ: https://github.com/PaddlePaddle/PaddleOCR#faq \n"
            "If your model has tps module:  "
            "TPS does not support variable shape.\n"
            "Please set --rec_image_shape='3,32,100' and --rec_char_type='en' ")
        exit()
    for ino in range(len(img_list)):
        print("Predicts of %s:%s" % (valid_image_file_list[ino], rec_res[ino]))
    print("Total predict time for %d images:%.3f" %
          (len(img_list), predict_time))


if __name__ == "__main__":
    main(utility.parse_args())


