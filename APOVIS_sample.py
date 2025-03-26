import argparse
import os
try:
    import ruamel_yaml as yaml
except ModuleNotFoundError:
    import ruamel.yaml as yaml
import numpy as np
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import sys
sys.path.append("ALBEF/")
import torch.backends.cudnn as cudnn
from scipy import ndimage
from PIL import  Image
from functools import partial
from ALBEF.models.vit import VisionTransformer
from ALBEF.models.xbert import BertConfig, BertModel
#from ALBEF.models.tokenization_bert import BertTokenizer
from transformers import AutoTokenizer, BertTokenizer

from ALBEF import utils
from ALBEF.dataset import create_dataset, create_sampler, create_loader
import pickle
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from segment_anything import sam_model_registry, SamPredictor
import copy
import math
import torch.optim as optim
from torch.autograd import Variable

class VL_Transformer_ITM(nn.Module):
    def __init__(self,
                 text_encoder=None,
                 config_bert='',
                 img_size=384
                 ):
        super().__init__()

        bert_config = BertConfig.from_json_file(config_bert)
        self.visual_encoder = VisionTransformer(
            img_size=img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)

        self.itm_head = nn.Linear(768, 2)

    def forward(self, image, text):
        image_embeds = self.visual_encoder(image)

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        output = self.text_encoder(text.input_ids,
                                   attention_mask=text.attention_mask,
                                   encoder_hidden_states=image_embeds,
                                   encoder_attention_mask=image_atts,
                                   return_dict=True,
                                   )

        vl_embeddings = output.last_hidden_state[:, 0, :]
        vl_output = self.itm_head(vl_embeddings)
        return vl_output

def net_normalized_act_map(total_act_map_obj):
    sum_act_map_obj = 0
    for i in range(len(total_act_map_obj)):
        req_act_map = total_act_map_obj[i]
        req_act_map = (req_act_map - req_act_map.min()) / (req_act_map.max() - req_act_map.min())
        sum_act_map_obj = req_act_map+sum_act_map_obj
    return sum_act_map_obj


def impaint_function(act_map_obj, img):
    net_img_mean = img.mean()

    act_map_obj = act_map_obj.detach().clone().squeeze().cuda()
    act_map_obj = (act_map_obj - act_map_obj.min()) / (act_map_obj.max() - act_map_obj.min())
    act_map_obj[act_map_obj < 0.5] = 0.0
    act_map_obj[act_map_obj > 0.5] = 1.0

    mask = act_map_obj
    inv_mask = mask.detach().clone()
    inv_mask[mask == 0] = 1
    inv_mask[mask == 1] = 0

    mask_img = mask * img
    impaint = mask_img.detach().clone()

    impaint[impaint != 0] = net_img_mean
    inv_mask_img = inv_mask * img

    net_img = inv_mask_img + impaint

    return net_img


def get_activation_map(output, model, image, text_input_mask, block_num, map_size, batch_index):
    loss = output[1].sum()
    image = image.unsqueeze(0)
    text_input_mask = text_input_mask.unsqueeze(0)

    model.zero_grad()
    loss.backward(retain_graph=True)

    with torch.no_grad():
        mask = text_input_mask.view(text_input_mask.size(0),1,-1,1,1)

        grads=model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.get_attn_gradients()
        cams=model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.get_attention_map()
        cams = cams[batch_index, :, :, 1:].reshape(image.size(0), 12, -1, map_size, map_size)
        cams = cams * mask
        grads = grads[batch_index, :, :, 1:].clamp(0).reshape(image.size(0), 12, -1, map_size, map_size) * mask

        gradcam = cams * grads
        gradcam = gradcam.mean(1)
    return gradcam[0, :, :, :].cpu().detach()


def find_best_bbox_and_center(act_map_obj):
    # Find the connected area with a pixel value of 1 in the activation diagram
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(act_map_obj, connectivity=8)

    # Initialize the best Bbox and best center point coordinates
    best_bbox = None
    best_center = None
    best_area = 0

    # Iterate through each connected area
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        center = centroids[label]

        # Calculate the aspect ratio of bounding boxes
        aspect_ratio = float(w) / h

        # Select the area with the largest area as the best Bbox coordinates
        if area > best_area and aspect_ratio < 2:
            best_area = area
            best_bbox = (x, y, x + w, y + h)
            best_center = (int(center[0]), int(center[1]))

    return best_bbox, best_center

def find_all_bbox_and_centers(act_map_obj):
    # Find the connected area with a pixel value of 1 in the activation diagram
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(act_map_obj, connectivity=8)

    results = []
    min_x=act_map_obj.shape[1]
    min_y=act_map_obj.shape[0]
    max_x=0
    max_y=0
    center_x, center_y = 0, 0
    num=0
   # Initialize the best Bbox and best center point coordinates
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        center = centroids[label]

        # Calculate the aspect ratio of bounding boxes
        aspect_ratio = float(w) / h

        # Select an area with an aspect ratio of less than 2
        if aspect_ratio < 2:
            bbox = (x, y, x + w, y + h)
            if x <= min_x:
                min_x = x
            if y <= min_y:
                min_y = y
            if x+w >= max_x:
                max_x = x+w
            if y+h >= max_y:
                max_y = y+h
            center = (int(center[0]), int(center[1]))
            center_x += int(center[0])
            center_y += int(center[1])
            num +=1
            results.append((bbox, center))
    maybe_bestbbox = [min_x,min_y,max_x,max_y]
    if num != 0:
        maybe_centerbest = [round(center_x/num, 2), round(center_y/num, 2)]
    else:
        maybe_centerbest = [0,0]

    return results, maybe_bestbbox, maybe_centerbest


#save results
def print_and_save(text, output_file):
    with open(output_file, 'a') as f:
        sys.stdout = f
        print(text)
        sys.stdout = sys.__stdout__

        
def generate_pseudo_bbox(model, tokenizer, data_loader, object_name_dict, args, block_num, map_size, device, predictor,outputfile):
    print('start_GPB!!\n')
    num_image_without_proposals = 0
    num_image = 0
    num_zeroscore_image = 0
    mpa_score = 0
    miou_score = 0
    seg_num = 0
    metric_logger = utils.MetricLogger(delimiter="  ")
    print_freq = 50

    tokenized_dict = {}
    for (k, v_list) in object_name_dict.items():
        tokenized_v_list = []
        for v in v_list:
            value_tmp = tokenizer._tokenize(v)
            value = ' '.join(value_tmp)
            tokenized_v_list.append(value)
        tokenized_dict[k] = tokenized_v_list

    for batch_i, (images, text, proposal_paths, true_bbox) in enumerate(metric_logger.log_every(data_loader, print_freq, '')):

        original_img = copy.deepcopy(images)

        objects_dict = {}  # key is the proposal_path
        objects = []
        ori_shape = []
        bbox1 = [np.array(tensor) for tensor in true_bbox[0]]
        bbox1 = np.stack(bbox1)
        real_bbox = np.transpose(bbox1)
        for (i, proposal_path) in enumerate(proposal_paths):#without use proposal_path
            wl = tokenizer._tokenize(text[i])
            tokenizeded_text = ' '.join(wl)
            tokenizeded_text = ' ' + tokenizeded_text + ' '
            objects_for_one = []
            for (k, v_list) in tokenized_dict.items():
                for v in v_list:
                    left_index = tokenizeded_text.find(' ' + v + ' ')
                    if left_index != -1:
                        space_count = tokenizeded_text[:(left_index + 1)].count(' ')
                        objects_for_one.append((k, v, space_count, space_count + len(v.strip().split(' '))))
            objects.append(objects_for_one)

        ########################################## Iterative Masking ##################################################
        total_iter = 3
        mask_dict = {}
        bbox_dict = {}
        box_cnt_thresh = 1
        for cnt in range(total_iter):
            image = images
            image = image.to(device, non_blocking=True)
            text_input = tokenizer(text, padding='longest', max_length=30, return_tensors="pt").to(device)
            output = model(image, text_input)

            impaint_img = []
            for i, img in enumerate(image):
                filename = proposal_paths[i].split('/')[-1]
                # print(f'{filename}\n')
                im_h, im_w = img.shape[1], img.shape[2]
                act_map = get_activation_map(output[i], model, img, text_input['attention_mask'][i], block_num, map_size, i) #action map

                list_bbox_act_map_obj = []
                for (original_obj_name, replaced_obj_name, obj_i_left, obj_i_right) in objects[i]:

                    file_object = filename.split('.')[0] + "_" + original_obj_name
                    act_map_obj = act_map[obj_i_left]

                    if obj_i_right - obj_i_left > 1:
                        for obj_i in range(obj_i_left + 1, obj_i_right):
                            act_map_obj += act_map[obj_i]
                    mask_act_map_obj = F.interpolate(act_map_obj.unsqueeze(0).unsqueeze(0), size=(im_h, im_w),
                                                     mode='bilinear').detach().clone()
                    bbox_act_map_obj = F.interpolate(act_map_obj.unsqueeze(0).unsqueeze(0),
                                                     size=(im_h, im_w)).detach().clone()

                    list_bbox_act_map_obj.append(bbox_act_map_obj)
                    if file_object not in mask_dict:

                        mask_act_map_obj = (mask_act_map_obj - mask_act_map_obj.min()) / (
                                    mask_act_map_obj.max() - mask_act_map_obj.min())

                        mask_act_map_obj_numpy = np.uint8(mask_act_map_obj.numpy().squeeze() * 255)
                        mask_act_map_obj_numpy = cv2.GaussianBlur(mask_act_map_obj_numpy, (5, 5), 0)
                        _, mask_act_map_obj_numpy = cv2.threshold(mask_act_map_obj_numpy, 0, 255,
                                                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        mask_act_map_obj_numpy[mask_act_map_obj_numpy == 255] = 1
                        mask_act_map_obj = torch.from_numpy(mask_act_map_obj_numpy).unsqueeze(0).unsqueeze(0).float()
                        mask_dict[file_object] = [mask_act_map_obj]
                        bbox_dict[file_object] = [(bbox_act_map_obj - bbox_act_map_obj.min()) / (
                                    bbox_act_map_obj.max() - bbox_act_map_obj.min())]
                    elif file_object in mask_dict:
                        mask_act_map_obj = (mask_act_map_obj - mask_act_map_obj.min()) / (
                                    mask_act_map_obj.max() - mask_act_map_obj.min())

                        mask_act_map_obj_numpy = np.uint8(mask_act_map_obj.numpy().squeeze() * 255)
                        mask_act_map_obj_numpy = cv2.GaussianBlur(mask_act_map_obj_numpy, (5, 5), 0)
                        _, mask_act_map_obj_numpy = cv2.threshold(mask_act_map_obj_numpy, 0, 255,
                                                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        mask_act_map_obj_numpy[mask_act_map_obj_numpy == 255] = 1
                        mask_act_map_obj = torch.from_numpy(mask_act_map_obj_numpy).unsqueeze(0).unsqueeze(0).float()
                        mask_dict[file_object].append(mask_act_map_obj)
                        if cnt < box_cnt_thresh:
                            bbox_dict[file_object].append((bbox_act_map_obj - bbox_act_map_obj.min()) / (
                                        bbox_act_map_obj.max() - bbox_act_map_obj.min()))

                net_act_map_obj = net_normalized_act_map(list_bbox_act_map_obj)
                if len(objects[i]) == 0:
                    impaint_img.append(img)
                else:
                    impaint_img.append(impaint_function(net_act_map_obj, img))
                impaint_img1 = torch.stack(impaint_img)
            images = torch.stack(impaint_img)

        for i, img in enumerate(image):
            filename = proposal_paths[i].split('/')[-1]
            with Image.open(proposal_paths[i]) as ori_img:
                width, height = ori_img.size
            im_h, im_w = height, width
            ########################################## Best Proposal Selection ##################################################
            root_proposal_path = './'+proposal_paths[i]
            sam_image = cv2.imread(root_proposal_path)

            sam_image = cv2.cvtColor(sam_image, cv2.COLOR_BGR2RGB)
            predictor.set_image(sam_image)
            x, y, w, h = real_bbox[i]
            true_bbox = [x, y, round(x + w, 2), round(y + h, 2)]
            input_box = np.array(true_bbox)
            true_masks, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )
            num_image += 1
            print_and_save("Processed " + str(num_image) + " images", outputfile)
            print_and_save('proposals_path: '+str(proposal_paths[i]), outputfile)
            sys.stdout = sys.__stdout__
            for (original_obj_name, replaced_obj_name, obj_i_left, obj_i_right) in objects[i]:
                file_object = filename.split('.')[0] + "_" + original_obj_name
                act_map_obj = sum(bbox_dict[file_object])
                act_map_obj = F.interpolate(act_map_obj, size=(im_h, im_w)).cpu().numpy()
                act_map_obj = act_map_obj.squeeze()

                act_map_obj = (act_map_obj - act_map_obj.min()) / (act_map_obj.max() - act_map_obj.min())
                act_map_obj = np.uint8(act_map_obj * 255)
                act_map_obj = cv2.GaussianBlur(act_map_obj, (5, 5), 0)
                _, act_map_obj = cv2.threshold(act_map_obj, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                act_map_obj[act_map_obj == 255] = 1


                results, maybe_bestbbox, maybe_centerbest = find_all_bbox_and_centers(act_map_obj)

                num_results = len(results)

                print_and_save('Prediction Category: '+str(original_obj_name)+str(replaced_obj_name),outputfile)
                print_and_save("Number of predicted points："+str(num_results), outputfile)
                center_list = []
                label_num = 0

                if num_results ==1:
                    for re_bbox, re_center in results:
                        center_list.append(re_center)
                        label_num +=1
                        print_and_save('Only  one bbox：'+ str(np.array(re_bbox)), outputfile)
                        print_and_save('Only one center：' + str(np.array(re_center)),outputfile)
                elif num_results > 1:
                    for re_bbox, re_center in results:
                        print_and_save('predicate bbox：' + str(np.array(re_bbox)), outputfile)
                        print_and_save('predicate center：'  + str(np.array(re_center)), outputfile)
                        if true_bbox[0]<= re_center[0] <= true_bbox[2] and true_bbox[1]<= re_center[1] <= true_bbox[3]:
                            center_list.append(list(re_center))
                            label_num +=1
                            print_and_save('NO'+ str(label_num) + 'bbox：'+ str(np.array(re_bbox)), outputfile)
                            print_and_save('NO'+str(label_num) + 'center：'+ str(np.array(re_center)), outputfile)
                if num_results == 0 or label_num == 0:
                    num_zeroscore_image += 1
                    break




def main(args, config):
    device = torch.device(args.device)
    cudnn.benchmark = True

    ########################################## Dataset ##########################################
    print("Creating dataset")
    datasets = [create_dataset('pseudolabel', config, args.root_directory, args.bbox_proposal_addr)]

    data_loader = create_loader(datasets, [None], batch_size=[config['batch_size']], num_workers=[4], is_trains=[True],
                                collate_fns=[None])[0]

    # tokenizer = AutoTokenizer.from_pretrained(args.text_encoder)
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    ########################################## Model Initialization ##########################################
    print("Creating model......")
    bert_config_path = 'ALBEF/configs/config_bert.json'
    model_path = args.model_path
    img_size = 256
    map_size = 16
    model = VL_Transformer_ITM(text_encoder='bert-base-uncased', config_bert=bert_config_path, img_size=img_size)
    model = model.to(device)

    ##load_sam model###
    sam_path = args.sam_model_path
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_path)
    sam.to(device="cuda")
    predictor = SamPredictor(sam)

    ########################################## Load the Model ##########################################
    print('load_model!!\n')
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    for key in list(state_dict.keys()):  # adjust different names in pretrained checkpoint
        if 'bert' in key:
            encoder_key = key.replace('bert.', '')
            state_dict[encoder_key] = state_dict[key]
            del state_dict[key]

    print("Start loading form the checkpoint......")
    msg = model.load_state_dict(state_dict, strict=False)
    assert len(msg.missing_keys) == 0

    model.eval()
    block_num = 8

    model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.save_attention = True
    print("Loading object name dictionary....")
    with open(args.object_dict, 'r') as fp:
        object_name_dict = json.load(fp)
    print("Start generating pseudo-mask annotation (box level + pixel level)...!!!")
    start_time = time.time()
    outputfile = args.outputfile
    generate_pseudo_bbox(model, tokenizer, data_loader, object_name_dict, args, block_num, map_size, device, predictor,
                         outputfile)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='ALBEF/configs/Pretrain.yaml')
    parser.add_argument('--model_path', default='examples/ALBEF_4M.pth')
    parser.add_argument('--sam_model_path', default='examples/sam_vit_h_4b8939.pth')#SAM SAM model selection
    parser.add_argument('--root_directory', default='datasets/')
    parser.add_argument('--output_dir', default='pseudo_label_output3/')
    parser.add_argument('--object_dict', default='examples/object_vocab90.json') #Set the dataset location and input image description
    parser.add_argument('--bbox_proposal_addr', default='datasets/')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--outputfile', default='datasets/show_image5.txt')  # The output results are saved in the txt file
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    #yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)

