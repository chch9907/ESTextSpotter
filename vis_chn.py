import os, sys
import torch
import numpy as np

from models.ests import build_ests
from util.slconfig import SLConfig
from util.visualizer import COCOVisualizer
from util import box_ops
from PIL import Image
# import torchvision.transforms as T  
import datasets.transforms as T 
import cv2
import pickle
np.set_printoptions(precision=3, suppress = True)

with open('chn_cls_list.txt', 'rb') as fp:
    CTLABELS = pickle.load(fp)

def _decode_recognition(rec):
    s = ''
    for c in rec:
        c = int(c)
        if c < 5461:
            s += str(chr(CTLABELS[c]))
        elif c == 5462:
            s += u''
    return s
    
def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    args.device = 'cuda'
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors

model_config_path = "config/ESTS/ESTS_4scale_chn_finetune.py" # change the path of the model config file
model_checkpoint_path = "rects_checkpoint.pth" # change the path of the model checkpoint

args = SLConfig.fromfile(model_config_path) 
model, criterion, postprocessors = build_model_main(args)
checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.eval()
model.cuda()

transform = T.Compose([
    T.RandomResize([(1000, 1000)],max_size=1100),
    # T.Resize([1000, 1000]),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]
)
# image_dir = 'test_dataset/'
# image_dir = '/home/user/HKUra/workspace/AnyText/test_imgs_cn/'
# dir = os.listdir(image_dir)
# for idx, i in enumerate(dir):
#     image = Image.open(image_dir + i).convert('RGB')

root = '/home/user/HKUra/workspace/AnyText/test_imgs_cn/'
prefix = 'nan_style_'
suffix = '_raw.png'
ori_idx = 0
# , 'cha.png', 'quchenshi.png', 'shihegang.png', 'farm.png'
ori_file_list = [root + item + suffix for item in [
                                            'nan', 'taichao', 'tanzai', 
                                            'leshan', 'chuannong', 'huayujie',
                                            'cha', 'quchenshi', 'shihetian', 'farm']] # , 'farm.png',
                                            
                                            # prefix + 'nan' + suffix, prefix + 'taichao3' + suffix, prefix + 'tanzai3' + suffix, prefix + 'leshan4' + suffix, prefix + 'chuannong4' + suffix, prefix + 'huayujie3' + suffix]]
                                            # prefix + 'nan', prefix + 'taichao', prefix + 'tanzai', prefix + 'leshan', prefix + 'chuannong', prefix + 'huayujie']]  # style
# prefix2 = 'gen/text_'  #
prefix2 = 'render/'   
text_file_list = [root + prefix2 + item for item in [
                                            # 'nan2.png', 'taichao.png', 'tanzai_v.png', 'leshan_v.png', 'chuannong2.png', 'huayujie.png', 
                                            # 'cha.png', 'quchenshi.png', 'shihetian.png', 'farm.png'
                                            'nan_song.png', 'taichao_song.png', 'tanzai_song.png', 'leshan_song.png', 'chuannong_song.png', 'huayujie_song.png', 'cha_song.png', 'quchenshi_song.png', 'shihetian_song.png', 'farm_song.png'
                                            ]]
image_file_list = ori_file_list + text_file_list
preds_neck = []
# img_list = [Image.open(img_file).convert('RGB') for img_file in image_file_list 
#             if os.path.exists(img_file)]

img_list = []
for idx, image_file in enumerate(image_file_list):
    if not os.path.exists(image_file):
        print('missing:', image_file)
        continue
    image = Image.open(image_file).convert('RGB')
    img_list.append(image)
length = len(ori_file_list)
# img_list = img_list[:length] + img_list[length:] * length
# image_file_list = image_file_list[:length] + image_file_list[length:] * length

reshape_img_list = []
for idx, image in enumerate(img_list):
    # print(image.size)
    # if idx >= length:
    #     ori_shape = img_list[int(idx // length) - 1].size
    #     image = image.resize((ori_shape[0], ori_shape[1]))  # (200, 200)
    
    # ori_shape = img_list[ori_idx].size  # ori_idx
    # image = image.resize((ori_shape[0], ori_shape[1]))
    
    # image = image.resize((128, 128))
    # image.show()
    # reshape_img_list.append(np.array(image).ravel())
    
    # ## canny is bad
    # raw_img = cv2.resize(cv2.imread(image_file), (256, 256))
    # edge = cv2.Canny(raw_img, 100, 200)
    # image = Image.fromarray(np.uint8(edge)).convert('RGB')

    image, _ = transform(image, None)
    # image = transform(image)
    # print('img:', image.shape)  # 3, 1000, 1000
    with torch.no_grad():
        raw_output = model(image[None].cuda(), img_name=image_file_list[idx])
    output = postprocessors['bbox'](raw_output, torch.Tensor([[1.0, 1.0]]))[0]
    rec = [_decode_recognition(i) for i in output['rec']]
    thershold = 0.2 # set a thershold
    scores = output['scores']
    labels = output['labels']
    boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
    select_mask = scores > thershold
    recs = []
    for i,r in zip(select_mask,rec):
        if i:
            recs.append(r)
    vslzr = COCOVisualizer()
    pred_dict = {
        'boxes': boxes[select_mask],
        'size': torch.tensor([image.shape[1],image.shape[2]]),
        'box_label': recs,
        'image_id': idx,
        'beziers': output['beziers'][select_mask]
    }
    # if idx >= length:
    print(recs)
    # print(raw_output['enc_out'].shape)
    # preds_neck.append(raw_output['enc_out'].detach().reshape(1, -1))
    preds_neck.append(raw_output['outputs_class_neck'].detach().reshape(1, -1))
    # preds_neck.append(raw_output['interm_outputs']['pred_boxSes'].detach().reshape(1, -1))
    
    # vslzr.visualize(image, pred_dict, savedir='vis_fin_raw2')
    # if idx >= 12:
    #     assert False

ori_embs = preds_neck[:length] #
# ori_embs = preds_neck[ori_idx:ori_idx + 1]
text_embs = preds_neck[length:]
matrix = np.zeros((len(ori_embs), len(ori_embs)))
from torch.nn.functional import cosine_similarity
with torch.no_grad():
    for i, ori_emb in enumerate(ori_embs):
        for j, text_emb in enumerate(text_embs):  # [i * 6: (i + 1) * 6]
            cos = cosine_similarity(ori_emb.reshape(1, -1), text_emb.reshape(1, -1))
            matrix[i, j] = cos
    # print(np.max(matrix, axis=1, keepdims=True).shape)
    max_row = np.max(matrix, axis=1, keepdims=True)
    matrix = matrix - max_row  # row
    for i, argmax in enumerate(np.argmax(matrix, axis=1)):
        matrix[i, argmax] = max_row[i][0]
    # matrix = torch.from_numpy(matrix)
    # matrix = matrix.softmax(dim=1)
print(matrix)


## direct compute cosine similarity between image arrays is bad
# from scipy import spatial
# matrix = np.zeros((6, 6))
# for i, ori_img in enumerate(reshape_img_list[:6]):
#     for j, text_img in enumerate(reshape_img_list[6:]):
#         matrix[i, j] = 1 - spatial.distance.cosine(ori_img, text_img)
# print(matrix)
