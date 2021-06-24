from pycocotools.mask import decode
import cv2
import numpy as np
import pickle
import json
import torch
import torch.nn.functional as F

a = pickle.load(open('chenhuilan.pkl', 'rb'))

j = json.load(open('testimgs/chenhuilan.json'))['images']

tot = len(a)

for i in range(tot):
    tmp = a[i]
    mask = tmp[1]

    tooth = mask[1]
    if len(tooth) != 0:
        oriqq = decode(tooth)

        oriqq = oriqq.max(axis=2) #h, w

        img = cv2.imread('testimgs/' + j[i]['file_name'])
        img0 = img[:, :, 0]
        img0 = torch.from_numpy(img0).cuda().unsqueeze(0).unsqueeze(0).to(dtype=torch.int32)

        bone = (img0 < 50).to(dtype=torch.int32)

        qq = oriqq.astype(np.float32)
        qq = torch.from_numpy(qq).cuda()
        qq = qq.unsqueeze(0).unsqueeze(0)
        qq_max = F.max_pool2d(qq, kernel_size=(7,7), padding=3, stride=1)

        q2 = (qq != qq_max).to(dtype=torch.int32)
        q2 = q2 * bone
        q2 = q2[0][0].cpu().numpy()
        q2 = q2.astype(np.uint8) * 255


        if (q2.max() != 0):
            print(i, q2.max())
        oriqq = np.stack([oriqq, oriqq, oriqq], axis=2) * 255
        oriqq[:, :, 0] = q2
        res = np.concatenate([img, oriqq], axis=1)

        cv2.imwrite('vis_bone/chenhuilan/%d.jpg'%i, res) 
