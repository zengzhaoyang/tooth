import mmcv
from mmdet.apis import init_detector#,  inference_detector
from mmcv.parallel import collate, scatter
from mmdet.datasets.pipelines import Compose
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import os
import sys
from mmdet.models import build_detector
import time
from tqdm import tqdm
from cal import cal

class LoadImage(object):

    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
        else:
            results['filename'] = None

        img = cv2.imread(results['img'])

        results['ori_filename'] = results['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results


class M(object):

    def __init__(self, model_name, model_path):

        self.model_name = model_name
        self.model_path = model_path

        self.model = init_detector(model_name, model_path, device='cuda:0')

        self.cfg = self.model.cfg
        self.device = next(self.model.parameters()).device

        self.test_pipeline = [LoadImage()] + self.cfg.data.test.pipeline[1:]
        self.test_pipeline = Compose(self.test_pipeline)

    def _inference(self, filename):

        ss = time.time()

        data = dict(img=filename)
        data = self.test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        data = scatter(data, [self.device])[0]

        with torch.no_grad():
            pred = self.model(return_loss=False, rescale=True, **data)


        #print(len(pred[0][0]), len(pred[0][1]), 'haha')
        #print(pred[0][0][0].shape, pred[0][0][1].shape)

        bbox = pred[0][0][1]
        area = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
        valid = area > 100
        bbox = bbox[valid]

        pred = pred[0][1]

        tot = len(pred[1])
        res = pred[1][0].astype(np.int32)

        for i in range(1, tot):
            res = res + pred[1][i].astype(np.int32)


        tot_bone = len(pred[0])
        res_bone = res * 0
        for i in range(0, tot_bone):
            res_bone = res_bone + pred[0][i].astype(np.int32)

        res = res > 0
        res = res.astype(np.uint8)

        res_bone = res_bone > 0
        res_bone = res_bone.astype(np.uint8)

        res = res * res_bone
        res = res * 255

        return res, bbox

    def _postprocess(self, data):

        return data


a = os.walk(sys.argv[3])
names = []
for b, c, d in a:
    for item in d:
        if item.endswith('.png'):
            names.append(b + '/' + item)


names = sorted(names, key=lambda x: int(x.split('/')[-1][:-4]))
model = M(sys.argv[1], sys.argv[2])


ii = 0
subfolder = sys.argv[3].split('/')[-2]
#os.system("mkdir -p vis/%s"%(subfolder))
segs = []
bboxs = []
for n in tqdm(names):
    res, bbox = model._inference(n)
    #cv2.imwrite('vis/%s/%d.png'%(subfolder, ii), res)
    segs.append(res)
    bboxs.append(bbox)
    ii += 1

print(len(segs), len(bboxs))
segs = np.stack(segs)
np.save('tmp_seg.npy', segs)
np.save('tmp_bbox.npy', bboxs)
print(segs.shape, segs.max())

#segs = np.load('tmp_seg.npy')
#bbox = np.load('tmp_bbox.npy', allow_pickle=True)

tmp = segs / 255
idx = tmp.sum(axis=(1,2)).argmax()

bbox = bboxs[idx]
print(bbox.shape)
print(bbox)

cnt = bbox.shape[0]
for i in range(cnt):
    xmin = int(bbox[i][0]+0.5)
    ymin = int(bbox[i][1]+0.5)
    xmax = int(bbox[i][2]+0.5)
    ymax = int(bbox[i][3]+0.5)
    seg = segs[:, ymin:ymax, xmin:xmax] / 255
    #seg = torch.from_numpy(seg).cuda()
    #seg = seg.unsqueeze(0).unsqueeze(0).float()

    #fseg = -seg
    #fseg_pool = F.max_pool3d(fseg, kernel_size=3, stride=1, padding=1)
    #seg_pool = -fseg_pool
    #edge = (seg != seg_pool).long() * 255
    #edge = edge.cpu().numpy()[0][0]
    #print((edge/255).sum())

    tmp, nowcnt = cal(seg)
    
    for j in range(1, nowcnt+1):
        seg = tmp == j
        seg = torch.from_numpy(seg).cuda()
        seg = seg.unsqueeze(0).unsqueeze(0).float()
        
        fsegs = -seg
        fsegs_pool = F.max_pool3d(fsegs, kernel_size=3, stride=1, padding=1)
        segs_pool = -fsegs_pool
        edge = (seg != segs_pool).long() *255
        
        edge = edge.cpu().numpy()[0][0]
        summ = (edge / 255).sum()
        if summ > 1000:
            print(i, j,(edge/255).sum())

#tot = edge.shape[0]
#for i in range(tot):
#    cv2.imwrite('vis/09733123/%d.png'%(i), edge[i])


