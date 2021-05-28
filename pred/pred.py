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

        pred = pred[0][1]
        tot = len(pred[1])
        res = pred[1][0].astype(np.int32)

        for i in range(1, tot):
            res = res + pred[1][i].astype(np.int32)

        res = res > 0
        res = res.astype(np.uint8) * 255

        return res

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
for n in tqdm(names):
    res = model._inference(n)
    #cv2.imwrite('vis/%s/%d.png'%(subfolder, ii), res)
    segs.append(res)
    ii += 1

segs = np.stack(segs)
np.save('tmp.npy', segs)
print(segs.shape, segs.max())

#segs = np.load('tmp.npy')
tmp, nowcnt = cal(segs)

print(nowcnt)
for i in range(1, nowcnt+1):
    segs = tmp == i
    segs = torch.from_numpy(segs).cuda()
    segs = segs.unsqueeze(0).unsqueeze(0).float()
    
    fsegs = -segs
    fsegs_pool = F.max_pool3d(fsegs, kernel_size=3, stride=1, padding=1)
    segs_pool = -fsegs_pool
    edge = (segs != segs_pool).long() *255
    
    edge = edge.cpu().numpy()[0][0]
    print((edge/255).sum())

#tot = edge.shape[0]
#for i in range(tot):
#    cv2.imwrite('vis/09733123/%d.png'%(i), edge[i])


