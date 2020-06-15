from nn import  *
import torch
from utils.torch_utils import select_device
from utils.utils import non_max_suppression
import cv2
import numpy as  np
import glob
from hyp import  hyp




device = select_device('cpu')


net_type = "mbv3_large_1"
long_side = -1  # -1 mean origin shape
backone = None

assert net_type in ['mbv3_small_1', 'mbv3_small_75', 'mbv3_large_1', 'mbv3_large_75',
                   "mbv3_large_75_light", "mbv3_large_1_light", 'mbv3_small_75_light', 'mbv3_small_1_light',
                   ]



if net_type.startswith("mbv3_small_1"):
    backone = mobilenetv3_small()
elif net_type.startswith("mbv3_small_75"):
    backone = mobilenetv3_small( width_mult=0.75)
elif net_type.startswith("mbv3_large_1"):
    backone = mobilenetv3_large()
elif net_type.startswith("mbv3_large_75"):
    backone = mobilenetv3_large( width_mult=0.75)
elif net_type.startswith("mbv3_large_f"):
    backone = mobilenetv3_large_full()

if 'light' in net_type:
    net = DarknetWithShh(backone, hyp, light_head=True).to(device)
else:
    net = DarknetWithShh(backone, hyp).to(device)


point_num = hyp['point_num']
weights = "./weights/{}_last.pt".format(net_type)

net.load_state_dict(torch.load(weights, map_location=device)['model'])
net.eval()

dir = "./test_imgs/inputs/*.jpg"


imgs = glob.glob(dir)



for path in imgs:
    print(path)
    orig_image = cv2.imread(path)
    ori_h, ori_w, _ = orig_image.shape
    LONG_SIDE = long_side
    if long_side == -1:
        max_size = max(ori_w,ori_h)
        LONG_SIDE = max(32,max_size - max_size%32)

    if ori_h > ori_w:
        scale_h = LONG_SIDE / ori_h
        tar_w = int(ori_w * scale_h)
        tar_w = tar_w - tar_w % 32
        tar_w = max(32, tar_w)
        tar_h = LONG_SIDE


    else:
        scale_w = LONG_SIDE / ori_w
        tar_h = int(ori_h * scale_w)
        tar_h = tar_h - tar_h % 32
        tar_h = max(32, tar_h)
        tar_w = LONG_SIDE

    scale_w = tar_w * 1.0 / ori_w
    scale_h = tar_h * 1.0 / ori_h

    image = cv2.resize(orig_image, (tar_w, tar_h))



    image = image[...,::-1]
    image = image.astype(np.float64)
    # image = (image - hyp['mean']) / hyp['std']
    image /= 255.0
    image = np.transpose(image, [2, 0, 1])

    image = np.expand_dims(image, axis=0)

    image = torch.from_numpy(image)

    image = image.to(device).float()
    pred = net(image)[0]


    pred = non_max_suppression(pred,0.25, 0.35,
                                       multi_label=False, classes=0, agnostic= False,land=True ,point_num= point_num)
    try:
        det = pred[0].cpu().detach().numpy()
        orig_image = orig_image.astype(np.uint8)

        det[:,:4] = det[:,:4] / np.array([scale_w, scale_h] * 2)
        det[:,5:5+point_num*2] = det[:,5:5+point_num*2] / np.array([scale_w, scale_h] * point_num)
    except:
        det = []
    for b in det:

        text = "{:.4f}".format(b[4])
        b = list(map(int, b))

        cv2.rectangle(orig_image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(orig_image, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        # landms

        # print(b[11], b[12])
        # print(b[13], b[14])
        w , h = b[2] - b[0] , b[3] - b[1]
        # if w >64 or h >64 :
        #     for i in range(point_num):
        #         cv2.circle(orig_image, (b[5+i*2], b[5+i*2+1]), 1, (255, 255, 255), 2)
        cv2.circle(orig_image, (b[5], b[6]), 1, (0, 0, 255), 4)
        cv2.circle(orig_image, (b[7], b[8]), 1, (0, 255, 255), 4)
        cv2.circle(orig_image, (b[9], b[10]), 1, (255, 0, 255), 4)
        cv2.circle(orig_image, (b[11], b[12]), 1, (0, 255, 0), 4)
        cv2.circle(orig_image, (b[13], b[14]), 1, (255, 0, 0), 4)
    # save image

    cv2.imwrite(path.replace("inputs","outputs"), orig_image)