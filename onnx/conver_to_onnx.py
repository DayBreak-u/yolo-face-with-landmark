import sys
sys.path.insert(0,'..')
from nn import *
import torch
from utils.torch_utils import select_device
from hyp import  hyp
from torchstat import stat


point_num = hyp['point_num']

device = select_device('cpu')


net_type = "mbv3_large_75"

assert net_type in ['mbv3_small_1','mbv3_small_75','mbv3_large_1','mbv3_large_75',
                    "mbv3_large_75_light", "mbv3_large_1_light",'mbv3_small_75_light','mbv3_small_1_light',
                       ]

if net_type.startswith("mbv3_small_1"):
    backone = mobilenetv3_small()
elif net_type.startswith("mbv3_small_75"):
    backone = mobilenetv3_small( width_mult=0.75)
elif net_type.startswith("mbv3_large_1"):
    backone = mobilenetv3_large()
elif net_type.startswith("mbv3_large_75"):
    backone = mobilenetv3_large( width_mult=0.75)


if 'light' in net_type:
    net = DarknetWithShh(backone, hyp, light_head=True,onnx_export=True).to(device)
else:
    net = DarknetWithShh(backone, hyp,onnx_export=True).to(device)


# net.load_state_dict(torch.load(weights, map_location=device)['model'])
net.eval()


print(stat(net, (3, 320, 240)))

#
 ##################export###############
output_onnx = '{}.onnx'.format(net_type)
print("==> Exporting models to ONNX format at '{}'".format(output_onnx))
input_names = ["input0"]
output_names = ["s8", "s16" , "s32"]
inputs = torch.randn(1, 3, 320 , 320).to(device)
torch_out = torch.onnx._export(net, inputs, output_onnx, export_params=True, verbose=False,
                               input_names=input_names, output_names = output_names ,
               dynamic_axes={'input0':{2:'shape_h',3:'shape_w'}})


import  os
os.system("python -m   onnxsim   {} {} --input-shape 1,3,320,320".format(output_onnx,output_onnx))