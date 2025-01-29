import torch
import argparse
import os
import onnx
from onnxsim import simplify
from lib.models.pose_resnet import get_pose_net
# from lib.models.pose_regnet import get_pose_net
from lib.core.config import config
from lib.core.config import update_config

#############################################################
# Comment out line 11 of lib/models/__init__.py and run it. #
#############################################################


class FlattenWrapper(torch.nn.Module):
    def __init__(self, model):
        super(FlattenWrapper, self).__init__()
        self.model = model
        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        x = self.model(x)
        x = self.flatten(x) 
        return x

parser = argparse.ArgumentParser(
    description='exporting torch model to onnx')

parser.add_argument("-v", "--version", default=11, type=int)

# exp file
# parser.add_argument('--target_path', default="/home/jhee/p41/datasets/room_mirror_data/(OMS)_person_class", type=str, help='path of video')
parser.add_argument('--save_path', default="exported_onnx", type=str, help='save folder')
parser.add_argument('--onnx_name', default="gv60_seat_kpt_resnet50_nkpt6", type=str, help='name of onnx file')
parser.add_argument('--cfg',
                    help='experiment configure file name',
                    default="experiments/coco/resnet50_test/gv60_custom_320x320_wo_flip_norm_resize_all_vis_top_seat.yaml",
                    # default="experiments/coco/regnet_test/gv60_custom_320x320_wo_flip_norm_resize_all_vis_nkpt6.yaml",
                    )
parser.add_argument("--checkpoint", 
                    type=str, 
                    default="./output/coco/pose_resnet_50/gv60_custom_320x320_wo_flip_norm_resize_all_vis_top_seat/final_state.pth.tar",
                    # default="./output/coco/pose_regnet_400m/gv60_custom_320x320_wo_flip_norm_resize_all_vis_nkpt6/final_state.pth.tar",
                    help="checkpoint path")

args = parser.parse_args()

update_config(args.cfg)

os.makedirs(args.save_path, exist_ok=True)

# load the model
model = get_pose_net(config, is_train=False)

model.eval()

# load the checkpoint
ckpt = torch.load(args.checkpoint, map_location='cpu')

state_dict = model.state_dict()
for k1, k2 in zip(state_dict.keys(), ckpt.keys()):
    state_dict[k1] = ckpt[k2]
model.load_state_dict(state_dict)


wrapperd_model = FlattenWrapper(model)

input_size=(320,320)
# input_size=(480,480)
input_data = torch.randn(1, 1, input_size[0], input_size[1])

# Convert the PyTorch model to an ONNX model

onnx_model_path = f"{args.onnx_name}.onnx"
torch.onnx.export(
    # model, 
    wrapperd_model,
    input_data, 
    os.path.join(args.save_path, onnx_model_path),
    opset_version=args.version, 
    input_names = ['input'],   # the model's input names
    output_names = ['output']
    )

print('Converting is done!!')

onnx_model = onnx.load(os.path.join(args.save_path, onnx_model_path))
model_simp, check = simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, os.path.join(args.save_path, "sim_"+onnx_model_path))
print('Simplified model is saved!!')