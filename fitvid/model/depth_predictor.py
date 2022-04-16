import torch
import torch.nn as nn

from fitvid.utils.depth_utils import mse_loss


class DepthPredictor(nn.Module):

    def __init__(self, depth_model_type, pretrained_weight_path, input_size, freeze_model=True):
        super().__init__()
        self.depth_model_type = depth_model_type
        self.pretrained_weight_path = pretrained_weight_path
        self.input_size = input_size
        self.freeze_model = freeze_model
        self.depth_head = self.load_pretrained_depth_model()
        self.depth_loss = mse_loss

    def load_pretrained_depth_model(self):
        # assert FLAGS.pretrained_depth_objective, 'Only pretrained depth objective available right now'

        from midas.dpt_depth import DPTDepthModel
        from midas.midas_net import MidasNet
        from midas.midas_net_custom import MidasNet_small

        depth_model_type = self.depth_model_type

        if self.pretrained_weight_path:
            model_location = self.pretrained_weight_path
        else:
            model_location = DEFAULT_WEIGHT_LOCATIONS[depth_model_type]
        print(f'Load depth model from {model_location}')
        if depth_model_type == 'dpt':
            depth_model = DPTDepthModel(
                path=model_location,
                backbone="vitb_rn50_384",
                non_negative=True,
            )
        elif depth_model_type == 'mn':
            depth_model = MidasNet(model_location, non_negative=True)
        elif depth_model_type == 'mns':
            depth_model = MidasNet_small(model_location, features=64, backbone="efficientnet_lite3",
                                         exportable=True,
                                         non_negative=False, blocks={'expand': True})
        # Setting the memory format to channels last saves around 600MB VRAM but costs computation time
        # depth_model = depth_model.to(memory_format=torch.channels_last)
        # depth_model = depth_model.half()
        print(f'Loaded depth weights from {self.pretrained_weight_path}!')
        print(f'Freeze model weights is {self.freeze_model}!')

        if self.freeze_model:
            for param in depth_model.parameters():
                param.requires_grad = False
        depth_model.eval()

        self.register_buffer('depth_head_mean', torch.Tensor([0.485, 0.456, 0.406])[..., None, None])
        self.register_buffer('depth_head_std', torch.Tensor([0.229, 0.224, 0.225])[..., None, None])
        return depth_model

    def forward(self, pred_frame, time_axis=True):
        if time_axis:
            shape = pred_frame.shape
            try:
                pred_frame = pred_frame.view(
                    (shape[0] * shape[1],) + shape[2:])  # collapse batch*time dims [b0t0, b0t1, b0t2... b1t0, b1t1, b1t2...]
            except Exception as e:
                # if the dimensions span across subspaces, need to use reshape
                pred_frame = pred_frame.reshape(
                    (shape[0] * shape[1],) + shape[2:])  # collapse batch*time dims [b0t0, b0t1, b0t2... b1t0, b1t1, b1t2...]
        pred_frame_norm = (pred_frame - self.depth_head_mean) / self.depth_head_std # normalize as done for pretrained MiDaS

        if self.input_size % pred_frame_norm.shape[-1] != 0:
            raise ValueError('depth model and frame pred size mismatch!')
        if self.input_size != pred_frame.shape[-1]:
            pred_frame_norm = torch.nn.Upsample(scale_factor=self.input_size //pred_frame_norm.shape[-1])(pred_frame_norm)
        depth_pred = self.depth_head(pred_frame_norm)[:, None].float()
        # normalize to [0, 1]
        if self.input_size != pred_frame.shape[-1]:
            depth_pred = torch.nn.functional.interpolate(depth_pred, size=pred_frame.shape[-2:])
        if time_axis:
            depth_pred = depth_pred.view((shape[0], shape[1],) + tuple(depth_pred.shape[1:]))
        else:
            depth_pred = depth_pred
        depth_pred = torch.clamp(depth_pred, min=0, max=1.0)
        return depth_pred

