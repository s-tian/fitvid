import numpy as np
import torch.nn as nn
import torch


class SEBlock(nn.Module):
    """Applies Squeeze-and-Excitation."""
    def __init__(self, channel):
        super(SEBlock, self).__init__()
        hidden_size = max(channel // 16, 4)

        self.reduce = nn.Linear(in_features=channel, out_features=hidden_size)
        self.act = nn.ReLU()
        self.expand = nn.Linear(in_features=hidden_size, out_features=channel)

    def forward(self, x): # (10, 3, 64, 64)
        y = x.mean(axis=(2, 3)) # (10, 3)
        y = self.reduce(y) # (10, 4)
        y = self.act(y)
        y = self.expand(y)
        y = y.view(y.shape + (1, 1))
        return torch.sigmoid(y) * x


class EncoderBlock(nn.Module):
    """NVAE ResNet block."""

    def __init__(self, in_channels, out_channels, downsample):
        super(EncoderBlock, self).__init__()

        stride = (2, 2) if downsample else (1, 1)

        self.Conv_0 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=stride, bias=False, padding=1) # pad = (kernel-1)//2
        self.Conv_1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), bias=False, padding=1)
        self.BatchNorm_0 = nn.BatchNorm2d(num_features=in_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.BatchNorm_1 = nn.BatchNorm2d(num_features=out_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.act1 = nn.SiLU()
        self.act2 = nn.SiLU()
        self.act3 = nn.SiLU()
        self.SEBlock_0 = SEBlock(out_channels)

        if (in_channels != out_channels) or downsample:
            self.reshape_residual = True
            self.conv_proj = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=stride, bias=False, padding=0)
            self.norm_proj = nn.BatchNorm2d(num_features=out_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        else:
            self.reshape_residual = False
        self.act4 = nn.SiLU()

    def forward(self, x):
        residual = x
        y = self.BatchNorm_0(x)
        y = self.act1(y)
        y = self.Conv_0(y)
        y = self.BatchNorm_1(y)
        y = self.act2(y)
        y = self.Conv_1(y)
        y = self.SEBlock_0(y)

        if self.reshape_residual:
            residual = self.conv_proj(residual)
            residual = self.norm_proj(residual)

        return self.act4(residual + y)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand, upsample):
        super(DecoderBlock, self).__init__()
        self.upsample = upsample
        if self.upsample:
            self.upsample_layer = nn.UpsamplingNearest2d(scale_factor=2)
        self.BatchNorm_0 = nn.BatchNorm2d(num_features=in_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.Conv_0 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels*expand, kernel_size=(1, 1), bias=False, padding=0) # pad = (kernel-1)//2
        self.BatchNorm_1 = nn.BatchNorm2d(num_features=out_channels*expand, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.act1 = nn.SiLU()
        self.Conv_1 = nn.Conv2d(in_channels=out_channels*expand, out_channels=out_channels*expand, kernel_size=(5, 5), bias=False, padding=2) # pad = (kernel-1)//2
        self.BatchNorm_2 = nn.BatchNorm2d(num_features=out_channels*expand, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.act2 = nn.SiLU()
        self.Conv_2 = nn.Conv2d(in_channels=out_channels*expand, out_channels=out_channels, kernel_size=(1, 1), bias=False, padding=0) # pad = (kernel-1)//2
        self.BatchNorm_3 = nn.BatchNorm2d(num_features=out_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        param = self.BatchNorm_3.state_dict()['weight']
        with torch.no_grad():
            param.copy_(torch.zeros_like(param))
        # for the 3 lines above, original code is: y = self.norm(scale_init=nn.initializers.zeros)(y)
        self.SEBlock_0 = SEBlock(out_channels)

        if in_channels != out_channels:
            self.reshape_residual = True
            self.conv_proj = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), bias=False, padding=0)
            self.norm_proj = nn.BatchNorm2d(num_features=out_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        else:
            self.reshape_residual = False
        self.act3 = nn.SiLU()

    def forward(self, x):
        if self.upsample:
            x = self.upsample_layer(x)

        residual = x
        y = self.BatchNorm_0(x)
        y = self.Conv_0(y)
        y = self.BatchNorm_1(y)
        y = self.act1(y)
        y = self.Conv_1(y)
        y = self.BatchNorm_2(y)
        y = self.act2(y)
        y = self.Conv_2(y)
        y = self.BatchNorm_3(y)
        y = self.SEBlock_0(y)

        if self.reshape_residual:
            residual = self.conv_proj(residual)
            residual = self.norm_proj(residual)

        return self.act3(residual + y)


class ModularEncoder(nn.Module):
    """Modular Encoder."""
    def __init__(self, stage_sizes, output_size, num_base_filters, num_input_channels=3):
        super(ModularEncoder, self).__init__()
        self.stages = nn.ModuleList()
        prev_num_filters = num_input_channels
        num_filters = None
        count = 0
        for i, block_size in enumerate(stage_sizes):
            blocks = nn.ModuleList()
            for j in range(block_size):
                num_filters = num_base_filters * 2 ** i
                downsample = True if i > 0 and j == 0 else False
                block = EncoderBlock(in_channels=prev_num_filters, out_channels=num_filters, downsample=downsample)
                setattr(self, f'EncoderBlock_{count}', block)
                count += 1
                prev_num_filters = num_filters
                blocks.append(block)
            self.stages.append(blocks)

        self.Dense_0 = nn.Linear(in_features=num_filters, out_features=output_size)

    def forward(self, x):
        skips = {}
        for i, stage in enumerate(self.stages):
            for j, block in enumerate(stage):
                x = block(x)
                skips[(i, j)] = x
        x = x.mean(axis=(2, 3))
        x = self.Dense_0(x)

        return x, skips


class ModularDecoder(nn.Module):
    def __init__(self, first_block_shape, input_size, stage_sizes, num_base_filters, skip_type, expand, num_output_channels=3):
        super(ModularDecoder, self).__init__()
        self.skip_type = skip_type
        self.stage_sizes = stage_sizes
        self.first_block_shape = first_block_shape

        first_block_size = np.prod(np.array(first_block_shape))
        self.Dense_0 = nn.Linear(in_features=input_size, out_features=first_block_size)

        self.stages = nn.ModuleList()
        prev_num_filters = self.first_block_shape[-3]
        count = 0
        for i, block_size in enumerate(reversed(stage_sizes)):
            blocks = nn.ModuleList()
            for j in range(block_size):
                num_filters = num_base_filters * 2 ** (len(stage_sizes) - i - 1)
                upsample = True if i > 0 and j == 0 else False
                block = DecoderBlock(in_channels=prev_num_filters, out_channels=num_filters, expand=expand, upsample=upsample)
                setattr(self, f'DecoderBlock_{count}', block)
                blocks.append(block)
                count += 1
                prev_num_filters = num_filters
            self.stages.append(blocks)

        self.Conv_0 = nn.Conv2d(in_channels=prev_num_filters, out_channels=num_output_channels, kernel_size=(3, 3), bias=False, padding=1) # pad = (kernel-1)//2

    def forward(self, x, skips, has_time_dim=True):
        if has_time_dim:
            shape = x.shape
            x = x.view((shape[0]*shape[1],) + shape[2:]) # collapse batch*time dims [b0t0, b0t1, b0t2... b1t0, b1t1, b1t2...]
        x = self.Dense_0(x)
        x = x.view((x.shape[0],) + tuple(self.first_block_shape))

        for i, stage in enumerate(self.stages):
            for j, block in enumerate(stage):
                x = block(x)
                if self.skip_type == 'residual':
                    skip = skips[(len(self.stage_sizes) - i - 1, len(stage) - j - 1)]
                    repeat = int(x.shape[0] / skip.shape[0])
                    skip = torch.repeat_interleave(skip, repeat, dim=0)
                    x = x + skip
                elif self.skip_type == 'concat':
                    skip = skips[(len(self.stage_sizes) - i - 1, len(stage) - j - 1)]
                    repeat = int(x.shape[0] / skip.shape[0])
                    skip = torch.repeat_interleave(skip, repeat, dim=0)
                    x = torch.concatenate([x, skip], axis=-1)
                elif self.skip_type in ['no_skip', 'pixel_residual_ff', 'pixel_residual_pf']:
                    pass
                elif self.skip_type is not None:
                    raise Exception('Unknown Skip Type.')
        x = self.Conv_0(x)
        x = torch.sigmoid(x)
        if has_time_dim: # add time dim back
            return x.view((shape[0], shape[1],) + tuple(x.shape[1:]))
        else:
            return x # for VAE