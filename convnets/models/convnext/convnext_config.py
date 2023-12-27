from dataclasses import dataclass, field
from typing      import Optional, List

import torch


@dataclass
class DepthwiseSeparableConv2dConfig:
    in_channels : int
    out_channels: int
    kernel_size : int
    stride      : int = 1
    padding     : int = 0
    dilation    : int = 1
    bias        : int = True
    padding_mode: int = 'zeros'
    device      : Optional[torch.device] = None
    dtype       : Optional[torch.dtype]  = None


@dataclass
class Conv2dConfig:
    in_channels : int
    out_channels: int
    kernel_size : int
    stride      : int  = 1
    padding     : int  = 0
    dilation    : int  = 1
    groups      : int  = 1
    bias        : bool = True
    padding_mode: str  = 'zeros'
    device      : Optional[torch.device] = None
    dtype       : Optional[torch.dtype]  = None


@dataclass
class LayerNormConfig:
    normalized_shape  : int
    eps               : float = 1e-5
    elementwise_affine: bool  = True


@dataclass
class ConvNeXTStemConfig:
    in_channels : int
    out_channels: int
    H           : int
    W           : int

    H_scaled    : int = field(init = False)
    W_scaled    : int = field(init = False)

    conv_config      : Conv2dConfig    = field(init = False)
    layer_norm_config: LayerNormConfig = field(init = False)

    def __post_init__(self):
        in_channels  = self.in_channels
        out_channels = self.out_channels
        kernel_size  = 4
        stride       = 4
        self.conv_config = Conv2dConfig(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
        )

        scale_factor = 4
        self.H_scaled = self.H // scale_factor
        self.W_scaled = self.W // scale_factor
        self.layer_norm_config = LayerNormConfig([out_channels, self.H_scaled, self.W_scaled])




@dataclass
class ConvNeXTBlockConfig:
    # In conv
    in_conv_in_channels: int
    in_conv_stride     : int

    # Mid conv
    mid_conv_out_channels: int
    mid_conv_stride      : int

    # Out conv
    out_conv_out_channels: int

    # Image dimension
    H: int
    W: int

    in_conv_config   : DepthwiseSeparableConv2dConfig = field(init = False)
    layer_norm_config: LayerNormConfig                = field(init = False)
    mid_conv_config  : Conv2dConfig                   = field(init = False)
    out_conv_config  : Conv2dConfig                   = field(init = False)

    def __post_init__(self):
        in_conv_kernel_size  = 7
        mid_conv_kernel_size = 1
        out_conv_kernel_size = 1

        in_conv_padding  = (in_conv_kernel_size  - 1) // 2
        mid_conv_padding = (mid_conv_kernel_size - 1) // 2
        out_conv_padding = (out_conv_kernel_size - 1) // 2

        self.in_conv_config = DepthwiseSeparableConv2dConfig(in_channels  = self.in_conv_in_channels,
                                                             out_channels = self.in_conv_in_channels,
                                                             kernel_size  = in_conv_kernel_size,
                                                             padding      = in_conv_padding,)    # ...Keep the spatial dimension unchanged

        self.mid_conv_config = Conv2dConfig(in_channels  = self.in_conv_in_channels,
                                            out_channels = self.mid_conv_out_channels,
                                            kernel_size  = mid_conv_kernel_size,
                                            padding      = mid_conv_padding,)

        self.out_conv_config = Conv2dConfig(in_channels  = self.mid_conv_out_channels,
                                            out_channels = self.out_conv_out_channels,
                                            kernel_size  = out_conv_kernel_size,
                                            padding      = out_conv_padding,)

        H = self.H
        W = self.W
        self.layer_norm_config = LayerNormConfig([self.in_conv_in_channels, H, W])


@dataclass
class ConvNeXTStageConfig:
    stage_in_channels     : int
    stage_out_channels    : int
    num_blocks            : int
    mid_conv_out_channels : int
    in_conv_stride        : int
    mid_conv_stride       : int

    H: int
    W: int

    block_config_list: List[ConvNeXTBlockConfig] = field(init = False)

    def __post_init__(self):
        self.block_config_list = [
            ConvNeXTBlockConfig(
                # First block uses stage_in_channels and rest uses prev stage_out_channels...
                in_conv_in_channels  = self.stage_in_channels if block_idx == 0 else self.stage_out_channels,

                # First block uses in_conv_stride and rest uses 1...
                in_conv_stride     = self.in_conv_stride    if block_idx == 0 else 1,
                mid_conv_stride    = self.mid_conv_stride   if block_idx == 0 else 1,

                mid_conv_out_channels = self.mid_conv_out_channels,
                out_conv_out_channels = self.stage_out_channels,

                H = self.H,
                W = self.W,
            )
            for block_idx in range(self.num_blocks)
        ]



@dataclass
class ConvNeXTConfig:
    stem_in_channels : int = 1
    stem_out_channels: int = 96

    H: int = 256
    W: int = 256

    stage_in_channels_in_layer    : List[int] = field(default_factory = lambda: [96, 96, 96, 96])
    stage_out_channels_in_layer   : List[int] = field(default_factory = lambda: [96, 96, 96, 96])
    num_blocks_in_layer           : List[int] = field(default_factory = lambda: [3,  3,  9,  3 ])
    mid_conv_out_channels_in_layer: List[int] = field(default_factory = lambda: [96, 96, 96, 96])
    in_conv_stride_in_layer       : List[int] = field(default_factory = lambda: [1,  1,  1,  1 ])
    mid_conv_stride_in_layer      : List[int] = field(default_factory = lambda: [1,  1,  1,  1 ])

    stem_config  : ConvNeXTStemConfig = field(init = False)
    layers_config: List[ConvNeXTStageConfig] = field(init = False)

    def __post_init__(self):
        self.stem_config = ConvNeXTStemConfig(
            in_channels  = self.stem_in_channels,
            out_channels = self.stem_out_channels,
            H            = self.H,
            W            = self.W,
        )

        H_scaled = self.stem_config.H_scaled
        W_scaled = self.stem_config.W_scaled
        num_layers = len(self.stage_in_channels_in_layer)
        self.layers_config = [
            ConvNeXTStageConfig(
                stage_in_channels     = self.stage_in_channels_in_layer    [layer_idx],
                stage_out_channels    = self.stage_out_channels_in_layer   [layer_idx],
                num_blocks            = self.num_blocks_in_layer           [layer_idx],
                mid_conv_out_channels = self.mid_conv_out_channels_in_layer[layer_idx],
                in_conv_stride        = self.in_conv_stride_in_layer       [layer_idx],
                mid_conv_stride       = self.mid_conv_stride_in_layer      [layer_idx],
                H = H_scaled,
                W = H_scaled,
            )
            for layer_idx in range(num_layers)
        ]
