from dataclasses import dataclass, field
from typing      import Optional

import torch


@dataclass
class LayerNormConfig:
    normalized_shape  : int
    eps               : float = 1e-5
    elementwise_affine: bool  = True


@dataclass
class ConvNeXTStemConfig:
    layer_norm : LayerNormConfig

    in_channels : int = 1
    out_channels: int = 96
    kernel_size : int = 4
    stride      : int = 4


@dataclass
class Conv2dConfig:
    in_channels : int
    out_channels: int
    kernel_size : int
    stride      : int = 1,
    padding     : int = 0,
    dilation    : int = 1,
    groups      : int = 1,
    bias        : bool = True,
    padding_mode: str = 'zeros',
    device      : Optional[torch.device] = None
    dtype       : Optional[torch.dtype]  = None


@dataclass
class ConvNeXTBlockConfig:
    layer_norm: LayerNormConfig

    in_channels : int = 96
    mid_channels: int = 96 * 4
    out_channels: int = 96

    in_kernel_size : int = 7
    mid_kernel_size: int = 1
    out_kernel_size: int = 1

    in_conv : Conv2dConfig = field(init = False)
    mid_conv: Conv2dConfig = field(init = False)
    out_conv: Conv2dConfig = field(init = False)

    uses_res_v1p5: bool = True

    def __post_init__(self):
        self.in_conv = Conv2dConfig(in_channels  = self.in_channels,
                                    out_channels = self.in_channels,
                                    kernel_size  = 7,
                                    padding      = 3,)    # ...Keep the spatial dimension unchanged

        self.mid_conv = Conv2dConfig(in_channels  = self.in_channels,
                                     out_channels = self.mid_channels,
                                     kernel_size  = 1)

        self.out_conv = Conv2dConfig(in_channels  = self.mid_channels,
                                     out_channels = self.out_channels,
                                     kernel_size  = 1)


@dataclass
class ConvNeXTStageConfig:
    RESBLOCK: ConvNeXTBlockConfig = field(default_factory = ConvNeXTBlockConfig)


@dataclass
class ConvNeXTConfig:
    RESSTEM : ConvNeXTStemConfig  = field(default_factory = ConvNeXTStemConfig)
    RESSTAGE: ConvNeXTStageConfig = field(default_factory = ConvNeXTStageConfig)
