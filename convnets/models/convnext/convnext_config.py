from dataclasses import dataclass, field
from typing      import Optional

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
    conv      : Conv2dConfig
    layer_norm: LayerNormConfig


@dataclass
class ConvNeXTBlockConfig:
    in_conv   : DepthwiseSeparableConv2dConfig
    layer_norm: LayerNormConfig
    mid_conv  : Conv2dConfig
    out_conv  : Conv2dConfig

    uses_res_v1p5: bool = True


@dataclass
class ConvNeXTStageConfig:
    RESBLOCK: ConvNeXTBlockConfig = field(default_factory = ConvNeXTBlockConfig)


@dataclass
class ConvNeXTConfig:
    RESSTEM : ConvNeXTStemConfig  = field(default_factory = ConvNeXTStemConfig)
    RESSTAGE: ConvNeXTStageConfig = field(default_factory = ConvNeXTStageConfig)
