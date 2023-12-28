"""
RegNet

Building blocks
- STEM
- BODY
  - STAGES (1..4)
    - BLOCKS (1..N)
- HEAD

width_per_stage: specifies the width (i.e., number of channels) for each stage.
"""

import torch
import torch.nn            as nn
import torch.nn.functional as F

from dataclasses import asdict

from .convnext_config import DepthwiseSeparableConv2dConfig, Conv2dConfig, LayerNormConfig, ConvNeXTStemConfig, ConvNeXTBlockConfig, ConvNeXTStageConfig, ConvNeXTConfig


class DepthwiseSeparableConv2d(nn.Module):
    """
    As the name suggests, it's a conv2d done in two steps:
    - Spatial only conv, no inter-channel communication.
    - Inter-channel communication, no spatial communication.
    """

    def __init__(self, in_channels,
                       out_channels,
                       kernel_size  = 1,
                       stride       = 1,
                       padding      = 0,
                       dilation     = 1,
                       bias         = True,
                       padding_mode = 'zeros',
                       device       = None,
                       dtype        = None):
        super().__init__()

        # Depthwise conv means channels are independent, only spatial bits communicate
        # Essentially it simply scales every tensor element
        self.depthwise_conv = nn.Conv2d(in_channels  = in_channels,
                                        out_channels = in_channels,
                                        kernel_size  = kernel_size,
                                        stride       = stride,
                                        padding      = padding,
                                        dilation     = dilation,
                                        groups       = in_channels,    # Input channels don't talk to each other
                                        bias         = bias,
                                        padding_mode = padding_mode,
                                        device       = device,
                                        dtype        = dtype)

        # Pointwise to facilitate inter-channel communication, no spatial bits communicate
        self.pointwise_conv = nn.Conv2d(in_channels  = in_channels,
                                        out_channels = out_channels,
                                        kernel_size  = 1,
                                        stride       = 1,
                                        padding      = 0,
                                        dilation     = 1,
                                        groups       = 1,    # Input channels don't talk to each other
                                        bias         = bias,
                                        padding_mode = padding_mode,
                                        device       = device,
                                        dtype        = dtype)


    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        return x


class DepthwiseConv2d(nn.Module):
    """
    As the name suggests, it's a conv2d done in two steps:
    - Spatial only conv, no inter-channel communication.
    """

    def __init__(self, in_channels,
                       out_channels,
                       kernel_size  = 1,
                       stride       = 1,
                       padding      = 0,
                       dilation     = 1,
                       bias         = True,
                       padding_mode = 'zeros',
                       device       = None,
                       dtype        = None):
        super().__init__()

        # Depthwise conv means channels are independent, only spatial bits communicate
        # Essentially it simply scales every tensor element
        self.depthwise_conv = nn.Conv2d(in_channels  = in_channels,
                                        out_channels = in_channels,
                                        kernel_size  = kernel_size,
                                        stride       = stride,
                                        padding      = padding,
                                        dilation     = dilation,
                                        groups       = in_channels,    # Input channels don't talk to each other
                                        bias         = bias,
                                        padding_mode = padding_mode,
                                        device       = device,
                                        dtype        = dtype)


    def forward(self, x):
        x = self.depthwise_conv(x)

        return x


class ChannelwiseLayerNorm(nn.Module):
    def __init__(self, layer_norm_config):
        super().__init__()

        self.layer_norm = nn.LayerNorm(**asdict(layer_norm_config))


    def forward(self, x):
        return self.layer_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)    # (B, C, H, W) -> (B, H, W, C) -> (B, C, H, W)



class ConvNeXTStem(nn.Module):
    """
    This class implments the first layer (STEM in RegNet's nomenclature) of
    ConvNeXT.

    Spatial dimension change: (1, H, W) -> (C, H//4, W//4)

    Each patch is turned into a C-dimension vector.
    """

    @staticmethod
    def get_default_config():
        in_channels  = 1
        out_channels = 96

        return ConvNeXTStemConfig(in_channels  = in_channels,
                                  out_channels = out_channels,)


    def __init__(self, config = None):
        super().__init__()

        self.config = ConvNeXTStem.get_default_config() if config is None else config

        self.conv       = nn.Conv2d   (**asdict(self.config.conv_config))
        self.layer_norm = ChannelwiseLayerNorm(self.config.layer_norm_config)    # Normalize (1, C, H, W)


    def forward(self, x):
        x = self.conv(x)
        x = self.layer_norm(x)

        return x




class ConvNeXTBlock(nn.Module):
    """
    Create X blocks for the RegNet architecture.

    in_channels, mid_channels (bottleneck channels), out_channels
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.in_conv = nn.Sequential(
            DepthwiseSeparableConv2d(**asdict(self.config.in_conv_config)),    # ...Keep the spatial dimension unchanged
            ChannelwiseLayerNorm(self.config.layer_norm_config),
        )

        self.mid_conv = nn.Sequential(
            nn.Conv2d(**asdict(self.config.mid_conv_config)),
            nn.GELU(),
        )

        self.out_conv = nn.Conv2d(**asdict(self.config.out_conv_config))


    def forward(self, x):
        y = self.in_conv(x)
        y = self.mid_conv(y)
        y = self.out_conv(y)

        y = y + x

        return y




class ConvNeXTStage(nn.Module):
    """
    This class implments the one stage in the RegNet architecture.

    Block means a res block.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        # Process all blocks sequentially...
        self.blocks = nn.Sequential(*[
            ConvNeXTBlock(config = block_config)
            for block_config in self.config.block_config_list
        ])


    def forward(self, x):
        x = self.blocks(x)

        return x




class ConvNeXTStageJoint(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.stage_joint = nn.Sequential(
            ChannelwiseLayerNorm(self.config.layer_norm_config),
            nn.Conv2d(**asdict(self.config.conv_config)),
        )


    def forward(self, x):
        return self.stage_joint(x)




class ConvNeXT(nn.Module):
    """
    This class implements single channel ConvNeXT using the RegNet
    nomenclature.

    ConvNeXT architecture reference: [NEED URL]

    ConvNeXTStage(s) are kept in nn.Sequential but not nn.ModuleList since they will
    be processed sequentially.
    """

    @staticmethod
    def get_default_config():
        return ConvNeXTConfig()


    def __init__(self, config = None):
        super().__init__()

        self.config = ConvNeXT.get_default_config() if config is None else config

        self.stem = ConvNeXTStem(self.config.stem_config)

        self.stages = nn.ModuleList([
            ConvNeXTStage(config = stage_config)
            for stage_config in self.config.stages_config
        ])

        self.stage_joints = nn.ModuleList([
            nn.Identity() if stage_joint_config is None else
            ConvNeXTStageJoint(config = stage_joint_config)
            for stage_joint_config in self.config.stage_joints_config
        ])


    def forward(self, x):
        x = self.stem(x)
        for stage_joint, stage in zip(self.stage_joints, self.stages):
            x = stage_joint(x)
            x = stage(x)

        return x
