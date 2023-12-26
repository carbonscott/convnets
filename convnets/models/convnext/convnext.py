"""
RegNet

Building blocks
- STEM
- BODY
  - STAGES (1..4)
    - BLOCKS (1..N)
- HEAD

width_per_stage: specifies the width (i.e., number of channels) for each stage.

Reference:
https://github.com/facebookresearch/pycls/blob/main/pycls/models/blocks.py
"""

import torch
import torch.nn            as nn
import torch.nn.functional as F

from dataclasses import asdict

from .blocks import conv2d, pool2d

from .convnext_config import DepthwiseSeparableConv2dConfig, Conv2dConfig, LayerNormConfig, ConvNeXTStemConfig, ConvNeXTBlockConfig, ConvNeXTStageConfig, ConvNeXTConfig


class DepthwiseSeparableConv2d(nn.Module):
    """
    As the name suggests, it's a conv2d doen in two steps:
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
        kernel_size  = 4
        stride       = 4
        conv_config = Conv2dConfig(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
        )

        C, H, W = 96, 256, 256
        scale_factor = kernel_size
        H //= scale_factor
        W //= scale_factor
        layer_norm_config = LayerNormConfig(normalized_shape = (C, H, W))

        return ConvNeXTStemConfig(layer_norm = layer_norm_config, conv = conv_config)


    def __init__(self, config = None):
        super().__init__()

        self.config = ConvNeXTStem.get_default_config() if config is None else config

        self.conv       = nn.Conv2d   (**asdict(self.config.conv      ))
        self.layer_norm = nn.LayerNorm(**asdict(self.config.layer_norm))    # Normalize (1, C, H, W)


    def forward(self, x):
        for layer in self.children():
            x = layer(x)

        return x




class ConvNeXTBlock(nn.Module):
    """
    Create X blocks for the RegNet architecture.

    in_channels, mid_channels (bottleneck channels), out_channels
    """

    @staticmethod
    def get_default_config():
        in_channels  = 96
        mid_channels = 96 * 4
        out_channels = 96

        in_kernel_size  = 7
        mid_kernel_size = 1
        out_kernel_size = 1

        in_padding  = (in_kernel_size  - 1) // 2
        mid_padding = (mid_kernel_size - 1) // 2
        out_padding = (out_kernel_size - 1) // 2

        in_conv_config = DepthwiseSeparableConv2dConfig(in_channels  = in_channels,
                                                        out_channels = in_channels,
                                                        kernel_size  = in_kernel_size,
                                                        padding      = in_padding,)    # ...Keep the spatial dimension unchanged

        mid_conv_config = Conv2dConfig(in_channels  = in_channels,
                                       out_channels = mid_channels,
                                       kernel_size  = mid_kernel_size,
                                       padding      = mid_padding,)

        out_conv_config = Conv2dConfig(in_channels  = mid_channels,
                                       out_channels = out_channels,
                                       kernel_size  = out_kernel_size,
                                       padding      = out_padding,)

        C, H, W = 96, 256, 256
        scale_factor = 4
        H //= scale_factor
        W //= scale_factor
        layer_norm_config = LayerNormConfig(normalized_shape = (C, H, W))

        return ConvNeXTBlockConfig(in_conv    = in_conv_config,
                                   mid_conv   = mid_conv_config,
                                   out_conv   = out_conv_config,
                                   layer_norm = layer_norm_config)


    def __init__(self, config = None):
        super().__init__()

        self.config = ConvNeXTBlock.get_default_config() if config is None else config

        self.in_conv = nn.Sequential(
            DepthwiseSeparableConv2d(**asdict(self.config.in_conv)),    # ...Keep the spatial dimension unchanged
            nn.LayerNorm(**asdict(self.config.layer_norm))
        )

        self.mid_conv = nn.Sequential(
            nn.Conv2d(**asdict(self.config.mid_conv)),
            nn.GELU(),
        )

        self.out_conv = nn.Conv2d(**asdict(self.config.out_conv))


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

    @staticmethod
    def get_default_config():
        return ConvNeXTStageConfig()

    def __init__(self, stage_in_channels,
                       stage_out_channels,
                       num_blocks,
                       mid_conv_channels,
                       mid_conv_groups,
                       in_conv_stride  = 1,
                       mid_conv_stride = 1,
                       config          = None):
        super().__init__()

        self.config = ConvNeXTStage.get_default_config() if config is None else config

        # Process all blocks sequentially...
        self.blocks = nn.Sequential(*[
            ConvNeXTBlock(
                # First block uses stage_in_channels and rest uses prev stage_out_channels...
                block_in_channels  = stage_in_channels if block_idx == 0 else stage_out_channels,

                block_out_channels = stage_out_channels,
                mid_conv_channels  = mid_conv_channels,
                mid_conv_groups    = mid_conv_groups,

                # First block uses in_conv_stride and rest uses 1...
                in_conv_stride     = in_conv_stride    if block_idx == 0 else 1,
                mid_conv_stride    = mid_conv_stride   if block_idx == 0 else 1,

                # Other config...
                config = self.config.RESBLOCK
            )
            for block_idx in range(num_blocks)
        ])


    def forward(self, x):
        x = self.blocks(x)

        return x




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

        # [[[ STEM ]]]
        self.stem = ConvNeXTStem(stem_in_channels = 1, stem_out_channels = 64, config = self.config.RESSTEM)

        # [[[ Layer 1 ]]]
        stage_in_channels  = 64
        stage_out_channels = 256
        mid_conv_channels  = stage_in_channels
        num_stages         = 1
        num_blocks         = 3
        in_conv_stride     = 1
        mid_conv_stride    = 1
        self.layer1 = nn.Sequential(
            pool2d(kernel_size = 3, stride = 2),    # ...Original ConvNeXTNet likes to have a pool in the first layer
            *[ ConvNeXTStage(stage_in_channels  = stage_in_channels if stage_idx == 0 else stage_out_channels,
                        stage_out_channels = stage_out_channels,
                        num_blocks         = num_blocks,
                        mid_conv_channels  = mid_conv_channels,
                        mid_conv_groups    = 1,
                        in_conv_stride     = in_conv_stride,
                        mid_conv_stride    = mid_conv_stride,
                        config             = self.config.RESSTAGE,)
            for stage_idx in range(num_stages) ]
        )

        # [[[ Layer 2 ]]]
        stage_in_channels  = 256
        stage_out_channels = 512
        mid_conv_channels  = 128
        num_stages         = 1
        num_blocks         = 4
        in_conv_stride     = 1 if self.config.RESSTAGE.RESBLOCK.USES_RES_V1p5 else 2
        mid_conv_stride    = 2 if self.config.RESSTAGE.RESBLOCK.USES_RES_V1p5 else 1
        self.layer2 = nn.Sequential(*[
            ConvNeXTStage(stage_in_channels  = stage_in_channels if stage_idx == 0 else stage_out_channels,
                     stage_out_channels = stage_out_channels,
                     num_blocks         = num_blocks,
                     mid_conv_channels  = mid_conv_channels,
                     mid_conv_groups    = 1,
                     in_conv_stride     = in_conv_stride,
                     mid_conv_stride    = mid_conv_stride,
                     config             = self.config.RESSTAGE,)
            for stage_idx in range(num_stages)
        ])

        # [[[ Layer 3 ]]]
        stage_in_channels  = 512
        stage_out_channels = 1024
        mid_conv_channels  = 256
        num_stages         = 1
        num_blocks         = 6
        in_conv_stride     = 1 if self.config.RESSTAGE.RESBLOCK.USES_RES_V1p5 else 2
        mid_conv_stride    = 2 if self.config.RESSTAGE.RESBLOCK.USES_RES_V1p5 else 1
        self.layer3 = nn.Sequential(*[
            ConvNeXTStage(stage_in_channels  = stage_in_channels if stage_idx == 0 else stage_out_channels,
                     stage_out_channels = stage_out_channels,
                     num_blocks         = num_blocks,
                     mid_conv_channels  = mid_conv_channels,
                     mid_conv_groups    = 1,
                     in_conv_stride     = in_conv_stride,
                     mid_conv_stride    = mid_conv_stride,
                     config             = self.config.RESSTAGE,)
            for stage_idx in range(num_stages)
        ])

        # [[[ Layer 4 ]]]
        stage_in_channels  = 1024
        stage_out_channels = 2048
        mid_conv_channels  = 512
        num_stages         = 1
        num_blocks         = 3
        in_conv_stride     = 1 if self.config.RESSTAGE.RESBLOCK.USES_RES_V1p5 else 2
        mid_conv_stride    = 2 if self.config.RESSTAGE.RESBLOCK.USES_RES_V1p5 else 1
        self.layer4 = nn.Sequential(*[
            ConvNeXTStage(stage_in_channels  = stage_in_channels if stage_idx == 0 else stage_out_channels,
                     stage_out_channels = stage_out_channels,
                     num_blocks         = num_blocks,
                     mid_conv_channels  = mid_conv_channels,
                     mid_conv_groups    = 1,
                     in_conv_stride     = in_conv_stride,
                     mid_conv_stride    = mid_conv_stride,
                     config             = self.config.RESSTAGE,)
            for stage_idx in range(num_stages)
        ])


    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
