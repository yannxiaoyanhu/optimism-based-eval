# This part of the code is based on the implementation of dinov2/dinov2/layers/__init__.py
# in the official repository of dinov2: https://github.com/facebookresearch/dinov2
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0

from .dino_head import DINOHead
from .mlp import Mlp
from .patch_embed import PatchEmbed
from .swiglu_ffn import SwiGLUFFN, SwiGLUFFNFused
from .block import NestedTensorBlock
from .attention import MemEffAttention
