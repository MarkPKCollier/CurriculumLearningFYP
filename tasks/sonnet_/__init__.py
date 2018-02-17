# Copyright 2017 The sonnet_ Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""This python module contains Neural Network Modules for TensorFlow.

Each module is a Python object which conceptually "owns" any
variables required in that part of the Neural Network. The `__call__` function
on the object is used to connect that Module into the Graph, and this may be
called repeatedly with sharing automatically taking place.

Everything public should be imported by this top level `__init__.py` so that the
library can be used as follows:

```
import sonnet_ as snt

linear = snt.Linear(...)
```
"""

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import sys

from python import custom_getters
from python.modules import experimental
# from python.modules import nets
from python.modules.attention import AttentiveRead
from python.modules.base import AbstractModule
from python.modules.base import Module
from python.modules.base import Transposable
from python.modules.base_errors import DifferentGraphError
from python.modules.base_errors import Error
from python.modules.base_errors import IncompatibleShapeError
from python.modules.base_errors import ModuleInfoError
from python.modules.base_errors import NotConnectedError
from python.modules.base_errors import NotInitializedError
from python.modules.base_errors import NotSupportedError
from python.modules.base_errors import ParentNotBuiltError
from python.modules.base_errors import UnderspecifiedError
from python.modules.base_errors import UnderspecifiedError
from python.modules.base_info import SONNET_COLLECTION_NAME
from python.modules.basic import AddBias
from python.modules.basic import BatchApply
from python.modules.basic import BatchFlatten
from python.modules.basic import BatchReshape
from python.modules.basic import FlattenTrailingDimensions
from python.modules.basic import Linear
from python.modules.basic import merge_leading_dims
from python.modules.basic import MergeDims
from python.modules.basic import SelectInput
from python.modules.basic import SliceByDim
from python.modules.basic import split_leading_dim
from python.modules.basic import TileByDim
from python.modules.basic import TrainableVariable
from python.modules.basic_rnn import DeepRNN
from python.modules.basic_rnn import ModelRNN
from python.modules.basic_rnn import VanillaRNN
from python.modules.batch_norm import BatchNorm
from python.modules.batch_norm_v2 import BatchNormV2
from python.modules.clip_gradient import clip_gradient
from python.modules.conv import CausalConv1D
from python.modules.conv import Conv1D
from python.modules.conv import Conv1DTranspose
from python.modules.conv import Conv2D
from python.modules.conv import Conv2DTranspose
from python.modules.conv import Conv3D
from python.modules.conv import Conv3DTranspose
from python.modules.conv import DepthwiseConv2D
from python.modules.conv import InPlaneConv2D
from python.modules.conv import SAME
from python.modules.conv import SeparableConv2D
from python.modules.conv import VALID
from python.modules.embed import Embed
from python.modules.gated_rnn import BatchNormLSTM
from python.modules.gated_rnn import Conv1DLSTM
from python.modules.gated_rnn import Conv2DLSTM
from python.modules.gated_rnn import GRU
from python.modules.gated_rnn import highway_core_with_recurrent_dropout
from python.modules.gated_rnn import HighwayCore
from python.modules.gated_rnn import LSTM
from python.modules.gated_rnn import lstm_with_recurrent_dropout
from python.modules.gated_rnn import lstm_with_zoneout
from python.modules.gated_rnn import LSTMState
from python.modules.layer_norm import LayerNorm
from python.modules.pondering_rnn import ACTCore
from python.modules.residual import Residual
from python.modules.residual import ResidualCore
from python.modules.residual import SkipConnectionCore
from python.modules.rnn_core import RNNCore
from python.modules.rnn_core import trainable_initial_state
from python.modules.rnn_core import TrainableInitialState
from python.modules.scale_gradient import scale_gradient
from python.modules.sequential import Sequential
from python.modules.spatial_transformer import AffineGridWarper
from python.modules.spatial_transformer import AffineWarpConstraints
from python.modules.spatial_transformer import GridWarper
from python.modules.util import check_initializers
from python.modules.util import check_partitioners
from python.modules.util import check_regularizers
from python.modules.util import custom_getter_router
from python.modules.util import format_variable_map
from python.modules.util import format_variables
from python.modules.util import get_normalized_variable_map
from python.modules.util import get_saver
from python.modules.util import get_variables_in_module
from python.modules.util import get_variables_in_scope
from python.modules.util import has_variable_scope
from python.modules.util import log_variables
from python.modules.util import reuse_variables
from python.modules.util import variable_map_items
from python.ops import nest
from python.ops.initializers import restore_initializer

__version__ = '1.16'

