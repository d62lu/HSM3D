"""
Point Transformer - V3 Mode1
Pointcept detached version

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""
import gc
import sys
from functools import partial
from addict import Dict
import math
import torch
import torch.nn as nn
import spconv.pytorch as spconv
import torch_scatter
from timm.layers import DropPath
from collections import OrderedDict
import torch.nn.functional as F
try:
    import flash_attn
except ImportError:
    flash_attn = None

from serialization import encode
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="spconv.pytorch.functional")
@torch.inference_mode()
def offset2bincount(offset):
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )


@torch.inference_mode()
def offset2batch(offset):
    bincount = offset2bincount(offset)
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)


@torch.inference_mode()
def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()


class Point(Dict):
    """
    Point Structure of Pointcept

    A Point (point cloud) in Pointcept is a dictionary that contains various properties of
    a batched point cloud. The property with the following names have a specific definition
    as follows:

    - "coord": original coordinate of point cloud;
    - "grid_coord": grid coordinate for specific grid size (related to GridSampling);
    Point also support the following optional attributes:
    - "offset": if not exist, initialized as batch size is 1;
    - "batch": if not exist, initialized as batch size is 1;
    - "feat": feature of point cloud, default input of model;
    - "grid_size": Grid size of point cloud (related to GridSampling);
    (related to Serialization)
    - "serialized_depth": depth of serialization, 2 ** depth * grid_size describe the maximum of point cloud range;
    - "serialized_code": a list of serialization codes;
    - "serialized_order": a list of serialization order determined by code;
    - "serialized_inverse": a list of inverse mapping determined by code;
    (related to Sparsify: SpConv)
    - "sparse_shape": Sparse shape for Sparse Conv Tensor;
    - "sparse_conv_feat": SparseConvTensor init with information provide by Point;
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # If one of "offset" or "batch" do not exist, generate by the existing one
        if "batch" not in self.keys() and "offset" in self.keys():
            self["batch"] = offset2batch(self.offset)
        elif "offset" not in self.keys() and "batch" in self.keys():
            self["offset"] = batch2offset(self.batch)

    def serialization(self, order="z", depth=None, shuffle_orders=False):
        """
        Point Cloud Serialization

        relay on ["grid_coord" or "coord" + "grid_size", "batch", "feat"]
        """
        assert "batch" in self.keys()
        if "grid_coord" not in self.keys():
            # if you don't want to operate GridSampling in data augmentation,
            # please add the following augmentation into your pipline:
            # dict(type="Copy", keys_dict={"grid_size": 0.01}),
            # (adjust `grid_size` to what your want)
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()

        if depth is None:
            # Adaptive measure the depth of serialization cube (length = 2 ^ depth)
            depth = int(self.grid_coord.max()).bit_length()
        self["serialized_depth"] = depth
        # Maximum bit length for serialization code is 63 (int64)
        assert depth * 3 + len(self.offset).bit_length() <= 63
        # Here we follow OCNN and set the depth limitation to 16 (48bit) for the point position.
        # Although depth is limited to less than 16, we can encode a 655.36^3 (2^16 * 0.01) meter^3
        # cube with a grid size of 0.01 meter. We consider it is enough for the current stage.
        # We can unlock the limitation by optimizing the z-order encoding function if necessary.
        assert depth <= 16

        # The serialization codes are arranged as following structures:
        # [Order1 ([n]),
        #  Order2 ([n]),
        #   ...
        #  OrderN ([n])] (k, n)
        code = [
            encode(self.grid_coord, self.batch, depth, order=order_) for order_ in order
        ]
        code = torch.stack(code)
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        self["serialized_code"] = code
        self["serialized_order"] = order
        self["serialized_inverse"] = inverse

    def sparsify(self, pad=96):
        """
        Point Cloud Serialization

        Point cloud is sparse, here we use "sparsify" to specifically refer to
        preparing "spconv.SparseConvTensor" for SpConv.

        relay on ["grid_coord" or "coord" + "grid_size", "batch", "feat"]

        pad: padding sparse for sparse shape.
        """
        assert {"feat", "batch"}.issubset(self.keys())
        if "grid_coord" not in self.keys():
            # if you don't want to operate GridSampling in data augmentation,
            # please add the following augmentation into your pipline:
            # dict(type="Copy", keys_dict={"grid_size": 0.01}),
            # (adjust `grid_size` to what your want)
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()
        if "sparse_shape" in self.keys():
            sparse_shape = self.sparse_shape
        else:
            sparse_shape = torch.add(
                torch.max(self.grid_coord, dim=0).values, pad
            ).tolist()
        sparse_conv_feat = spconv.SparseConvTensor(
            features=self.feat,
            indices=torch.cat(
                [self.batch.unsqueeze(-1).int(), self.grid_coord.int()], dim=1
            ).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=self.batch[-1].tolist() + 1,
        )
        self["sparse_shape"] = sparse_shape
        self["sparse_conv_feat"] = sparse_conv_feat


class PointModule(nn.Module):
    r"""PointModule
    placeholder, all module subclass from this will take Point in PointSequential.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PointSequential(PointModule):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError("index {} is out of range".format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input):
        for k, module in self._modules.items():
            # Point module
            if isinstance(module, PointModule):
                input = module(input)
            # Spconv module
            elif spconv.modules.is_spconv_module(module):
                if isinstance(input, Point):
                    input.sparse_conv_feat = module(input.sparse_conv_feat)
                    input.feat = input.sparse_conv_feat.features
                else:
                    input = module(input)
            # PyTorch module
            else:
                if isinstance(input, Point):
                    input.feat = module(input.feat)
                    if "sparse_conv_feat" in input.keys():
                        input.sparse_conv_feat = input.sparse_conv_feat.replace_feature(
                            input.feat
                        )
                elif isinstance(input, spconv.SparseConvTensor):
                    if input.indices.shape[0] != 0:
                        input = input.replace_feature(module(input.features))
                else:
                    input = module(input)
        return input


class PDNorm(PointModule):
    def __init__(
        self,
        num_features,
        norm_layer,
        context_channels=256,
        conditions=("ScanNet", "S3DIS", "Structured3D"),
        decouple=True,
        adaptive=False,
    ):
        super().__init__()
        self.conditions = conditions
        self.decouple = decouple
        self.adaptive = adaptive
        if self.decouple:
            self.norm = nn.ModuleList([norm_layer(num_features) for _ in conditions])
        else:
            self.norm = norm_layer
        if self.adaptive:
            self.modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(context_channels, 2 * num_features, bias=True)
            )

    def forward(self, point):
        assert {"feat", "condition"}.issubset(point.keys())
        if isinstance(point.condition, str):
            condition = point.condition
        else:
            condition = point.condition[0]
        if self.decouple:
            assert condition in self.conditions
            norm = self.norm[self.conditions.index(condition)]
        else:
            norm = self.norm
        point.feat = norm(point.feat)
        if self.adaptive:
            assert "context" in point.keys()
            shift, scale = self.modulation(point.context).chunk(2, dim=1)
            point.feat = point.feat * (1.0 + scale) + shift
        return point


class RPE(torch.nn.Module):
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = torch.nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord):
        idx = (
            coord.clamp(-self.pos_bnd, self.pos_bnd)  # clamp into bnd
            + self.pos_bnd  # relative position to positive index
            + torch.arange(3, device=coord.device) * self.rpe_num  # x, y, z stride
        )
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
        return out


class SerializedAttention(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe
        self.enable_flash = enable_flash
        if enable_flash:
            assert (
                enable_rpe is False
            ), "Set enable_rpe to False when enable Flash Attention"
            assert (
                upcast_attention is False
            ), "Set upcast_attention to False when enable Flash Attention"
            assert (
                upcast_softmax is False
            ), "Set upcast_softmax to False when enable Flash Attention"
            assert flash_attn is not None, "Make sure flash_attn is installed."
            self.patch_size = patch_size
            self.attn_drop = attn_drop
        else:
            # when disable flash attention, we still don't want to use mask
            # consequently, patch size will auto set to the
            # min number of patch_size_max and number of points
            self.patch_size_max = patch_size
            self.patch_size = 0
            self.attn_drop = torch.nn.Dropout(attn_drop)

        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(channels, channels)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.rpe = RPE(patch_size, num_heads) if self.enable_rpe else None

    @torch.no_grad()
    def get_rel_pos(self, point, order):
        K = self.patch_size
        rel_pos_key = f"rel_pos_{self.order_index}"
        if rel_pos_key not in point.keys():
            grid_coord = point.grid_coord[order]
            grid_coord = grid_coord.reshape(-1, K, 3)
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        return point[rel_pos_key]

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        if (
            pad_key not in point.keys()
            or unpad_key not in point.keys()
            or cu_seqlens_key not in point.keys()
        ):
            offset = point.offset
            bincount = offset2bincount(offset)
            bincount_pad = (
                torch.div(
                    bincount + self.patch_size - 1,
                    self.patch_size,
                    rounding_mode="trunc",
                )
                * self.patch_size
            )
            # only pad point when num of points larger than patch_size
            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []
            for i in range(len(offset)):
                unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    pad[
                        _offset_pad[i + 1]
                        - self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                    ] = pad[
                        _offset_pad[i + 1]
                        - 2 * self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                        - self.patch_size
                    ]
                pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = nn.functional.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )
        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, point):
        if not self.enable_flash:
            self.patch_size = min(
                offset2bincount(point.offset).min().tolist(), self.patch_size_max
            )

        H = self.num_heads
        K = self.patch_size
        C = self.channels

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        # padding and reshape feat and batch for serialized point patch
        qkv = self.qkv(point.feat)[order]

        if not self.enable_flash:
            # encode and reshape qkv: (N', K, 3, H, C') => (3, N', H, K, C')
            q, k, v = (
                qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            )
            # attn
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
            if self.enable_rpe:
                attn = attn + self.rpe(self.get_rel_pos(point, order))
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn)
            attn = self.attn_drop(attn).to(qkv.dtype)
            feat = (attn @ v).transpose(1, 2).reshape(-1, C)
        else:
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.half().reshape(-1, 3, H, C // H),
                cu_seqlens,
                max_seqlen=self.patch_size,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            ).reshape(-1, C)
            feat = feat.to(qkv.dtype)
        feat = feat[inverse]

        # ffn
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


from mamba_ssm.modules.mamba_simple import Mamba

class SerializedMamba(PointModule):
    def __init__(
        self,
        channels,
        patch_size,
        mlp_ratio=2.0,
        drop=0.0,
        order_index=0,
    ):
        super().__init__()
        self.channels = channels
        self.patch_size = patch_size
        self.order_index = order_index

        self.norm1 = nn.LayerNorm(channels)
        self.mamba = Mamba(d_model=channels)
        self.norm2 = nn.LayerNorm(channels)

        self.ffn = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(channels * mlp_ratio), channels),
            nn.Dropout(drop),
        )

    def forward(self, point):
        K = self.patch_size
        C = self.channels

        # 排序点
        order = point.serialized_order[self.order_index]
        inverse = point.serialized_inverse[self.order_index]

        # (N, C) -> (N', K, C)
        # print(point.feat.shape)
        feat = point.feat[order].unsqueeze(0)  # [B', K, C]

        # Mamba block
        x = feat + self.mamba(self.norm1(feat))
        x = x + self.ffn(self.norm2(x))

        # 展平并 inverse 回原始顺序
        feat_out = x.reshape(-1, C)[inverse]
        point.feat = feat_out
        return point


class GlobalMamba(PointModule):
    def __init__(
        self,
        channels,
        mlp_ratio=2.0,
        drop=0.0,
    ):
        super().__init__()
        self.channels = channels


        self.norm1 = nn.LayerNorm(channels)
        self.mamba = Mamba(d_model=channels)
        self.norm2 = nn.LayerNorm(channels)

        self.ffn = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(channels * mlp_ratio), channels),
            nn.Dropout(drop),
        )

    def forward(self, point):


  

        # (N, C) -> (N', K, C)
        # print(point.feat.shape)
        feat = point # [1, N, C]

        # Mamba block
        x = feat + self.mamba(self.norm1(feat))
        x = x + self.ffn(self.norm2(x))

        # 展平并 inverse 回原始顺序
        feat_out = x # 1 N C

        return feat_out


class SparseMamba(PointModule):
    def __init__(
        self,
        channels,
        mlp_ratio=2.0,
        drop=0.0,
    ):
        super().__init__()
        self.channels = channels


        self.norm1 = nn.LayerNorm(channels)
        self.mamba = Mamba(d_model=channels)
        self.norm2 = nn.LayerNorm(channels)

        self.ffn = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(channels * mlp_ratio), channels),
            nn.Dropout(drop),
        )

    def forward(self, point):

        C = self.channels

  

        # (N, C) -> (N', K, C)
        # print(point.feat.shape)
        feat = point # [B', K, C]

        # Mamba block
        x = feat + self.mamba(self.norm1(feat))
        x = x + self.ffn(self.norm2(x))

        # 展平并 inverse 回原始顺序
        feat_out = x[:,-1,:]  # [B', C]

        return feat_out


class Block(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
        feat_opt = 'mamba',
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        self.feat_opt = feat_opt

        self.cpe = PointSequential(
            spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,
            ),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )
        self.mamba = SerializedMamba(
            channels=channels,
            patch_size=patch_size,
            mlp_ratio=mlp_ratio,
            drop=proj_drop,
            order_index=order_index,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, point: Point):
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        if self.feat_opt == 'mamba':
            point = self.mamba(point)
        elif self.feat_opt == 'attn':
            point = self.attn(point)
        else:
            assert False, f"Unknown feat_opt: {self.feat_opt}"
        point = self.drop_path(point)
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
      
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class SerializedPooling(PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,  # record parent and cluster
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()  # 2, 4, 8
        # TODO: add support to grid pool (any stride)
        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        if act_layer is not None:
            self.act = PointSequential(act_layer())

    def forward(self, point: Point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0
        assert {
            "serialized_code",
            "serialized_order",
            "serialized_inverse",
            "serialized_depth",
        }.issubset(
            point.keys()
        ), "Run point.serialization() point cloud before SerializedPooling"

        code = point.serialized_code >> pooling_depth * 3
        code_, cluster, counts = torch.unique(
            code[0],
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        # indices of point sorted by cluster, for torch_scatter.segment_csr
        _, indices = torch.sort(cluster)
        # index pointer for sorted point, for torch_scatter.segment_csr
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        # head_indices of each cluster, for reduce attr e.g. code, batch
        head_indices = indices[idx_ptr[:-1]]
        # generate down code, order, inverse
        code = code[:, head_indices]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        # collect information
        point_dict = Dict(
            feat=torch_scatter.segment_csr(
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            ),
            coord=torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],
        )

        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        point.sparsify()
        return point


class SerializedUnpooling(PointModule):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
        traceable=False,  # record parent and cluster
    ):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))

        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))

        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())

        self.traceable = traceable

    def forward(self, point):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        point = self.proj(point)
        parent = self.proj_skip(parent)
        parent.feat = parent.feat + point.feat[inverse]

        if self.traceable:
            parent["unpooling_parent"] = point
        return parent


class Embedding(PointModule):
    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels

        # TODO: check remove spconv
        self.stem = PointSequential(
            conv=spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            )
        )
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        point = self.stem(point)
        return point

import torch
import torch_scatter

import torch
def _clean_point_cache(pt):
    for k in [
        "serialized_code", "serialized_order", "serialized_inverse",
        "pad", "unpad", "cu_seqlens_key",
        "rel_pos_0", "rel_pos_1", "rel_pos_2", "rel_pos_3"
    ]:
        if k in pt:
            del pt[k]



def average_pool_by_sp(x: torch.Tensor, sp_idx: torch.Tensor):
    """
    Args:
        x: (N, D) 特征
        sp_idx: (N,) superpoint 编号（不要求连续）

    Returns:
        x_sp: (S, D) 每个 superpoint 的 pooled 特征
        inverse: (N,) 每个 raw 点对应的 superpoint 索引（即 x_sp[inverse] = 每个点的 superpoint 特征）
    """
    unique_sp, inverse = torch.unique(sp_idx, sorted=True, return_inverse=True)  # inverse ∈ [0, S)
    S = unique_sp.shape[0]
    D = x.shape[1]

    # 初始化池化结果和计数器
    x_sp = torch.zeros((S, D), dtype=x.dtype, device=x.device)
    count = torch.zeros(S, dtype=torch.long, device=x.device)

    # 累加特征
    x_sp.index_add_(0, inverse, x)
    count.index_add_(0, inverse, torch.ones_like(inverse, dtype=torch.long))

    # 平均池化
    x_sp = x_sp / count.unsqueeze(-1)

    return x_sp, inverse

def build_sp_sequences_tensor(raw_feats: torch.Tensor,
                              raw2sp_idx: torch.Tensor,
                              sp_feats_init: torch.Tensor,
                              K: int = 10) -> torch.Tensor:
    """
    构建每个 superpoint 的序列：K 个 raw 点特征 + 1 个 sp token。
    全程使用 PyTorch 操作，无 numpy。

    Args:
        raw_feats (Tensor): 原始点特征，shape (N_raw, D)
        raw2sp_idx (Tensor): 每个点的 superpoint 编号，shape (N_raw,)
        sp_feats_init (Tensor): 初始 superpoint 特征，shape (N_sp, D)
        K (int): 每个 superpoint 采样 raw 点数量（默认10）

    Returns:
        seq_tensor (Tensor): 输出序列，shape (N_sp, K+1, D)
    """
    device = raw_feats.device
    N_sp, D = sp_feats_init.shape
    N_raw = raw_feats.shape[0]
    L = K + 1  # 总序列长度

    # print('sp_feats_init.shape: ', sp_feats_init.shape)
    # print('raw2sp_idx.shape: ', raw2sp_idx.shape)
    # print('unique superpoint count: ', raw2sp_idx.unique().numel())

    assert sp_feats_init.shape[0] == raw2sp_idx.unique().numel(), \
        "wrong shape of sp_feats_init, should match unique superpoint count"

    # Step 1: 排序 raw2sp_idx，让同一个 superpoint 的点连续排列
    sorted_idx = torch.argsort(raw2sp_idx)
    sorted_raw2sp = raw2sp_idx[sorted_idx]
    sorted_feats = raw_feats[sorted_idx]

    # Step 2: 找出每个 superpoint 的起始和结束索引
    unique_sp, counts = torch.unique_consecutive(sorted_raw2sp, return_counts=True)
    # print('min unique_sp: ', torch.min(unique_sp), ', max unique_sp: ', torch.max(unique_sp), ', unique_sp.shape: ', unique_sp)
    sp_start = torch.zeros(N_sp, dtype=torch.long, device=device)
    sp_len = torch.zeros(N_sp, dtype=torch.long, device=device)

    sp_start[unique_sp] = torch.cumsum(torch.cat([torch.tensor([0], device=device), counts[:-1]], dim=0), dim=0)
    sp_len[unique_sp] = counts

    # Step 3: 为每个 superpoint 构造采样索引（K 个）
    max_len = torch.clamp(sp_len, min=1)
    rand_idx = torch.randint(0, K, (N_sp, K), device=device)
    mod_idx = rand_idx % max_len.unsqueeze(1)  # 防止溢出
    gather_idx = sp_start.unsqueeze(1) + mod_idx  # (N_sp, K)

    # Step 4: 取出对应的 raw_feats，并拼接上 sp_token
    sampled_feats = sorted_feats[gather_idx]  # (N_sp, K, D)
    sp_token = sp_feats_init.unsqueeze(1)     # (N_sp, 1, D)
    seq_tensor = torch.cat([sampled_feats, sp_token], dim=1)  # (N_sp, K+1, D)

    return seq_tensor

import torch



def reshape_with_dynamic_offset_fast(neighbor_sp_idx_array):
    """
    输入：
    - neighbor_sp_idx_array: Tensor of shape (B, L, N, K)

    输出：
    - neighbor_sp_idx_array_offset: shape (L, B*N, K)，每个 (b,l,k) 的 superpoint ID 根据前面 batch 动态偏移
    """
    
    B, L, N, K = neighbor_sp_idx_array.shape
    device = neighbor_sp_idx_array.device
    spl_list_levels= []
    for l in range(neighbor_sp_idx_array.shape[1]):
        spn_list_b = []
        for b in range(B):
            spn_list_b.append(torch.unique(neighbor_sp_idx_array[b,l,:,0]).shape[0])
        spl_list_levels.append(spn_list_b)



    # 将输入重排为 (B*L*K, N)
    reshaped = neighbor_sp_idx_array.permute(0, 1, 3, 2).reshape(-1, N)  # [B*L*K, N]

    # 统计每个 (b,l,k) 的 unique 数量
    # 注意：unique 操作是逐行执行的，不是 batch 的，所以我们拆成单维处理
    # 使用 torch v2.0+ 可通过 torch.func.vmap 进一步加速
    unique_counts = torch.tensor([
        torch.unique(row).numel() for row in reshaped
    ], device=device, dtype=torch.int32)  # shape: [B*L*K]

    # reshape 为 (B, L, K)
    unique_counts = unique_counts.view(B, L, K)  # 每个 batch 每层每通道的 superpoint 数

    # 计算偏移量：沿 batch 维度做 cumsum，然后 roll 一位右移，首个 batch 设为 0
    offset = torch.cumsum(unique_counts, dim=0)  # shape: [B, L, K]
    offset = torch.roll(offset, shifts=1, dims=0)
    offset[0] = 0  # 第一个 batch 无偏移

    # expand 到 [B, L, N, K]
    offset_expanded = offset[:, :, None, :].expand(B, L, N, K)

    # 加偏移
    neighbor_sp_idx_array_offset = neighbor_sp_idx_array + offset_expanded  # shape: [B, L, N, K]

    # reshape 为 (L, B*N, K)
    out = neighbor_sp_idx_array_offset.permute(1, 0, 2, 3).reshape(L, B * N, K)
    # print(torch.unique(out[0, :, 0]).shape)
    # print('min unique_sp: ', torch.min(out[0, :, 0]), ', max unique_sp: ', torch.max(out[0, :, 0]))
    # print(torch.unique(out[1, :, 0]).shape)
    # print('min unique_sp: ', torch.min(out[1, :, 0]), ', max unique_sp: ', torch.max(out[1, :, 0]))
    return out, spl_list_levels


class HSM3D(PointModule):
    def __init__(
        self,
        num_class = 8,
        grid_size = 0.02,
        in_channels=8,
        sm_K = 10,

        order=("z", "z-trans", "hilbert", "hilbert-trans"),

        stride=(2, 2, 2),
        enc_depths=(2, 2, 2, 6),
        enc_channels_sp1=(32, 64, 128, 256),
        enc_channels_sp2=(64, 128, 256, 512),
        enc_num_head_sp1=(2, 4, 8, 16),
        enc_num_head_sp2=(4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024), #(1024, 1024, 1024, 1024, 1024) (1024, 512, 512, 512, 512)
        dec_depths=(2, 2, 2),
        dec_channels_sp1=(64, 64, 128),
        dec_channels_sp2=(64, 128, 256),
        dec_num_head_sp1=(4, 8, 16),
        dec_num_head_sp2=(4, 8, 16),
        dec_patch_size=(1024, 1024, 1024), #(1024, 1024, 1024, 1024) (512, 512, 512, 512)

        # stride=(2, 2),
        # enc_depths=(4, 8, 4),
        # enc_channels=(32, 128, 512),
        # enc_num_head=(2, 8, 32),
        # enc_patch_size=(1024, 1024, 1024),
        # dec_depths=(4,4),
        # dec_channels=(64, 128),
        # dec_num_head=(4, 8),
        # dec_patch_size=(1024, 1024),

        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders
        self.grid_size = grid_size
        self.sm_K = sm_K

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels_sp1)
        assert self.num_stages == len(enc_num_head_sp1)
        assert self.num_stages == len(enc_patch_size)
        assert self.cls_mode or self.num_stages == len(dec_depths) + 1
        assert self.cls_mode or self.num_stages == len(dec_channels_sp1) + 1
        assert self.cls_mode or self.num_stages == len(dec_num_head_sp1) + 1
        assert self.cls_mode or self.num_stages == len(dec_patch_size) + 1

        # norm layers
        if pdnorm_bn:
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(
                    nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine
                ),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        if pdnorm_ln:
            ln_layer = partial(
                PDNorm,
                norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            ln_layer = nn.LayerNorm
        # activation layers
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels_sp1[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        # encoder
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]


        s = 0
        enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
        self.enc_0_module = PointSequential()
        
        for i in range(enc_depths[s]):
                self.enc_0_module.add(
                    Block(
                        channels=enc_channels_sp1[s],
                        num_heads=enc_num_head_sp1[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                        feat_opt = 'attn',
                    ),
                    name=f"block{i}",
                )
        
        s = 1
        enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
        self.sm_sp1 = SparseMamba(
                        channels=enc_channels_sp1[s - 1],
                        mlp_ratio=mlp_ratio,
                        drop=proj_drop,
 
                    )
        self.gm_sp1 = GlobalMamba(
                        channels=enc_channels_sp1[s - 1],
                        mlp_ratio=mlp_ratio,
                        drop=proj_drop,
 
                    )
        self.ds_1_sp1 = SerializedPooling(
                        in_channels=enc_channels_sp1[s - 1],
                        out_channels=enc_channels_sp1[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    )
        self.sm_sp2 = SparseMamba(
                        channels=enc_channels_sp2[s - 1],
                        mlp_ratio=mlp_ratio,
                        drop=proj_drop,

                    )
        self.gm_sp2 = GlobalMamba(
                        channels=enc_channels_sp2[s - 1],
                        mlp_ratio=mlp_ratio,
                        drop=proj_drop,
                    )
        self.ds_1_sp2 = SerializedPooling(
                        in_channels=enc_channels_sp2[s - 1],
                        out_channels=enc_channels_sp2[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    )
        

        self.enc_1_module_sp1 = PointSequential()
        
        for i in range(enc_depths[s]):
                self.enc_1_module_sp1.add(
                    Block(
                        channels=enc_channels_sp1[s],
                        num_heads=enc_num_head_sp1[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                        feat_opt = 'attn',
                    ),
                    name=f"block{i}",
                )
        
        self.enc_1_module_sp2 = PointSequential()
        
        for i in range(enc_depths[s]):
                self.enc_1_module_sp2.add(
                    Block(
                        channels=enc_channels_sp2[s],
                        num_heads=enc_num_head_sp2[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                        feat_opt = 'attn',
                    ),
                    name=f"block{i}",
                )
        
        s = 2
        enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
        self.ds_2_sp1 = SerializedPooling(
                        in_channels=enc_channels_sp1[s - 1],
                        out_channels=enc_channels_sp1[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    )
        self.ds_2_sp2 = SerializedPooling(
                        in_channels=enc_channels_sp2[s - 1],
                        out_channels=enc_channels_sp2[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    )
        

        self.enc_2_module_sp1 = PointSequential()
        
        for i in range(enc_depths[s]):
                self.enc_2_module_sp1.add(
                    Block(
                        channels=enc_channels_sp1[s],
                        num_heads=enc_num_head_sp1[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                        feat_opt = 'attn',
                    ),
                    name=f"block{i}",
                )
        
        self.enc_2_module_sp2 = PointSequential()
        
        for i in range(enc_depths[s]):
                self.enc_2_module_sp2.add(
                    Block(
                        channels=enc_channels_sp2[s],
                        num_heads=enc_num_head_sp2[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                        feat_opt = 'attn',
                    ),
                    name=f"block{i}",
                )

        s = 3
        enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
        self.ds_3_sp1 = SerializedPooling(
                        in_channels=enc_channels_sp1[s - 1],
                        out_channels=enc_channels_sp1[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    )
        self.ds_3_sp2 = SerializedPooling(
                        in_channels=enc_channels_sp2[s - 1],
                        out_channels=enc_channels_sp2[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    )
        

        self.enc_3_module_sp1 = PointSequential()
        
        for i in range(enc_depths[s]):
                self.enc_3_module_sp1.add(
                    Block(
                        channels=enc_channels_sp1[s],
                        num_heads=enc_num_head_sp1[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                        feat_opt = 'attn',
                    ),
                    name=f"block{i}",
                )
        
        self.enc_3_module_sp2 = PointSequential()
        
        for i in range(enc_depths[s]):
                self.enc_3_module_sp2.add(
                    Block(
                        channels=enc_channels_sp2[s],
                        num_heads=enc_num_head_sp2[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                        feat_opt = 'attn',
                    ),
                    name=f"block{i}",
                )

        dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
        
        dec_channels_sp1 = list(dec_channels_sp1) + [enc_channels_sp1[-1]]
        dec_channels_sp2 = list(dec_channels_sp2) + [enc_channels_sp2[-1]]

        s = 2

        dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
        dec_drop_path_.reverse()

        

        self.us_2_sp1 = SerializedUnpooling(
                        in_channels=dec_channels_sp1[s + 1],
                        skip_channels=enc_channels_sp1[s],
                        out_channels=dec_channels_sp1[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    )
        self.us_2_sp2 = SerializedUnpooling(
                        in_channels=dec_channels_sp2[s + 1],
                        skip_channels=enc_channels_sp2[s],
                        out_channels=dec_channels_sp2[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    )
        
        self.dec_2_module_sp1 = PointSequential()
        
        for i in range(dec_depths[s]):
                    self.dec_2_module_sp1.add(
                        Block(
                            channels=dec_channels_sp1[s],
                            num_heads=dec_num_head_sp1[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                            feat_opt = 'attn',
                        ),
                        name=f"block{i}",
                    )
        self.dec_2_module_sp2 = PointSequential()
        
        for i in range(dec_depths[s]):
                    self.dec_2_module_sp2.add(
                        Block(
                            channels=dec_channels_sp2[s],
                            num_heads=dec_num_head_sp2[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                            feat_opt = 'attn',
                        ),
                        name=f"block{i}",
                    )

        s = 1

        dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
        dec_drop_path_.reverse()

        

        self.us_1_sp1 = SerializedUnpooling(
                        in_channels=dec_channels_sp1[s + 1],
                        skip_channels=enc_channels_sp1[s],
                        out_channels=dec_channels_sp1[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    )
        self.us_1_sp2 = SerializedUnpooling(
                        in_channels=dec_channels_sp2[s + 1],
                        skip_channels=enc_channels_sp2[s],
                        out_channels=dec_channels_sp2[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    )
        
        self.dec_1_module_sp1 = PointSequential()
        
        for i in range(dec_depths[s]):
                    self.dec_1_module_sp1.add(
                        Block(
                            channels=dec_channels_sp1[s],
                            num_heads=dec_num_head_sp1[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                            feat_opt = 'attn',
                        ),
                        name=f"block{i}",
                    )
        self.dec_1_module_sp2 = PointSequential()
        
        for i in range(dec_depths[s]):
                    self.dec_1_module_sp2.add(
                        Block(
                            channels=dec_channels_sp2[s],
                            num_heads=dec_num_head_sp2[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                            feat_opt = 'attn',
                        ),
                        name=f"block{i}",
                    )
        
        s = 0

        dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
        dec_drop_path_.reverse()

        self.us_0_sp1 = SerializedUnpooling(
                        in_channels=dec_channels_sp1[s + 1],
                        skip_channels=enc_channels_sp1[s],
                        out_channels=dec_channels_sp1[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    )
        self.us_0_sp2 = SerializedUnpooling(
                        in_channels=dec_channels_sp2[s + 1],
                        skip_channels=enc_channels_sp2[s],
                        out_channels=dec_channels_sp1[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    )
        
        self.dec_0_module_sp1 = PointSequential()
        
        for i in range(dec_depths[s]):
                    self.dec_0_module_sp1.add(
                        Block(
                            channels=dec_channels_sp1[s],
                            num_heads=dec_num_head_sp1[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                            feat_opt = 'attn',
                        ),
                        name=f"block{i}",
                    )
        self.dec_0_module_sp2 = PointSequential()
        
        for i in range(dec_depths[s]):
                    self.dec_0_module_sp2.add(
                        Block(
                            channels=dec_channels_sp2[s],
                            num_heads=dec_num_head_sp2[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                            feat_opt = 'attn',
                        ),
                        name=f"block{i}",
                    )

        self.skip_dec_sp1 = Embedding(
            in_channels=enc_channels_sp1[0],
            embed_channels= dec_channels_sp1[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        self.skip_dec_sp2 = Embedding(
            in_channels=enc_channels_sp2[0],
            embed_channels= dec_channels_sp2[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        feature_dim = dec_channels_sp2[0] + dec_channels_sp1[0]
        
        self.cls_head = nn.Sequential(
            nn.Linear(feature_dim, num_class)
        )

    def forward(self, points, neightbor_sp_idx_array):
        """
        A data_dict is a dictionary containing properties of a batched point cloud.
        It should contain the following properties for HSM3D:
        1. "feat": feature of point cloud
        2. "grid_coord": discrete coordinate after grid sampling (voxelization) or "coord" + "grid_size"
        3. "offset" or "batch": https://github.com/Pointcept/Pointcept?tab=readme-ov-file#offset
        """
 
        points_list = []
        points = points.permute(0, 2, 1)
        (B1,N1,C1) = points.shape
        # print(points.shape)
        device = points.device
        feat = points
        coord= points[:, :, :3]
        
        # 展平 feat 和 coord
        feat_flat = feat.reshape(B1 * N1, -1)     # [B*N, C]
        coord_flat = coord.reshape(B1 * N1, 3)    # [B*N, 3]
        # 创建 batch 张量
        batch = torch.arange(B1, dtype=torch.long).unsqueeze(1).repeat(1, N1).view(-1)  # [B*N]
    
        # 准备输入字典
        data_dict = {
            'feat': feat_flat.to(device),  # [B* N, 2]，nch和pch
            'coord': coord_flat.to(device), # [B* N, 3]，x, y, z
            'batch': batch.to(device),
            'grid_size':self.grid_size
        }

        point = Point(data_dict)
        # print(point.coord[:5])
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.embedding(point)
        point = self.enc_0_module(point)
        # print('enc_0: point_feat:', point.feat.shape)
        neightbor_idx_array_offset, spl_list_levels = reshape_with_dynamic_offset_fast(neightbor_sp_idx_array)  # [5, B*N, K]
        

#############################################################################################sp1#######################################
        # print(point.keys())
        # print(point.feat.shape)
        # print(point.coord[:5])


            

        with torch.no_grad():
            

                sp1, sp1_inverse = average_pool_by_sp(point.feat.detach(), neightbor_idx_array_offset[0,:,0])
                sp1_coord, _ = average_pool_by_sp(point.coord.detach(), neightbor_idx_array_offset[0,:,0])


        
        sp1_neighbor_idx = neightbor_idx_array_offset[0, :, 0]  # [B*N, K]
        # print('point feat:', point.feat.shape)
        # print('sp1 feat:', sp1.shape)
        # print('sp1_neighbor_idx:', sp1_neighbor_idx.shape)
        sp1_seq_tensor = build_sp_sequences_tensor(point.feat.detach(), sp1_neighbor_idx, sp1, self.sm_K)
        # print('sp1_seq_tensor:', sp1_seq_tensor.shape)
        sp1 = self.sm_sp1(sp1_seq_tensor)
        # print('after sm sp1:', sp1.shape)
        sp1 = self.gm_sp1(sp1.unsqueeze(0)).squeeze(0)
        spn_list = spl_list_levels[0]  # [B*N, K]
        sp1_labels = torch.arange(len(spn_list))         # tensor([0, 1, 2])
        batch = torch.repeat_interleave(sp1_labels, torch.tensor(spn_list)) # [B*N]

        # print('batch:', batch.shape)
        # print('sp1 feat:', sp1.shape)
        # print('sp1 coord:', sp1_coord.shape)

        
    
        # 准备输入字典
        sp1_data_dict = {
            'feat': sp1.to(device),  # [B* N, 2]，nch和pch
            'coord': sp1_coord.to(device), # [B* N, 3]，x, y, z
            'batch': batch.to(device),
            'grid_size':self.grid_size
        }

        sp1 = Point(sp1_data_dict)
        # print(point.coord[:5])
        sp1.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        sp1.sparsify()
        # print('enc_0: sp_feat:', sp.feat.shape)

        sp1 = self.ds_1_sp1(sp1)
        sp1 = self.enc_1_module_sp1(sp1)
        sp1 = self.ds_2_sp1(sp1)
        sp1 = self.enc_2_module_sp1(sp1)
        sp1 = self.ds_3_sp1(sp1)
        sp1 = self.enc_3_module_sp1(sp1)
        sp1 = self.us_2_sp1(sp1)
        sp1 = self.dec_2_module_sp1(sp1)
        sp1 = self.us_1_sp1(sp1)
        sp1 = self.dec_1_module_sp1(sp1)
        sp1 = self.us_0_sp1(sp1)
        sp1 = self.dec_0_module_sp1(sp1)

        if 'unpooling_parent' in sp1:
            del sp1['unpooling_parent']



        output_f = sp1.feat[sp1_inverse]
        residual_f = self.skip_dec_sp1(point).feat

        output_f = output_f + residual_f

        points_list.append(output_f)

        point.feat = output_f
#############################################################################################sp2#######################################
        with torch.no_grad():
    

            sp2, sp2_inverse = average_pool_by_sp(point.feat.detach(), neightbor_idx_array_offset[1,:,0])
            sp2_coord, _ = average_pool_by_sp(point.coord.detach(), neightbor_idx_array_offset[1,:,0])


        
        sp2_neighbor_idx = neightbor_idx_array_offset[1, :, 0]  # [B*N, K]
        sp2_seq_tensor = build_sp_sequences_tensor(point.feat.detach(), sp2_neighbor_idx, sp2, self.sm_K)
        sp2 = self.sm_sp2(sp2_seq_tensor)
        sp2 = self.gm_sp2(sp2.unsqueeze(0)).squeeze(0)
        spn_list = spl_list_levels[1]  # [B*N, K]
        sp2_labels = torch.arange(len(spn_list))         # tensor([0, 1, 2])
        batch = torch.repeat_interleave(sp2_labels, torch.tensor(spn_list)) # [B*N]
    
        # 准备输入字典
        sp2_data_dict = {
            'feat': sp2.to(device),  # [B* N, 2]，nch和pch
            'coord': sp2_coord.to(device), # [B* N, 3]，x, y, z
            'batch': batch.to(device),
            'grid_size':self.grid_size
        }

        sp2 = Point(sp2_data_dict)
        # print(point.coord[:5])
        sp2.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        sp2.sparsify()
        # print('enc_0: sp_feat:', sp.feat.shape)

        sp2 = self.ds_1_sp2(sp2)
        sp2 = self.enc_1_module_sp2(sp2)
        sp2 = self.ds_2_sp2(sp2)
        sp2 = self.enc_2_module_sp2(sp2)
        sp2 = self.ds_3_sp2(sp2)
        sp2 = self.enc_3_module_sp2(sp2)
        sp2 = self.us_2_sp2(sp2)
        sp2 = self.dec_2_module_sp2(sp2)
        sp2 = self.us_1_sp2(sp2)
        sp2 = self.dec_1_module_sp2(sp2)
        sp2 = self.us_0_sp2(sp2)
        sp2 = self.dec_0_module_sp2(sp2)


        if 'unpooling_parent' in sp2:
            del sp2['unpooling_parent']


        # inverse sp.feat to point.feat
        output_f = sp2.feat[sp2_inverse]
        residual_f = self.skip_dec_sp2(point).feat

        output_f = output_f + residual_f

        points_list.append(output_f)

        point.feat = output_f

        



        logits = self.cls_head(torch.cat(points_list, dim=-1))
        _clean_point_cache(point)
        _clean_point_cache(sp1)
        _clean_point_cache(sp2)
        del point
        del sp1
        del sp2
        gc.collect()
        torch.cuda.empty_cache()
        return logits.reshape(B1, N1, -1)



class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, gold, weight, smoothing=True):
        gold = gold.contiguous().view(-1)

        if smoothing:
            eps = 0.2
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, gold, reduction='mean')

        return loss

class get_loss_weighted(nn.Module):
    def __init__(self):
      super(get_loss_weighted, self).__init__()
    def forward(self, pred, target, weight):
        total_loss = F.cross_entropy(pred, target, weight)
        #total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss