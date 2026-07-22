import importlib
import torch.utils.checkpoint as checkpoint

from .common_utils_mbyolo import *
from .legnet import EdgeEnhancingStem, Gaussian, LFEA, Scharr

__all__ = ("VSSBlock", "EdgeVSSBlock", "DVSSBlock", "SimpleStem", "EdgeStem", "VisionClueMerge", "XSSBlock")


class SS2D(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # ======================
            forward_type="v2",
            **kwargs,
    ):
        """
        ssm_rank_ratio would be used in the future...
        """
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * d_model)
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.d_conv = d_conv
        self.K = 4
        self.allow_cpu_fallback_for_build = False

        # tags for forward_type ==============================
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("no32", forward_type)
        self.disable_z, forward_type = checkpostfix("noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("nozact", forward_type)

        self.out_norm = nn.LayerNorm(d_inner)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v2=partial(self.forward_corev2, force_fp32=None, SelectiveScan=SelectiveScanCore),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, FORWARD_TYPES.get("v2", None))

        # in proj =======================================
        d_proj = d_expand if self.disable_z else (d_expand * 2)
        self.in_proj = nn.Conv2d(d_model, d_proj, kernel_size=1, stride=1, groups=1, bias=bias, **factory_kwargs)
        self.act: nn.Module = nn.GELU()

        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # rank ratio =====================================
        self.ssm_low_rank = False
        if d_inner < d_expand:
            self.ssm_low_rank = True
            self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False, **factory_kwargs)
            self.out_rank = nn.Linear(d_inner, d_expand, bias=False, **factory_kwargs)

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False,
                      **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_proj = nn.Conv2d(d_expand, d_model, kernel_size=1, stride=1, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        # simple init dt_projs, A_logs, Ds
        self.Ds = nn.Parameter(torch.ones((self.K * d_inner)))
        self.A_logs = nn.Parameter(
            torch.zeros((self.K * d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
        self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
        self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner)))

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev2(self, x: torch.Tensor, channel_first=False, SelectiveScan=SelectiveScanCore,
                       cross_selective_scan=cross_selective_scan, force_fp32=None):
        force_fp32 = (self.training and (not self.disable_force32)) if force_fp32 is None else force_fp32
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.ssm_low_rank:
            x = self.in_rank(x)
        x = cross_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds,
            out_norm=getattr(self, "out_norm", None),
            out_norm_shape=getattr(self, "out_norm_shape", "v0"),
            delta_softplus=True, force_fp32=force_fp32,
            SelectiveScan=SelectiveScan, ssoflex=self.training,  # output fp32
            allow_cpu_fallback_for_build=self.allow_cpu_fallback_for_build,
        )
        if self.ssm_low_rank:
            x = self.out_rank(x)
        return x

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=1)  # (b, d, h, w)
            if not self.disable_z_act:
                z1 = self.act(z)
        if self.d_conv > 0:
            x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)
        y = self.forward_core(x, channel_first=(self.d_conv > 1))
        y = y.permute(0, 3, 1, 2).contiguous()
        if not self.disable_z:
            y = y * z1
        out = self.dropout(self.out_proj(y))
        return out


class RGBlock(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=True,
                                groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.act(self.dwconv(x) + x) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LSBlock(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0):
        super().__init__()
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=3, padding=3 // 2, groups=hidden_features)
        self.norm = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=1, padding=0)
        self.act = act_layer()
        self.fc3 = nn.Conv2d(hidden_features, in_features, kernel_size=1, padding=0)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        input = x
        x = self.fc1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        x = input + self.drop(x)
        return x


class Linear2d(nn.Linear):
    """Linear layer applied as a 1x1 convolution for BCHW tensors."""

    def forward(self, x: torch.Tensor):
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        weight_key = prefix + "weight"
        if weight_key in state_dict and state_dict[weight_key].shape != self.weight.shape:
            state_dict[weight_key] = state_dict[weight_key].view(self.weight.shape)
        return super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0,
                 channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class gMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0,
                 channels_first=False):
        super().__init__()
        self.channels_first = channels_first
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, 2 * hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x, z = self.fc1(x).chunk(2, dim=(1 if self.channels_first else -1))
        x = self.fc2(x * self.act(z))
        return self.drop(x)


def _split_channels(channels, num_groups):
    split_channels = [channels // num_groups for _ in range(num_groups)]
    split_channels[0] += channels - sum(split_channels)
    return split_channels


def _channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.size()
    if num_channels % groups != 0:
        raise ValueError(f"num_channels={num_channels} must be divisible by groups={groups}.")
    channels_per_group = num_channels // groups
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    return x.view(batch_size, num_channels, height, width)


def _import_dcnv4():
    try:
        return importlib.import_module("DCNv4").DCNv4, None
    except Exception as exc:
        return None, exc


class DCNv4Spatial(nn.Module):
    """BCHW adapter around the DCNv4 token-layout module."""

    def __init__(self, channels, group, d_model=None, kernel_size=3, offset_scale=1.0, output_bias=False):
        super().__init__()
        self.allow_cpu_fallback_for_build = False
        self.channels = channels
        self.group = group
        dw_kernel_size = 3 if d_model is not None and d_model % 80 == 0 else None
        DCNv4, import_error = _import_dcnv4()
        self._dcnv4_import_error = import_error
        self.op = None
        if DCNv4 is not None:
            self.op = DCNv4(
                channels=channels,
                kernel_size=kernel_size,
                pad=(kernel_size - 1) // 2,
                group=group,
                offset_scale=offset_scale,
                dw_kernel_size=dw_kernel_size,
                output_bias=output_bias,
            )

    def _missing_dcnv4_error(self):
        return ImportError(
            "DVSSBlock requires the DCNv4 extension for CUDA execution. Build it with "
            "'cd dcnv4 && python setup.py build install' inside the active environment."
        )

    def forward(self, x):
        if not x.is_cuda:
            if self.allow_cpu_fallback_for_build:
                return x
            if self.op is None:
                raise self._missing_dcnv4_error() from self._dcnv4_import_error
            raise RuntimeError(
                "DCNv4 CPU fallback is available only during model construction stride probing. "
                "Move the model to CUDA for training or inference."
            )
        if self.op is None:
            raise self._missing_dcnv4_error() from self._dcnv4_import_error

        b, c, h, w = x.shape
        if c != self.channels:
            raise ValueError(f"DCNv4Spatial expected {self.channels} channels, got {c}.")
        tokens = x.permute(0, 2, 3, 1).reshape(b, h * w, c).contiguous()
        tokens = self.op(tokens, shape=(h, w))
        return tokens.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()


class DSS2D(nn.Module):
    """Deformable 2D selective scan used by HRVMamba DVSS."""

    def __init__(
            self,
            d_model=96,
            d_state=1,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            d_conv=3,
            conv_bias=False,
            dropout=0.0,
            bias=False,
            initialize="v0",
            forward_type="v05_noz",
            **kwargs,
    ):
        super().__init__()
        factory_kwargs = {"device": None, "dtype": None}
        d_expand = int(ssm_ratio * d_model)
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state
        self.d_conv = d_conv
        self.K = 4
        self.allow_cpu_fallback_for_build = False

        def checkpostfix(tag, value):
            ret = value.endswith(tag)
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        forward_options = {
            "v2": (None, False),
            "v05": (False, True),
        }
        if forward_type not in forward_options:
            raise ValueError(f"DSS2D unsupported forward_type '{forward_type}'.")
        self.force_fp32, self.no_einsum = forward_options[forward_type]

        d_proj = d_expand if self.disable_z else d_expand * 2
        self.in_proj = Linear2d(d_model, d_proj, bias=bias, **factory_kwargs)
        self.act = act_layer()
        if self.d_conv > 1:
            self.conv2d = DCNv4Spatial(
                channels=d_expand,
                group=self.dt_rank,
                d_model=d_model,
                kernel_size=d_conv,
                output_bias=bias,
            )

        self.ssm_low_rank = False
        if d_inner < d_expand:
            self.ssm_low_rank = True
            self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False, **factory_kwargs)
            self.out_rank = nn.Linear(d_inner, d_expand, bias=False, **factory_kwargs)

        self.out_norm = nn.LayerNorm(d_inner)
        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        if initialize == "v0":
            dt_projs = [
                SS2D.dt_init(self.dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1,
                             dt_init_floor=1e-4, **factory_kwargs)
                for _ in range(self.K)
            ]
            self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0))
            self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0))
            del dt_projs
            self.A_logs = SS2D.A_log_init(self.d_state, d_inner, copies=self.K, merge=True)
            self.Ds = SS2D.D_init(d_inner, copies=self.K, merge=True)
        elif initialize == "v1":
            self.Ds = nn.Parameter(torch.ones((self.K * d_inner)))
            self.A_logs = nn.Parameter(torch.randn((self.K * d_inner, self.d_state)))
            self.dt_projs_weight = nn.Parameter(0.1 * torch.randn((self.K, d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((self.K, d_inner)))
        elif initialize == "v2":
            self.Ds = nn.Parameter(torch.ones((self.K * d_inner)))
            self.A_logs = nn.Parameter(torch.zeros((self.K * d_inner, self.d_state)))
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((self.K, d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((self.K, d_inner)))
        else:
            raise ValueError(f"DSS2D unsupported initialization '{initialize}'.")

        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear2d(d_expand, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forward_core(self, x):
        force_fp32 = (self.training and (not self.disable_force32)) if self.force_fp32 is None else self.force_fp32
        if self.ssm_low_rank:
            x = self.in_rank(x)
        x = cross_selective_scan(
            x,
            self.x_proj_weight,
            None,
            self.dt_projs_weight,
            self.dt_projs_bias,
            self.A_logs,
            self.Ds,
            out_norm=self.out_norm,
            out_norm_shape=getattr(self, "out_norm_shape", "v0"),
            delta_softplus=True,
            force_fp32=force_fp32,
            SelectiveScan=SelectiveScanCore,
            ssoflex=self.training,
            allow_cpu_fallback_for_build=self.allow_cpu_fallback_for_build,
            no_einsum=self.no_einsum,
        )
        if self.ssm_low_rank:
            x = self.out_rank(x)
        return x

    def forward(self, x):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=1)
            z = z if self.disable_z_act else self.act(z)
        if self.d_conv > 1:
            self.conv2d.allow_cpu_fallback_for_build = self.allow_cpu_fallback_for_build
            x = self.conv2d(x)
        x = self.act(x)
        y = self.forward_core(x)
        y = self.out_act(y).permute(0, 3, 1, 2).contiguous()
        if not self.disable_z:
            y = y * z
        return self.dropout(self.out_proj(y))


class DVSSBlock(nn.Module):
    """HRVMamba Dynamic Visual State Space block with an identity residual path."""

    def __init__(
            self,
            in_channels: int = 0,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(LayerNorm2d, eps=1e-5),
            ssm_d_state: int = 1,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=False,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v05_noz",
            mlp_ratio=2.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            gmlp=False,
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        if in_channels != hidden_dim:
            raise ValueError(
                f"DVSSBlock requires in_channels == hidden_dim for its identity residual, got "
                f"{in_channels} and {hidden_dim}. Use a transition layer before the block."
            )
        self.proj_conv = nn.Identity()

        self.num_groups = 4
        self.split_channels = _split_channels(hidden_dim, self.num_groups)
        self.esinb_convs = nn.ModuleList(
            nn.Conv2d(c, c, kernel_size=i * 2 + 3, padding=i + 1, groups=c)
            for i, c in enumerate(self.split_channels)
        )
        self.norm0 = norm_layer(hidden_dim)
        self.act0 = nn.GELU()

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = DSS2D(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                forward_type=forward_type,
            )

        self.drop_path = DropPath(drop_path)
        if self.mlp_branch:
            _MLP = gMlp if gmlp else Mlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = _MLP(
                in_features=hidden_dim,
                hidden_features=mlp_hidden_dim,
                act_layer=mlp_act_layer,
                drop=mlp_drop_rate,
                channels_first=True,
            )

        # Match HRVMamba's explicit initialization for Linear2d layers without
        # touching the state-space parameters' specialized initialization.
        reference_linears = []
        if self.ssm_branch:
            reference_linears.extend((self.op.in_proj, self.op.out_proj))
        if self.mlp_branch:
            reference_linears.extend((self.mlp.fc1, self.mlp.fc2))
        for module in reference_linears:
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _forward(self, input):
        if self.post_norm:
            x_split = torch.split(input, self.split_channels, dim=1)
            x = [conv(t) for conv, t in zip(self.esinb_convs, x_split)]
            x = _channel_shuffle(torch.cat(x, dim=1), self.num_groups)
            x = input + self.drop_path(self.act0(self.norm0(x)))
        else:
            x = self.norm0(input)
            x_split = torch.split(x, self.split_channels, dim=1)
            x = [conv(t) for conv, t in zip(self.esinb_convs, x_split)]
            x = _channel_shuffle(torch.cat(x, dim=1), self.num_groups)
            x = input + self.drop_path(self.act0(x))

        if self.ssm_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm(self.op(x)))
            else:
                x = x + self.drop_path(self.op(self.norm(x)))
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, input):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        return self._forward(input)


class XSSBlock(nn.Module):
    def __init__(
            self,
            in_channels: int = 0,
            hidden_dim: int = 0,
            n: int = 1,
            mlp_ratio=4.0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(LayerNorm2d, eps=1e-6),
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()

        self.in_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        ) if in_channels != hidden_dim else nn.Identity()
        self.hidden_dim = hidden_dim
        # ==========SSM============================
        self.norm = norm_layer(hidden_dim)
        self.ss2d = nn.Sequential(*(SS2D(d_model=self.hidden_dim,
                                         d_state=ssm_d_state,
                                         ssm_ratio=ssm_ratio,
                                         ssm_rank_ratio=ssm_rank_ratio,
                                         dt_rank=ssm_dt_rank,
                                         act_layer=ssm_act_layer,
                                         d_conv=ssm_conv,
                                         conv_bias=ssm_conv_bias,
                                         dropout=ssm_drop_rate, ) for _ in range(n)))
        self.drop_path = DropPath(drop_path)
        self.lsblock = LSBlock(hidden_dim, hidden_dim)
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = RGBlock(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                               drop=mlp_drop_rate)

    def forward(self, input):
        input = self.in_proj(input)
        # ====================
        X1 = self.lsblock(input)
        input = input + self.drop_path(self.ss2d(self.norm(X1)))
        # ===================
        if self.mlp_branch:
            input = input + self.drop_path(self.mlp(self.norm2(input)))
        return input


class VSSBlock(nn.Module):
    def __init__(
            self,
            in_channels: int = 0,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(LayerNorm2d, eps=1e-6),
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        # proj
        self.proj_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        )

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SS2D(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
            )

        self.drop_path = DropPath(drop_path)
        self.lsblock = LSBlock(hidden_dim, hidden_dim)
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = RGBlock(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                               drop=mlp_drop_rate, channels_first=False)

    def forward(self, input: torch.Tensor):
        input = self.proj_conv(input)
        X1 = self.lsblock(input)
        x = input + self.drop_path(self.op(self.norm(X1)))
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
        return x


class EdgeVSSBlock(nn.Module):
    def __init__(
            self,
            in_channels: int = 0,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(LayerNorm2d, eps=1e-6),
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        self.proj_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        )

        self.lsblock = LSBlock(hidden_dim, hidden_dim)
        self.lfea = LFEA(hidden_dim, nn.ReLU)
        self.edge = Scharr(hidden_dim, nn.ReLU) if hidden_dim <= 128 else Gaussian(hidden_dim, 5, 1.0, nn.ReLU)

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SS2D(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                forward_type=forward_type,
            )

        self.drop_path = DropPath(drop_path)
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = RGBlock(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                               drop=mlp_drop_rate, channels_first=False)

    def forward(self, input: torch.Tensor):
        input = self.proj_conv(input)
        x_local = self.lsblock(input)
        x_edge = self.lfea(x_local, self.edge(x_local))
        x = input + self.drop_path(self.op(self.norm(x_edge)))
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
        return x


class SimpleStem(nn.Module):
    def __init__(self, inp, embed_dim, ks=3):
        super().__init__()
        self.hidden_dims = embed_dim // 2
        self.conv = nn.Sequential(
            nn.Conv2d(inp, self.hidden_dims, kernel_size=ks, stride=2, padding=autopad(ks, d=1), bias=False),
            nn.BatchNorm2d(self.hidden_dims),
            nn.GELU(),
            nn.Conv2d(self.hidden_dims, embed_dim, kernel_size=ks, stride=2, padding=autopad(ks, d=1), bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.conv(x)


class EdgeStem(EdgeEnhancingStem):
    def __init__(self, inp, embed_dim, ks=3):
        if ks != 3:
            raise ValueError(f"EdgeStem only supports ks=3, got ks={ks}.")
        super().__init__(inp, embed_dim, nn.ReLU)


class VisionClueMerge(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.hidden = int(dim * 4)

        self.pw_linear = nn.Sequential(
            nn.Conv2d(self.hidden, out_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_dim),
            nn.SiLU()
        )

    def forward(self, x):
        y = torch.cat([
            x[..., ::2, ::2],
            x[..., 1::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, 1::2]
        ], dim=1)
        return self.pw_linear(y)
