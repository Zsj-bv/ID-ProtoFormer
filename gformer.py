import torch
import torch.nn.functional as F

from mne.utils import warn
from einops import rearrange
from torch import nn, Tensor
from einops.layers.torch import Rearrange
from braindecode.models.base import EEGModuleMixin


class _FeedForwardBlock(nn.Module):
    def __init__(
        self,
        emb_size: int,
        expansion: int,
        drop_p: float,
        activation: nn.Module = nn.GELU,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_size, expansion * emb_size),
            activation(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)



class _MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)

        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)

        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)

        out = torch.einsum("bhal, bhlv -> bhav ", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class _ResidualAdd(nn.Module):
    def __init__(self, module: nn.Module, emb_size: int, drop_p: float):
        super().__init__()
        self.module = module
        self.drop = nn.Dropout(drop_p)
        self.layernorm = nn.LayerNorm(emb_size)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        res = self.module(x, **kwargs)
        out = self.layernorm(self.drop(res) + x)
        return out

class FeatureAggregationDenoise(nn.Module):
    def __init__(
        self,
        tokens: int,
        emb_size: int,
        smooth_kernel: int = 5,
        attn_reduction: int = 4,
    ):
        super().__init__()
        padding = (smooth_kernel - 1) // 2
        self.smooth = nn.Conv1d(
            in_channels=emb_size,
            out_channels=emb_size,
            kernel_size=smooth_kernel,
            padding=padding,
            groups=emb_size,  # depthwise
            bias=False,
        )
        self.norm = nn.LayerNorm(emb_size)
        self.attn = nn.Sequential(
            nn.Linear(emb_size, emb_size // attn_reduction, bias=False),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(emb_size // attn_reduction, emb_size, bias=False),
            nn.Sigmoid(),
            nn.Dropout(0.5)
        )

    def forward(self, x: Tensor) -> Tensor:
        B, T, E = x.shape
        y = x.transpose(1, 2)
        y = self.smooth(y)
        y = y.transpose(1, 2)
        y = self.norm(y)
        w = self.attn(y)
        return x + y * w
    
class Gformer(EEGModuleMixin, nn.Module):
    def __init__(
        self,
        n_outputs=None,
        n_chans=None,
        sfreq=None,
        chs_info=None,
        n_times=None,
        input_window_seconds=None,
        activation: nn.Module = nn.GELU,
        drop_prob_cnn: float = 0.5,
        drop_prob_posi: float = 0.5,
        drop_prob_final: float = 0.5,
        heads: int = 5,
        emb_size: int = 50,
        depth: int = 1,
        kernel_size: int = 64,
        depth_multiplier: int = 1,
        pool_size: int = 5,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )

        self.emb_size = emb_size
        self.activation = activation

        self.drop_prob_cnn = drop_prob_cnn
        self.pool_size = pool_size
        self.depth_multiplier = depth_multiplier
        self.kernel_size = kernel_size
        self.drop_prob_posi = drop_prob_posi
        self.drop_prob_final = drop_prob_final

        self.flatten = nn.Flatten()

        self.ensuredim = Rearrange("b c t -> b c 1 t")

        self.cnn = _PatchEmbeddingEEGNet(
            kernel_size=self.kernel_size,
            depth_multiplier=self.depth_multiplier,
            pool_size=self.pool_size,
            drop_prob=self.drop_prob_cnn,
            n_chans=self.n_chans,
            n_times=self.n_times,
            emb_size=self.emb_size,
            activation=self.activation,
        )

        self.position = nn.Parameter(torch.randn(1, n_chans, 1))

        self.trans = _TransformerEncoder(heads, depth, emb_size, activation=self.activation)

        self.agg_denoise = FeatureAggregationDenoise(
            tokens=self.n_chans,
            emb_size=self.emb_size,
            smooth_kernel=5,
            attn_reduction=4,
        )
        
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=emb_size * n_chans, out_features=self.n_outputs)            
        )

        self.center_mgr = CenterManager(
            num_subjects=n_outputs,
            n_chans=n_chans,
            feat_dim=emb_size,
            base_momentum=0.9,
            beta=8.0,
            conf_thresh=0.7,
            device="cpu",
        )

        self.proto_alpha = nn.Parameter(torch.tensor(0.5))
        
    def _get_device(self):
        return next(self.parameters()).device
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.ensuredim(x)

        cnn_feat = self.cnn(x)
        pos_feat = cnn_feat + self.position
        trans_out = self.trans(pos_feat)
        # residual
        features = pos_feat + trans_out
        features = self.agg_denoise(features)

        feat = features.flatten(1)

        use_proto = (not self.training)
        if use_proto:
            feat_norm = F.normalize(feat, dim=1)
            centers = self.center_mgr.centers.flatten(1)
            centers = F.normalize(centers, dim=1)
            proto_logits = feat_norm @ centers.t() / 0.1
            proto_prob   = F.softmax(proto_logits, dim=1)
            weighted_proto = proto_prob @ centers
            feat_to_classify = feat + self.proto_alpha * weighted_proto
        else:
            feat_to_classify = feat
        logits = self.fc_layer(feat_to_classify)


        if self.training:
            return logits, features
        else:
            return logits

class _PatchEmbeddingEEGNet(nn.Module):
    def __init__(
        self,
        kernel_size: int = 64,
        depth_multiplier: int = 1,
        pool_size: int = 5,
        drop_prob: float = 0.5,
        n_chans: int = 62,
        n_times: int = 500,
        emb_size: int = 128,
        activation: nn.Module = nn.ELU,
    ):
        super().__init__()
        n_filters_out = depth_multiplier * n_chans

        self.spatial_emb = nn.Conv2d(in_channels=n_chans, out_channels=n_filters_out, kernel_size=(1, kernel_size), padding='same')
        self.spatial_norm = nn.BatchNorm2d(n_filters_out)

        self.spatial_tempral_emb = nn.Conv2d(in_channels=n_filters_out, out_channels=n_chans, kernel_size=(1, 1), groups=n_chans)

        self.activate = activation()
        self.pooled = nn.AdaptiveAvgPool2d((1, n_times // pool_size))

        self.drop = nn.Dropout(drop_prob)
        self.drop1 = nn.Dropout(drop_prob)

        self.emb_fc = nn.Linear(n_times // pool_size, emb_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.spatial_emb(x)
        x = self.spatial_norm(x)

        x = self.spatial_tempral_emb(x)
        x = self.drop(self.pooled(self.activate(x)))

        x = self.emb_fc(x)
        x = self.drop1(self.activate(x))

        return rearrange(x, "b c 1 emb -> b c emb")
    
class _TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        dim_feedforward: int,
        num_heads: int = 4,
        drop_prob: float = 0.5,
        forward_expansion: int = 4,
        forward_drop_p: float = 0.5,
        activation: nn.Module = nn.GELU,

    ):
        super().__init__()
        self.attention = _ResidualAdd(
            _MultiHeadAttention(dim_feedforward, num_heads, drop_prob),
            dim_feedforward,
            drop_prob,
        )
        self.feed_forward = _ResidualAdd(
            _FeedForwardBlock(
                dim_feedforward,
                expansion=forward_expansion,
                drop_p=forward_drop_p,
                activation=activation,
            ),
            dim_feedforward,
            drop_prob,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.attention(x)
        x = self.feed_forward(x)
        return x

class _TransformerEncoder(nn.Module):
    def __init__(
        self,
        nheads: int,
        depth: int,
        dim_feedforward: int,
        drop_prob: float = 0.5,
        forward_expansion: int = 4,
        forward_drop_p: float = 0.5,
        activation: nn.Module = nn.GELU,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [ _TransformerEncoderBlock(dim_feedforward=dim_feedforward,
                            num_heads=nheads,
                            drop_prob=drop_prob,
                            forward_expansion=forward_expansion,
                            forward_drop_p=forward_drop_p,
                            activation=activation,) for _ in range(depth) ]
            )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

class CenterManager:
    def __init__(self, num_subjects: int, n_chans: int, feat_dim: int,
                 base_momentum: float = 0.9, beta: float = 5.0,
                 conf_thresh: float = 0.7, device: str = 'cuda'):
        self.base_m = base_momentum
        self.beta   = beta
        self.tau    = conf_thresh
        self.device = device
        self.centers = torch.zeros(num_subjects, n_chans, feat_dim, device=device)
        self.initialized = torch.zeros(num_subjects, dtype=torch.bool, device=device)

    @torch.no_grad()
    def update(self, feats: torch.Tensor, labels: torch.Tensor, logits: torch.Tensor):
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        mask_conf = (pred == labels) & (conf > self.tau)
        if mask_conf.sum() == 0:
            return
        for s in torch.unique(labels):
            idxs = (labels == s) & mask_conf
            if idxs.sum() == 0:
                continue
            feats_s = feats[idxs]       
            m_feat = feats_s.mean(dim=0)
            var_s = feats_s.var(dim=0, unbiased=False).mean()
            alpha_s = torch.exp(-self.beta * var_s).item()
            alpha_s = alpha_s * (1 - self.base_m) + self.base_m
            s_int = s.item()
            if not self.initialized[s_int]:
                self.centers[s_int] = m_feat
                self.initialized[s_int] = True
            else:
                c_old = self.centers[s_int]
                self.centers[s_int] = alpha_s * c_old + (1 - alpha_s) * m_feat

    def get(self, labels: torch.Tensor) -> torch.Tensor:
        return self.centers[labels]
