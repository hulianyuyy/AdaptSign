from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, checkpointing=False):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.S_Adapter = Adapter(d_model, skip_connect=False)
        #self.S_Adapter = Adapter(d_model)
        self.MLP_Adapter = Adapter(d_model, skip_connect=False)
        #self.MLP_Adapter = Adapter(d_model)

        scale = d_model ** -0.5
        self.prefix_length = 8
        self.prefix_embedding_k = nn.Parameter(scale * torch.randn(self.prefix_length, 1, d_model))
        self.prefix_embedding_v = nn.Parameter(scale * torch.randn(self.prefix_length, 1, d_model))
        self.checkpointing = checkpointing

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, torch.concat((x, self.prefix_embedding_k.repeat(1, x.shape[1], 1)), 0), torch.concat((x, self.prefix_embedding_v.repeat(1, x.shape[1], 1)), 0), need_weights=False, attn_mask=self.attn_mask)[0]
        #return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        if self.checkpointing:
            x = x + checkpoint(self.attention, checkpoint(self.ln_1,x)) + self.S_Adapter(x)
        else:
            x = x + self.attention(self.ln_1(x)) + self.S_Adapter(x)
        #x = x + self.attention(self.ln_1(x)) + self.S_Adapter(x)
        #x = x + self.mlp(self.ln_2(x))
        #x = x + self.mlp(self.ln_2(x)) + self.MLP_Adapter(x)
        if self.checkpointing:
             x = x + checkpoint(self.mlp, checkpoint(self.ln_2,x)) + self.MLP_Adapter(x)
        else:
            x = x + self.mlp(self.ln_2(x)) + self.MLP_Adapter(x)
        return x

        '''x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x'''

class Adapter(nn.Module):
    def __init__(self, d_model: int, skip_connect=True):
        super().__init__()
        r = 1/4
        self.mlp = nn.Sequential(OrderedDict([
            ("c_in", nn.Linear(d_model, int(d_model * r))),
            ("gelu", QuickGELU()),
            ("c_out", nn.Linear( int(d_model * r), d_model))
        ]))
        self.skip_connect = skip_connect

    def forward(self, x: torch.Tensor):
        if self.skip_connect:
            return self.mlp(x)+x
        else:
            return self.mlp(x)
        
class AggregationBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 1)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 1, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x[:1], x[1:], x[1:], need_weights=False, attn_mask=self.attn_mask)[0]   # cls token attends to others

    def forward(self, x: torch.Tensor, cls):
        cls = cls + self.attention(self.ln_1(torch.concat((cls, x), 0)))
        cls = cls + self.mlp(self.ln_2(cls))
        return cls

class UnfoldTemporalWindows(nn.Module):
    def __init__(self, window_size=5, window_stride=1, window_dilation=1):
        super().__init__()
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_dilation = window_dilation

        self.padding = (window_size + (window_size-1) * (window_dilation-1) - 1) // 2
        self.unfold = nn.Unfold(kernel_size=(self.window_size, 1),
                                dilation=(self.window_dilation, 1),
                                stride=(self.window_stride, 1),
                                padding=(self.padding, 0))

    def forward(self, x, T):
        # Input shape: (N,C,T,H,W), out: (N,C,T,V*window_size)
        NT, C, P = x.shape
        x = x.view(-1, T, C, P).permute(0,2,1,3)
        x = self.unfold(x)  #(N, C*Window_Size, T, P)
        # Permute extra channels from window size to the graph dimension; -1 for number of windows
        #x = x.view(-1, C, self.window_size, T, P).permute(0,3,1,2,4).reshape(NT, C, -1)# (NT)C(SP)
        x = x.view(-1, C, self.window_size, T, P)
        # wo_current
        #x = torch.concat((x[:,:,:(self.window_size-1)//2], x[:,:,(self.window_size-1)//2+1:]), 2).permute(0,3,1,2,4).reshape(NT, C, -1)
        # normal
        x = x.view(-1, C, self.window_size, T, P).permute(0,3,1,2,4).reshape(NT, C, -1)# (NT)C(SP)
        return x

class Correlation_Module(nn.Module):
    def __init__(self, neighbors=5):
        super().__init__()

        #self.down_conv = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        #self.weights = nn.Parameter(torch.ones(neighbors) / neighbors, requires_grad=True)

        #self.spatial_pos_embedding = nn.Parameter(torch.randn(1, channels, 1, spacial_dim, spacial_dim) / channels ** 0.5)
        #self.temporal_pos_embedding = nn.Parameter(torch.randn(1, channels, self.neighbors * 2, 1, 1) / channels ** 0.5)

    def forward(self, x, upfold):

        #x_mean = self.attpool(x) 
        #x2 = self.down_conv(x)
        L, N, D = x.shape
        import math
        affinities = torch.einsum('lnd,ond->lon', x, upfold)/math.sqrt(D)
        features = torch.einsum('lon,ond->lnd', F.sigmoid(affinities)-0.5, upfold)

        return features

class TemporalAggregationBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        #self.attn = nn.MultiheadAttention(d_model, n_head)
        self.attn = Correlation_Module()
        self.ln_1 = LayerNorm(d_model)
        '''self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 1)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 1, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)'''
        self.attn_mask = attn_mask
        self.upfold = UnfoldTemporalWindows(5)

    def attention(self, x: torch.Tensor, T):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        x_upfold = self.upfold(x[1:].permute(1,2,0), T).permute(2,0,1)   #LND -> NDL -> LND
        #return self.attn(x[:1], x_upfold, x_upfold, need_weights=False, attn_mask=self.attn_mask)[0]   # cls token attends to others
        return self.attn(x[:1], x_upfold)   # cls token attends to others

    def forward(self, x: torch.Tensor, T):
        #cls = x[:1]
        #cls = cls + self.attention(self.ln_1(x), T)
        cls = self.attention(self.ln_1(x), T)
        #cls = cls + self.mlp(self.ln_2(cls))
        return cls

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        def backward_hook(module, grad_in, grad_out):
            print(grad_in[0].shape)  #LND 
            print(grad_in[0][:2,:3])
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, checkpointing= i>=0) for i in range(layers)])
        self.query = nn.Parameter(torch.rand(1,1,width), requires_grad=True)
        self.aggblocks = nn.Sequential(*[AggregationBlock(width, heads, attn_mask) for _ in range(layers)])  # 14 for ViT-B/16
        #self.ada_weight = nn.Parameter(torch.zeros(width), requires_grad=True)
        #self.resblocks[-2].mlp.register_backward_hook(backward_hook)
        self.taggblocks = TemporalAggregationBlock(width, heads, attn_mask)
        self.temporal_ada_weight = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x: torch.Tensor, T=None):
        L, N, D = x.shape
        query = self.query.repeat(1,N,1)
        for i in range(len(self.resblocks)):
            x = self.resblocks[i](x)
            #query = self.aggblocks[i](x, query)
            query = checkpoint(self.aggblocks[i], x, query)
        #x = x + self.taggblocks(x, T) * self.temporal_ada_weight
        x = x + checkpoint(self.taggblocks, x, T) * self.temporal_ada_weight
        return x, query
        '''for i in range(len(self.resblocks)):
            x = self.ada_weight * self.aggblocks[i](x, query) + self.resblocks[i](x) 
        return x'''
        #return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)
        #self.spatial_enhance = Transformer(width, 1, heads, Nograd=False)

        self.ln_post = LayerNorm(width)
        self.ln_post_cls = LayerNorm(width)
        #self.proj = None
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        #self.proj_cls = nn.Parameter(scale * torch.randn(width, output_dim))
        #self.weight = nn.Parameter(torch.ones(2)/2, requires_grad=True)
        self.ada_weight = nn.Parameter(torch.tensor([0.5, 0.5]), requires_grad=True)

        ## initialize S_Adapter
        for n, m in self.transformer.named_modules():
            if 'S_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'c_out' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        ## initialize MLP_Adapter
        for n, m in self.transformer.named_modules():
            if 'MLP_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'c_out' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

    def forward(self, x: torch.Tensor, T):
        #with torch.no_grad():
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        #x = self.transformer(x)
        x, new_cls = self.transformer(x, T)
        #x = self.spatial_enhance(x, T=T)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :]) * self.ada_weight[0] + self.ln_post_cls(new_cls.permute(1, 0, 2))[:,0]* self.ada_weight[1]
        #x = self.ln_post(x[:, 0, :] * self.weight[0] + new_cls.permute(1, 0, 2)[:,0]* self.weight[1])
        #x = self.ln_post(x[:, 0, :])
        #new_cls = self.ln_post_cls(new_cls.permute(1, 0, 2)[:,0])

        if self.proj is not None:
            x = x @ self.proj
            #new_cls = new_cls @ self.proj_cls

        #return torch.concat((x, new_cls), 1)
        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 ):
        super().__init__()

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )


        self.initialize_parameters()

    def initialize_parameters(self):

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, T):
        return self.visual(image.type(self.dtype), T)

    def forward(self, image):
        image_features = self.encode_image(image)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]
    
    for key in list(state_dict.keys()):
        if not  'visual' in key:
            del state_dict[key]

    #convert_weights(model)
    model.load_state_dict(state_dict, strict=False)
    return model