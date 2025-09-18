import torch
from torch import nn
from torch.nn import BatchNorm2d, Conv2d, Module, PReLU, Sequential
from typing import Dict, List, Optional, Tuple
try: 
    from nets.my_modules import DropPath, DropBlock2D
    from nets.my_modules import wConv2d_new as wConv2d
    from nets.my_modules import TeLU, BSiLU, BlurPool, ECA, GeM, CoordAtt
except ImportError:
    from my_modules import DropPath, DropBlock2D
    from my_modules import wConv2d_new as wConv2d
    from my_modules import TeLU, BSiLU, BlurPool, ECA, GeM, CoordAtt
    
def count_and_log_wconv(model):
    total_conv, replaced = 0, 0
    for mod in model.modules():
        if isinstance(mod, nn.Conv2d):
            total_conv += 1
        if isinstance(mod, wConv2d):
            replaced += 1
    print(f"[wConv2d] replaced={replaced}, remaining_Conv2d={total_conv}")


DEFAULT_DEN = {
    3: [0.75],          
    5: [0.5, 0.9],      
    7: [0.3, 0.6, 0.9], 
}

def _make_wconv_from_conv(conv: nn.Conv2d, den_map):
    kH, kW = conv.kernel_size
    if kH != kW or kH % 2 == 0:
        return None  
    if kH == 1:
        return None  
    den = den_map.get(kH)
    if den is None:
        return None  
    new = wConv2d(
        in_channels=conv.in_channels, out_channels=conv.out_channels,
        kernel_size=kH, den=list(den),
        stride=conv.stride, padding=conv.padding,
        groups=conv.groups, dilation=conv.dilation,
        bias=(conv.bias is not None)
    )
    with torch.no_grad():
        new.weight.copy_(conv.weight)
        if new.bias is not None and conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new

def _make_conv_from_wconv(wc: wConv2d):
    kH, kW = wc.kernel_size
    new = nn.Conv2d(
        in_channels=wc.weight.shape[1] * wc.groups,
        out_channels=wc.weight.shape[0],
        kernel_size=(kH, kW),
        stride=wc.stride, padding=wc.padding,
        groups=wc.groups, dilation=wc.dilation,
        bias=(wc.bias is not None)
    )
    with torch.no_grad():
        new.weight.copy_(wc.weight)
        if new.bias is not None and wc.bias is not None:
            new.bias.copy_(wc.bias)
    return new

def swap_conv2d(module: nn.Module, to="wconv", den_map=None):
    if den_map is None:
        den_map = DEFAULT_DEN

    for name, child in list(module.named_children()):
        if to == "wconv" and isinstance(child, nn.Conv2d):
            replaced = _make_wconv_from_conv(child, den_map)
            if replaced is not None:
                setattr(module, name, replaced)
        elif to == "conv" and isinstance(child, wConv2d):
            restored = _make_conv_from_wconv(child)
            setattr(module, name, restored)
        swap_conv2d(getattr(module, name), to=to, den_map=den_map)

def replace_prelu(module: nn.Module, act: str = "TeLU", **kwargs) -> None:
    act_l = act.lower()

    def new_act():
        if act_l == "telu":        return TeLU(**kwargs)
        if act_l == "bsilu":       return BSiLU(**kwargs)
        if act_l == "silu":        return nn.SiLU()
        if act_l == "mish":        return nn.Mish()
        if act_l == "relu":        return nn.ReLU(inplace=True)
        if act_l in ("leakyrelu", "lrelu"):
                                    return nn.LeakyReLU(**kwargs)
        if act_l == "gelu":        return nn.GELU()
        raise ValueError(f"Unknown activation: {act}")

    for child_name, child in list(module.named_children()):
        if isinstance(child, PReLU):
            setattr(module, child_name, new_act())
        else:
            replace_prelu(child, act, **kwargs)

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv   = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn     = BatchNorm2d(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Residual_Block(Module):
     def __init__(self, in_c, out_c, residual = False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Residual_Block, self).__init__()
        self.conv       = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw    = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride,
                                     anti_alias=(stride==(2,2)), aa_filt_size=4, aa_pad_type='reflect')
        self.project    = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual   = residual
     def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output

class Residual(Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(Residual_Block(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = Sequential(*modules)
    def forward(self, x):
        return self.model(x)

class Conv_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1,
                 anti_alias: bool = False, aa_filt_size: int = 4, aa_pad_type: str = 'reflect'):
        super(Conv_block, self).__init__()
        use_aa = anti_alias and (stride == (2, 2))
        conv_stride = (1, 1) if use_aa else stride

        self.conv   = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=conv_stride, padding=padding, bias=False)
        self.bn     = BatchNorm2d(out_c)
        self.prelu  = PReLU(out_c)
        self.aa = BlurPool(out_c, pad_type=aa_pad_type, filt_size=aa_filt_size, stride=2) if use_aa else None
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        if self.aa is not None:
            x = self.aa(x)
        return x

class MobileFaceNet(Module):
    def __init__(self, embedding_size, att_type: str = 'eca', eca_k: int = 3, coord_reduction: int = 32):
        super(MobileFaceNet, self).__init__()
        self.conv1      = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1),
                                     anti_alias=True, aa_filt_size=4, aa_pad_type='reflect')

        self.conv2_dw   = Conv_block(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)

        self.conv_23    = Residual_Block(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3     = Residual(64, num_block=4, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv_34    = Residual_Block(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4     = Residual(128, num_block=6, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv_45    = Residual_Block(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5     = Residual(128, num_block=2, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        att = att_type.lower()
        if att == 'eca':
            self.att3    = ECA(channel=64,  k_size=eca_k)
            self.att4    = ECA(channel=128, k_size=eca_k)
            self.att5    = ECA(channel=128, k_size=eca_k)
            self.att_sep = ECA(channel=512, k_size=eca_k)
        elif att == 'coordatt':
            self.att3    = CoordAtt(inp=64,  oup=64,  reduction=coord_reduction)
            self.att4    = CoordAtt(inp=128, oup=128, reduction=coord_reduction)
            self.att5    = CoordAtt(inp=128, oup=128, reduction=coord_reduction)
            self.att_sep = CoordAtt(inp=512, oup=512, reduction=coord_reduction)
        else:
            raise ValueError(f"Unknown att_type: {att_type} (choose 'eca' or 'coordatt')")
        
        self.sep        = nn.Conv2d(128, 512, kernel_size=1, bias=False)
        self.sep_bn     = nn.BatchNorm2d(512)
        self.prelu      = nn.PReLU(512)

        self.GDC_dw     = nn.Conv2d(512, 512, kernel_size=7, bias=False, groups=512)
        self.GDC_bn     = nn.BatchNorm2d(512)

        self.gem = GeM(p=3, eps=1e-6)

        self.features = nn.Conv2d(512, embedding_size, kernel_size=1, bias=False)
        self.last_bn = nn.BatchNorm2d(embedding_size)
  
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_dw(x)

        x = self.conv_23(x)
        x = self.conv_3(x)
        x = self.att3(x)

        x = self.conv_34(x)
        x = self.conv_4(x)
        x = self.att4(x)

        x = self.conv_45(x)
        x = self.conv_5(x)
        x = self.att5(x)

        x = self.sep(x)
        x = self.sep_bn(x)
        x = self.prelu(x)
        x = self.att_sep(x)

        x = self.GDC_dw(x)
        x = self.GDC_bn(x)
        
        x = self.gem(x)
        x = self.features(x)
        x = self.last_bn(x)
        x = x.view(x.size(0), -1)
        return x


def get_our_model(embedding_size, pretrained, att_type: str = 'eca', eca_k: int = 3, coord_reduction: int = 32):
    if pretrained:
        raise ValueError("No pretrained model for mobilefacenet")
    return MobileFaceNet(embedding_size, att_type=att_type, eca_k=eca_k, coord_reduction=coord_reduction)


def _fit_den_length(den: List[float], tgt_len: int) -> List[float]:
    if tgt_len <= 0:
        return []
    if len(den) == tgt_len:
        return den

    import numpy as np
    src = np.asarray(den, dtype=np.float32)
    if len(src) == 1:
        return [float(src[0])] * tgt_len
    xs = np.linspace(0, 1, num=len(src), dtype=np.float32)
    xt = np.linspace(0, 1, num=tgt_len, dtype=np.float32)
    yt = np.interp(xt, xs, src)
    return yt.tolist()

def _auto_den(K: int) -> List[float]:
    import math
    half = (K - 1) // 2
    return [math.exp(-0.8 * i) for i in range(1, half + 1)]

def _make_wconv_from_conv_new(
    conv: nn.Conv2d,
    den_map: Dict[int, List[float]],
    learnable_phi: bool = True,
    normalize_phi: bool = True,
) -> Optional["wConv2d"]:
    from typing import cast
    kH, kW = conv.kernel_size
    if kH != kW or (kH % 2 == 0) or kH == 1:
        return None

    K = kH
    half = (K - 1) // 2
    den = den_map.get(K, None)
    if den is None:
        den = _auto_den(K)
    den = _fit_den_length(den, half)

    new = wConv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=K,
        den_init=list(den),
        stride=conv.stride,
        padding=conv.padding,
        groups=conv.groups,
        dilation=conv.dilation,
        bias=(conv.bias is not None),
        learnable_phi=learnable_phi,
        normalize_phi=normalize_phi,
    )

    with torch.no_grad():
        new.weight.copy_(conv.weight)
        if new.bias is not None and conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new

def _make_conv_from_wconv_new(wc: "wConv2d") -> nn.Conv2d:
    kH, kW = wc.kernel_size
    new = nn.Conv2d(
        in_channels=wc.weight.shape[1] * wc.groups,
        out_channels=wc.weight.shape[0],
        kernel_size=(kH, kW),
        stride=wc.stride,
        padding=wc.padding,
        groups=wc.groups,
        dilation=wc.dilation,
        bias=(wc.bias is not None),
    )
    with torch.no_grad():
        new.weight.copy_(wc.weight)
        if new.bias is not None and wc.bias is not None:
            new.bias.copy_(wc.bias)
    return new

def swap_conv2d_new(
    module: nn.Module,
    to: str = "wconv",
    den_map: Optional[Dict[int, List[float]]] = None,
    learnable_phi: bool = True,
    normalize_phi: bool = True,
):
    if den_map is None:
        den_map = DEFAULT_DEN

    for name, child in list(module.named_children()):
        if to == "wconv" and isinstance(child, nn.Conv2d):
            replaced = _make_wconv_from_conv_new(
                child, den_map,
                learnable_phi=learnable_phi,
                normalize_phi=normalize_phi,
            )
            if replaced is not None:
                setattr(module, name, replaced)

        elif to == "conv" and isinstance(child, wConv2d):
            restored = _make_conv_from_wconv_new(child)
            setattr(module, name, restored)

        swap_conv2d_new(
            getattr(module, name),
            to=to,
            den_map=den_map,
            learnable_phi=learnable_phi,
            normalize_phi=normalize_phi,
        )

if __name__ == "__main__":
    model = MobileFaceNet(embedding_size=128, att_type='eca', eca_k=3)
    den_map = {**DEFAULT_DEN, 3:[0.75], 7:[0.3,0.6,0.9]}  
    swap_conv2d_new(model, to="wconv", den_map=den_map, learnable_phi=True, normalize_phi=True)
    count_and_log_wconv(model)

    input = torch.randn(4, 3, 256, 256)
    output = model(input)
    print(output.shape)
