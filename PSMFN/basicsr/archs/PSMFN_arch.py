<<<<<<< HEAD
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from basicsr.archs import Upsamplers
from torch import Tensor
from basicsr.utils.registry import ARCH_REGISTRY


class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div=4):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, num_feat, reduction=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // reduction, 1, padding=0),
            nn.GELU(),
            nn.Conv2d(num_feat // reduction, num_feat, 1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.attention(x)
        return x * y


class MSFM(nn.Module):
    def __init__(self, dim, n_levels=4):
        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels

        self.mfr = nn.ModuleList(
            [Partial_conv3(chunk_dim) for i in range(self.n_levels)])

        self.conv_mix = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        h, w = x.size()[-2:]
        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h // 2 ** i, w // 2 ** i)
                s = F.adaptive_max_pool2d(xc[i], p_size)
                s = self.mfr[i](s)
                s = F.interpolate(s, size=(h, w), mode='nearest')
            else:
                s = self.mfr[i](xc[i])
            out.append(s)

        out = torch.cat(out, dim=1)
        out = self.act(self.conv_mix(out))
        return out


class MFFB(nn.Module):
    def __init__(self, dim):
        super(MFFB, self).__init__()
        self.path_1 = MSFM(dim)

        self.path_2 = nn.Sequential(
            Partial_conv3(dim),
            nn.Conv2d(dim, dim, kernel_size=(1, 1),
                      stride=1, padding=0, dilation=1, groups=1, bias=False),
            nn.GELU(),
            ChannelAttention(dim)
        )
        self.mid = nn.Sequential(
            Partial_conv3(dim),
            nn.Conv2d(dim, dim, kernel_size=(1, 1),
                      stride=1, padding=0, dilation=1, groups=1, bias=False),
            nn.GELU(),
        )
        self.conv_mix = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        forward = self.mid(x)
        path1 = self.path_1(x)
        path2 = self.path_2(x)
        out = forward + path1 + path2
        out = self.act(self.conv_mix(out))
        return out


class EIRB(nn.Module):
    def __init__(self, dim):
        super(EIRB, self).__init__()
        num_feat = dim * 2
        self.pconv = Partial_conv3(dim)
        self.pwconv_up = nn.Conv2d(dim, num_feat, kernel_size=(1, 1),
                                stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.pwconv_down = nn.Conv2d(num_feat, dim, kernel_size=(1, 1),
                                   stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        temp = x.clone()
        x = self.pconv(x)
        x = self.act(self.pwconv_up(x))
        x = self.pwconv_down(x)
        output = x + temp
        return output


class LEFB(nn.Module):
    def __init__(self, dim):
        super(LEFB, self).__init__()

        self.front = EIRB(dim)
        self.conv_last = MFFB(dim)

    def forward(self, x):
        x = self.front(x)
        x = self.conv_last(x)
        return x


@ARCH_REGISTRY.register()  
class PSMFN(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=64, num_out_ch=3, upscale=4):
        super(PSMFN, self).__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, padding=1)

        self.block1 = LEFB(num_feat)
        self.block2 = LEFB(num_feat)
        self.block3 = LEFB(num_feat)
        self.block4 = LEFB(num_feat)
        self.block5 = LEFB(num_feat)
        self.block6 = LEFB(num_feat)
        self.block7 = LEFB(num_feat)
        self.block8 = LEFB(num_feat)

        self.conv_last = nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.upsampler = Upsamplers.PixelShuffleDirect(scale=upscale, num_feat=num_feat, num_out_ch=num_out_ch)

    def forward(self, x):
        x = self.conv_first(x)
        short_cut = x
# ********************************************************
        out_1 = self.block1(x)
        out_1 = out_1 + short_cut

        out_2 = self.block2(out_1)
        out_2 = out_2 + short_cut

        out_3 = self.block3(out_2)
        out_3 = out_3 + short_cut

        out_4 = self.block4(out_3)
        out_4 = out_4 + short_cut

        out_5 = self.block5(out_4)
        out_5 = out_5 + short_cut

        out_6 = self.block6(out_5)
        out_6 = out_6 + short_cut

        out_7 = self.block7(out_6)
        out_7 = out_7 + short_cut

        out_8 = self.block8(out_7)
        out_8 = out_8 + short_cut
# ********************************************************
        out = self.conv_last(out_8)
        out = self.act(out)
        out_lr = out + x
        output = self.upsampler(out_lr)
        return output



=======
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from basicsr.archs import Upsamplers
from torch import Tensor
from basicsr.utils.registry import ARCH_REGISTRY


class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div=4):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, num_feat, reduction=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // reduction, 1, padding=0),
            nn.GELU(),
            nn.Conv2d(num_feat // reduction, num_feat, 1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.attention(x)
        return x * y


class MSFM(nn.Module):
    def __init__(self, dim, n_levels=4):
        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels

        self.mfr = nn.ModuleList(
            [Partial_conv3(chunk_dim) for i in range(self.n_levels)])

        self.conv_mix = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        h, w = x.size()[-2:]
        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h // 2 ** i, w // 2 ** i)
                s = F.adaptive_max_pool2d(xc[i], p_size)
                s = self.mfr[i](s)
                s = F.interpolate(s, size=(h, w), mode='nearest')
            else:
                s = self.mfr[i](xc[i])
            out.append(s)

        out = torch.cat(out, dim=1)
        out = self.act(self.conv_mix(out))
        return out


class MFFB(nn.Module):
    def __init__(self, dim):
        super(MFFB, self).__init__()
        self.path_1 = MSFM(dim)

        self.path_2 = nn.Sequential(
            Partial_conv3(dim),
            nn.Conv2d(dim, dim, kernel_size=(1, 1),
                      stride=1, padding=0, dilation=1, groups=1, bias=False),
            nn.GELU(),
            ChannelAttention(dim)
        )
        self.mid = nn.Sequential(
            Partial_conv3(dim),
            nn.Conv2d(dim, dim, kernel_size=(1, 1),
                      stride=1, padding=0, dilation=1, groups=1, bias=False),
            nn.GELU(),
        )
        self.conv_mix = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        forward = self.mid(x)
        path1 = self.path_1(x)
        path2 = self.path_2(x)
        out = forward + path1 + path2
        out = self.act(self.conv_mix(out))
        return out


class EIRB(nn.Module):
    def __init__(self, dim):
        super(EIRB, self).__init__()
        num_feat = dim * 2
        self.pconv = Partial_conv3(dim)
        self.pwconv_up = nn.Conv2d(dim, num_feat, kernel_size=(1, 1),
                                stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.pwconv_down = nn.Conv2d(num_feat, dim, kernel_size=(1, 1),
                                   stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        temp = x.clone()
        x = self.pconv(x)
        x = self.act(self.pwconv_up(x))
        x = self.pwconv_down(x)
        output = x + temp
        return output


class LEFB(nn.Module):
    def __init__(self, dim):
        super(LEFB, self).__init__()

        self.front = EIRB(dim)
        self.conv_last = MFFB(dim)

    def forward(self, x):
        x = self.front(x)
        x = self.conv_last(x)
        return x


@ARCH_REGISTRY.register()  
class PSMFN(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=64, num_out_ch=3, upscale=4):
        super(PSMFN, self).__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, padding=1)

        self.block1 = LEFB(num_feat)
        self.block2 = LEFB(num_feat)
        self.block3 = LEFB(num_feat)
        self.block4 = LEFB(num_feat)
        self.block5 = LEFB(num_feat)
        self.block6 = LEFB(num_feat)
        self.block7 = LEFB(num_feat)
        self.block8 = LEFB(num_feat)

        self.conv_last = nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.upsampler = Upsamplers.PixelShuffleDirect(scale=upscale, num_feat=num_feat, num_out_ch=num_out_ch)

    def forward(self, x):
        x = self.conv_first(x)
        short_cut = x
# ********************************************************
        out_1 = self.block1(x)
        out_1 = out_1 + short_cut

        out_2 = self.block2(out_1)
        out_2 = out_2 + short_cut

        out_3 = self.block3(out_2)
        out_3 = out_3 + short_cut

        out_4 = self.block4(out_3)
        out_4 = out_4 + short_cut

        out_5 = self.block5(out_4)
        out_5 = out_5 + short_cut

        out_6 = self.block6(out_5)
        out_6 = out_6 + short_cut

        out_7 = self.block7(out_6)
        out_7 = out_7 + short_cut

        out_8 = self.block8(out_7)
        out_8 = out_8 + short_cut
# ********************************************************
        out = self.conv_last(out_8)
        out = self.act(out)
        out_lr = out + x
        output = self.upsampler(out_lr)
        return output



>>>>>>> 19b0de10a7c5a2cb53bf02e644c3f69c6d365127
