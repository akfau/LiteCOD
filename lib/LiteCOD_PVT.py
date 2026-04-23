import timm
import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import numbers

# Copy all your existing classes exactly as they are
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class AdaptiveNorm(nn.Module):
    """Adaptive normalization that switches between BatchNorm and LayerNorm based on backbone type"""
    def __init__(self, channels, backbone_type="transformer"):
        super(AdaptiveNorm, self).__init__()
        self.backbone_type = backbone_type
        if backbone_type == "transformer":
            # Use min(16, channels) groups for fewer parameters
            num_groups = min(16, channels)
            if channels % num_groups != 0:
                num_groups = 1
            self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        else:
            self.norm = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        return self.norm(x)

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, need_relu=True,
                 backbone_type="transformer"):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = AdaptiveNorm(out_channels, backbone_type)
        self.relu = nn.ReLU()
        self.need_relu = need_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.need_relu:
            x = self.relu(x)
        return x

class CrissCrossAttention(nn.Module):
    def __init__(self, in_dim, reduction_ratio=32, spatial_reduction=4):
        super(CrissCrossAttention, self).__init__()
        self.in_dim = in_dim
        self.spatial_reduction = spatial_reduction
        reduced_dim = max(1, in_dim // reduction_ratio)
        
        self.qk_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, padding=1, groups=in_dim, bias=False),
            nn.Conv2d(in_dim, reduced_dim, 1, bias=False)
        )
        
        self.v_conv = nn.Conv2d(in_dim, in_dim // 4, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        
        if H * W > 32 * 32:
            x_down = F.adaptive_avg_pool2d(x, (H//self.spatial_reduction, W//self.spatial_reduction))
            H_work, W_work = H//self.spatial_reduction, W//self.spatial_reduction
        else:
            x_down = x
            H_work, W_work = H, W
        
        qk = self.qk_conv(x_down)
        v = self.v_conv(x_down)
        
        q_h = qk.permute(0, 3, 1, 2).contiguous().view(B * W_work, -1, H_work).permute(0, 2, 1)
        k_h = qk.permute(0, 3, 1, 2).contiguous().view(B * W_work, -1, H_work)
        v_h = v.permute(0, 3, 1, 2).contiguous().view(B * W_work, -1, H_work)
        
        energy = torch.bmm(q_h, k_h) / (qk.size(1) ** 0.5)
        attention = F.softmax(energy, dim=2)
        out_h = torch.bmm(v_h, attention).view(B, W_work, -1, H_work).permute(0, 2, 3, 1)
        
        if H_work != H or W_work != W:
            out_h = F.interpolate(out_h, size=(H, W), mode='bilinear', align_corners=False)
        
        out_h = F.conv2d(out_h, self.v_conv.weight.transpose(0, 1), bias=None)
        
        return self.gamma * out_h + x

class CrossModalCrissCross(nn.Module):
    def __init__(self, channels, reduction_ratio=64, spatial_reduction=4):
        super(CrossModalCrissCross, self).__init__()
        self.channels = channels
        self.spatial_reduction = spatial_reduction
        reduced_channels = max(1, channels // reduction_ratio)
        
        self.shared_proj = nn.Conv2d(channels * 2, reduced_channels, 1, bias=False)
        self.out_proj = nn.Conv2d(reduced_channels, channels, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, global_feat, local_feat):
        B, C, H, W = global_feat.size()
        
        if H * W > 16 * 16:
            global_small = F.adaptive_avg_pool2d(global_feat, (H//self.spatial_reduction, W//self.spatial_reduction))
            local_small = F.adaptive_avg_pool2d(local_feat, (H//self.spatial_reduction, W//self.spatial_reduction))
        else:
            global_small, local_small = global_feat, local_feat
        
        fused = torch.cat([global_small, local_small], dim=1)
        attended = self.shared_proj(fused)
        
        attended = F.softmax(attended.view(B, -1, attended.size(2) * attended.size(3)), dim=2)
        attended = attended.view_as(self.shared_proj(fused))
        
        out = self.out_proj(attended)
        if out.size(2) != H or out.size(3) != W:
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        
        enhanced_global = self.gamma * out + global_feat
        enhanced_local = self.gamma * out + local_feat
        
        return enhanced_global, enhanced_local

class Global(nn.Module):
    def __init__(self, in_channel, out_channel, backbone_type="transformer"):
        super(Global, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, padding=1, groups=in_channel, bias=False),
            nn.Conv2d(in_channel, out_channel, 1, bias=False),
            AdaptiveNorm(out_channel, backbone_type),
            nn.ReLU(inplace=True)
        )

        self.attention = CrissCrossAttention(out_channel)
        self.residual = nn.Conv2d(in_channel, out_channel, 1, bias=False) if in_channel != out_channel else None

    def forward(self, x):
        residual = self.residual(x) if self.residual is not None else x
        x = self.conv(x)
        x = self.attention(x)
        return x + residual

class Local(nn.Module):
    def __init__(self, in_channel, out_channel, backbone_type="transformer"):
        super(Local, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1, groups=min(in_channel, out_channel), bias=False),
            AdaptiveNorm(out_channel, backbone_type),
            nn.ReLU(inplace=True)
        )

        self.channel_scale = nn.Parameter(torch.ones(1, out_channel, 1, 1))
        self.residual = nn.Conv2d(in_channel, out_channel, 1, bias=False) if in_channel != out_channel else None

    def forward(self, x):
        residual = self.residual(x) if self.residual is not None else x
        x = self.conv(x)
        x = x * self.channel_scale
        return x + residual

class GL_FI(nn.Module):
    def __init__(self, in_channels=128, backbone_type="transformer"):
        super(GL_FI, self).__init__()
        
        self.process = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, bias=False),
            AdaptiveNorm(in_channels, backbone_type),
            nn.ReLU(inplace=True)
        )
        
        self.cross_modal = CrossModalCrissCross(in_channels)

    def forward(self, G, L):
        G_enhanced, L_enhanced = self.cross_modal(G, L)
        fused = torch.cat([G_enhanced, L_enhanced], dim=1)
        out = self.process(fused)
        return out

class FI_1(nn.Module):
    def __init__(self, in_channels, mid_channels, backbone_type="transformer"):
        super(FI_1, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels // 2, 1, bias=False),
            AdaptiveNorm(in_channels // 2, backbone_type),
            nn.ReLU(inplace=True)
        )

        self.out_y = nn.Conv2d(in_channels // 2, 1, 1, bias=False)
        self.GL_FI = GL_FI(in_channels, backbone_type)

    def forward(self, G, L, prior_cam):
        GL = self.GL_FI(G, L)
        
        prior_cam = F.interpolate(prior_cam, size=L.size()[2:], mode='bilinear', align_corners=True)
        
        r_prior_cam = 1 - torch.sigmoid(prior_cam)
        enhanced_GL = r_prior_cam.expand(-1, GL.size(1), -1, -1) * GL
        
        combined = self.conv(torch.cat([enhanced_GL, GL], dim=1))
        y = self.out_y(combined) + prior_cam
        
        return y

class RegionAwareAttention(nn.Module):
    def __init__(self, channels):
        super(RegionAwareAttention, self).__init__()
        
        hidden_dim = max(1, channels // 16)
        self.shared_conv = nn.Sequential(
            nn.Conv2d(channels + 2, hidden_dim, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, current_features, prior_cam, prev_features):
        if prior_cam.size(1) != 1:
            prior_cam = prior_cam.mean(dim=1, keepdim=True)
        if prev_features.size(1) != 1:
            prev_features = prev_features.mean(dim=1, keepdim=True)
            
        attention_input = torch.cat([current_features, 1 - prior_cam, prev_features], dim=1)
        attention_weights = self.shared_conv(attention_input)
        
        return current_features * attention_weights

class FI_2(nn.Module):
    def __init__(self, in_channels, mid_channels, backbone_type="transformer"):
        super(FI_2, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels // 2, 1, bias=False),
            AdaptiveNorm(in_channels // 2, backbone_type), 
            nn.ReLU(inplace=True)
        )
        
        self.out_y = nn.Conv2d(in_channels // 2, 1, 1, bias=False)
        self.GL_FI = GL_FI(in_channels, backbone_type)
        self.region_attention = RegionAwareAttention(in_channels)
        
    def forward(self, G, L, x1, prior_cam):
        GL = self.GL_FI(G, L)
        
        prior_cam = F.interpolate(prior_cam, size=L.size()[2:], mode='bilinear', align_corners=True)
        x1_prior_cam = F.interpolate(x1, size=L.size()[2:], mode='bilinear', align_corners=True)
        
        region_enhanced = self.region_attention(GL, prior_cam, x1_prior_cam)
        
        combined = self.conv(torch.cat([region_enhanced, GL], dim=1))
        y = self.out_y(combined) + prior_cam + x1_prior_cam
        
        return y

class Global_1(nn.Module):
    def __init__(self, in_channel, out_channel, backbone_type="transformer"):
        super(Global_1, self).__init__()
        
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, padding=1, groups=in_channel, bias=False),
            nn.Conv2d(in_channel, out_channel, 1, bias=False),
            AdaptiveNorm(out_channel, backbone_type),
            nn.ReLU(inplace=True)
        )

        self.attention = CrissCrossAttention(out_channel)
        self.out = nn.Conv2d(out_channel, 1, 1, bias=False)
        self.residual = nn.Conv2d(in_channel, out_channel, 1, bias=False) if in_channel != out_channel else None

    def forward(self, x):
        residual = self.residual(x) if self.residual is not None else x
        x = self.backbone(x)
        x = self.attention(x) + residual
        x = self.out(x)
        return x


def make_h5_processor(in_ch=640, out_ch=64):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

# ---------- Baseline (reference) - reuse your BaselineNetwork if present ----------
class Baseline(nn.Module):
    def __init__(self, channels=64, encoder_name='mobilevit_s.cvnets_in1k', pretrained=True):
        super(Baseline, self).__init__()
        self.shared_encoder = timm.create_model(encoder_name, pretrained=pretrained, features_only=True)
        # process highest-level (x4)
        self.process = nn.Sequential(
            nn.Conv2d(640, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 1)
        )

    def forward(self, x):
        _, _, _, _, x4 = self.shared_encoder(x)
        out = self.process(x4)
        out = F.interpolate(out, size=x.size()[2:], mode='bilinear', align_corners=True)
        # return five maps to match your other networks' API
        return out, out, out, out, out

# ---------- 1) Baseline + ECG only ----------
class BaselineWithECG(nn.Module):
    def __init__(self, channels=64, encoder_name='mobilevit_s.cvnets_in1k', pretrained=True):
        super(BaselineWithECG, self).__init__()
        self.shared_encoder = timm.create_model(encoder_name, pretrained=pretrained, features_only=True)
        self.h5_process = make_h5_processor(in_ch=640, out_ch=channels)
        # ECG / Enhanced Context Generator (Global_1 expects 640 + channels as in your Network)
        self.ECG = Global_1(640 + channels, channels)
        self.FI_1 = FI_1(channels, channels)

    def forward(self, x):
        image = x
        _, _, _, _, x4 = self.shared_encoder(x)

        h5 = self.h5_process(x4)                       # B x channels x H x W
        concat = torch.cat([x4, h5], dim=1)           # B x (640+channels) x H x W

        enhanced_context = self.ECG(concat)           # B x channels x H x W

        # produce outputs: use enhanced_context and FI_1 to build consistent multi-level maps
        final_h5 = self.FI_1(h5, h5, enhanced_context)

        f5 = F.interpolate(enhanced_context, size=image.size()[2:], mode='bilinear', align_corners=True)
        f4 = F.interpolate(final_h5, size=image.size()[2:], mode='bilinear', align_corners=True)
        # keep other levels same as f4 for baseline comparison
        f3 = f2 = f1 = f4
        return f5, f4, f3, f2, f1

# ---------- 2) Baseline + Attention-only (no ECG) ----------
class BaselineWithAttention(nn.Module):
    def __init__(self, channels=64, encoder_name='mobilevit_s.cvnets_in1k', pretrained=True):
        super(BaselineWithAttention, self).__init__()
        self.shared_encoder = timm.create_model(encoder_name, pretrained=pretrained, features_only=True)
        self.h5_process = make_h5_processor(in_ch=640, out_ch=channels)
        # apply CrissCrossAttention to h5 feature maps
        self.attention = CrissCrossAttention(channels)

        # produce final single channel map
        self.out_conv = nn.Conv2d(channels, 1, 1, bias=False)

    def forward(self, x):
        image = x
        _, _, _, _, x4 = self.shared_encoder(x)

        h5 = self.h5_process(x4)
        h5_att = self.attention(h5)                     # attends on h5
        out = self.out_conv(h5_att)
        out = F.interpolate(out, size=image.size()[2:], mode='bilinear', align_corners=True)
        return out, out, out, out, out

# ---------- 3) Baseline + Local-only ----------
class BaselineWithLocal(nn.Module):
    def __init__(self, channels=64, encoder_name='mobilevit_s.cvnets_in1k', pretrained=True):
        super(BaselineWithLocal, self).__init__()
        self.shared_encoder = timm.create_model(encoder_name, pretrained=pretrained, features_only=True)
        # Local module expects in_channel=640, out_channel=channels
        self.local5 = Local(640, channels)
        self.out_conv = nn.Conv2d(channels, 1, 1, bias=False)

    def forward(self, x):
        image = x
        _, _, _, _, x4 = self.shared_encoder(x)

        l5 = self.local5(x4)
        out = self.out_conv(l5)
        out = F.interpolate(out, size=image.size()[2:], mode='bilinear', align_corners=True)
        return out, out, out, out, out

# ---------- 4) Baseline + Global-only ----------
class BaselineWithGlobal(nn.Module):
    def __init__(self, channels=64, encoder_name='mobilevit_s.cvnets_in1k', pretrained=True):
        super(BaselineWithGlobal, self).__init__()
        self.shared_encoder = timm.create_model(encoder_name, pretrained=pretrained, features_only=True)
        # Global module expects in_channel=640
        self.global5 = Global(640, channels)
        self.h5_process = make_h5_processor(in_ch=640, out_ch=channels)  # keep a processed h5 for parity
        self.out_conv = nn.Conv2d(channels, 1, 1, bias=False)

    def forward(self, x):
        image = x
        _, _, _, _, x4 = self.shared_encoder(x)

        g5 = self.global5(x4)  # B x channels x H x W
        out = self.out_conv(g5)
        out = F.interpolate(out, size=image.size()[2:], mode='bilinear', align_corners=True)
        return out, out, out, out, out

# ---------- 5) Baseline + (ECG + Attention) ----------
class BaselineECGAttention(nn.Module):
    def __init__(self, channels=64, encoder_name='mobilevit_s.cvnets_in1k', pretrained=True):
        super(BaselineECGAttention, self).__init__()
        self.shared_encoder = timm.create_model(encoder_name, pretrained=pretrained, features_only=True)

        self.h5_process = make_h5_processor(in_ch=640, out_ch=channels)
        self.attention = CrissCrossAttention(channels)

        # ECG (Global_1) working on concat(x4, h5_att)
        self.ECG = Global_1(640 + channels, channels)
        self.FI_1 = FI_1(channels, channels)
        self.out_conv = nn.Conv2d(channels, 1, 1, bias=False)

    def forward(self, x):
        image = x
        _, _, _, _, x4 = self.shared_encoder(x)

        h5 = self.h5_process(x4)
        h5_att = self.attention(h5)

        concat = torch.cat([x4, h5_att], dim=1)
        enhanced_context = self.ECG(concat)

        final_h5 = self.FI_1(h5_att, h5_att, enhanced_context)

        f5 = F.interpolate(enhanced_context, size=image.size()[2:], mode='bilinear', align_corners=True)
        f4 = F.interpolate(final_h5, size=image.size()[2:], mode='bilinear', align_corners=True)
        f3 = f2 = f1 = f4
        return f5, f4, f3, f2, f1

# ---------- 6) Baseline + (ECG + Local + Global) - fuller fusion ----------
class BaselineECGLocalGlobal(nn.Module):
    def __init__(self, channels=64, encoder_name='mobilevit_s.cvnets_in1k', pretrained=True):
        super(BaselineECGLocalGlobal, self).__init__()
        self.shared_encoder = timm.create_model(encoder_name, pretrained=pretrained, features_only=True)

        # modules for level-5
        self.global5 = Global(640, channels)
        self.local5 = Local(640, channels)
        self.holistic5 = GL_FI(channels)         # expects channel-sized inputs (G and L)
        # h5 processor for baseline parity
        self.h5_process = make_h5_processor(in_ch=640, out_ch=channels)

        # ECG that takes concat(x4, fused_h5)
        self.ECG = Global_1(640 + channels, channels)
        self.FI_1 = FI_1(channels, channels)

    def forward(self, x):
        image = x
        _, _, _, _, x4 = self.shared_encoder(x)

        g5 = self.global5(x4)                       # B x channels x H x W
        l5 = self.local5(x4)                        # B x channels x H x W
        h5 = self.holistic5(g5, l5)                 # fused B x channels x H x W

        # build ECG input from raw x4 and fused holistics
        concat = torch.cat([x4, h5], dim=1)
        enhanced_context = self.ECG(concat)

        final_h5 = self.FI_1(h5, h5, enhanced_context)

        f5 = F.interpolate(enhanced_context, size=image.size()[2:], mode='bilinear', align_corners=True)
        f4 = F.interpolate(final_h5, size=image.size()[2:], mode='bilinear', align_corners=True)
        f3 = f2 = f1 = f4
        return f5, f4, f3, f2, f1


# 4. Full Network (your original)
class Network(nn.Module):
    def __init__(self, channels=64):
        super(Network, self).__init__()
        self.shared_encoder = timm.create_model('mobilevit_s.cvnets_in1k', pretrained=True, features_only=True)
        
        self.dePixelShuffle = torch.nn.PixelShuffle(2)
        self.reduce = nn.Sequential(
            BasicConv2d(channels*2, channels, kernel_size=1),
            BasicConv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.up = nn.Sequential(
            BasicConv2d(channels//4, channels, kernel_size=1),
            BasicConv2d(channels, channels, kernel_size=3, padding=1)
        )
        
        self.Global5 = Global(640, channels)
        self.Local5 = Local(640, channels)
        self.Holistic5 = GL_FI(channels)
        
        self.Global4 = Global(128+channels, channels)
        self.Local4 = Local(128+channels, channels)
        self.Holistic4 = FI_2(channels, channels)
        
        self.Global3 = Global(96+channels, channels)
        self.Local3 = Local(96+channels, channels)
        self.Holistic3 = FI_2(channels, channels)
        
        self.Global2 = Global(64+channels, channels)
        self.Local2 = Local(64+channels, channels)
        self.Holistic2 = FI_2(channels, channels)
        
        self.Global6 = Global_1(640 + channels, channels)  # ECG module
        self.FI_1 = FI_1(channels, channels)
        
    def forward(self, x):
        image = x
        out0_bk, x1, x2, x3, x4 = self.shared_encoder(x)

        g5 = self.Global5(x4)
        l5 = self.Local5(x4)
        h5 = self.Holistic5(g5, l5)
        h5_up = self.up(self.dePixelShuffle(h5))
        
        enhanced_context = self.Global6(torch.cat((x4, h5), 1))  # ECG module
        
        input4 = torch.cat((x3, h5_up), 1)
        g4 = self.Global4(input4)
        l4 = self.Local4(input4)
        h4 = self.Holistic4(g4, l4, h5, enhanced_context)
        h4_up = self.up(self.dePixelShuffle(h4))
        
        input3 = torch.cat((x2, h4_up), 1)
        g3 = self.Global3(input3)
        l3 = self.Local3(input3)
        h3 = self.Holistic3(g3, l3, h4, h5)
        h3_up = self.up(self.dePixelShuffle(h3))
        
        input2 = torch.cat((x1, h3_up), 1)
        g2 = self.Global2(input2)
        l2 = self.Local2(input2)
        h2 = self.Holistic2(g2, l2, h3, h4)
        
        final_h5 = self.FI_1(h5, h5, enhanced_context)
        final_h4 = self.FI_1(h4, h5_up, enhanced_context)
        final_h3 = self.FI_1(h3, h4_up, enhanced_context)
        final_h2 = self.FI_1(h2, h3_up, enhanced_context)
        
        f5 = F.interpolate(enhanced_context, size=image.size()[2:], mode='bilinear', align_corners=True)
        f4 = F.interpolate(final_h5, size=image.size()[2:], mode='bilinear', align_corners=True)
        f3 = F.interpolate(final_h4, size=image.size()[2:], mode='bilinear', align_corners=True)
        f2 = F.interpolate(final_h3, size=image.size()[2:], mode='bilinear', align_corners=True)
        f1 = F.interpolate(final_h2, size=image.size()[2:], mode='bilinear', align_corners=True)
        
        return f5, f4, f3, f2, f1


# ---------- 7) Full network - alias to your original `Network` ----------
# If your `Network` class is defined in the same module, reuse it; otherwise import it.
# Here we assume it exists with signature Network(channels=64)
try:
    FullNetwork = Network  # if Network already defined in the same scope
except NameError:
    # If not present, create an alias class that raises clear error
    class FullNetwork(nn.Module):
        def __init__(self, channels=64):
            super(FullNetwork, self).__init__()
            raise RuntimeError("FullNetwork: original `Network` class not found in scope. "
                               "Please import or define `Network` (your full model) and replace this alias.")

# ---------- AblationConfig updated with systematic variants ----------
class AblationConfig:
    ABLATION_STEPS = {
        'baseline': {
            'model_class': Baseline,
            'description': 'Simple baseline encoder-decoder',
            'components': ['Encoder', 'BasicConv']
        },
        'baseline_ecg': {
            'model_class': BaselineWithECG,
            'description': 'Baseline + ECG (Global_1) only',
            'components': ['Encoder', 'ECG', 'FI_1']
        },
        'baseline_attention': {
            'model_class': BaselineWithAttention,
            'description': 'Baseline + CrissCrossAttention only',
            'components': ['Encoder', 'CrissCrossAttention']
        },
        'baseline_local': {
            'model_class': BaselineWithLocal,
            'description': 'Baseline + Local module only',
            'components': ['Encoder', 'Local']
        },
        'baseline_global': {
            'model_class': BaselineWithGlobal,
            'description': 'Baseline + Global module only',
            'components': ['Encoder', 'Global']
        },
        'baseline_ecg_attention': {
            'model_class': BaselineECGAttention,
            'description': 'Baseline + ECG + Attention',
            'components': ['Encoder', 'ECG', 'CrissCrossAttention', 'FI_1']
        },
        'baseline_ecg_local_global': {
            'model_class': BaselineECGLocalGlobal,
            'description': 'Baseline + ECG + Local + Global + Holistic fusion',
            'components': ['Encoder', 'ECG', 'Local', 'Global', 'GL_FI']
        },
        'full': {
            'model_class': FullNetwork,
            'description': 'Full model with all components (your original Network)',
            'components': ['Encoder', 'Global', 'Local', 'GL_FI', 'FI_2', 'FI_1', 'ECG']
        }
    }

    @classmethod
    def get_model(cls, step_name, channels=64, **kwargs):
        if step_name not in cls.ABLATION_STEPS:
            raise ValueError(f"Unknown ablation step: {step_name}")
        model_class = cls.ABLATION_STEPS[step_name]['model_class']
        return model_class(channels=channels, **kwargs)

    @classmethod
    def print_plan(cls):
        print("\nSYSTEMATIC ABLATION PLAN:")
        for key, v in cls.ABLATION_STEPS.items():
            print(f"{key:30s} - {v['description']}")


    @classmethod
    def get_description(cls, step_name):
        return cls.ABLATION_STEPS[step_name]['description']

    @classmethod
    def get_model_class(cls, step_name):
        return cls.ABLATION_STEPS[step_name]['model_class']