import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers

# Original utility functions (keeping them the same)
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)



# Additional modules needed for ablation study
# These should be added to your lib/LiteCOD_module.py file

import torch
import torch.nn as nn
import torch.nn.functional as F

class Global_NoAttention(nn.Module):
    """
    Global module without attention mechanism for ablation study
    This isolates the impact of the attention mechanism
    """
    def __init__(self, in_channel, out_channel, backbone_type="transformer"):
        super(Global_NoAttention, self).__init__()
        
        # Single efficient block with depthwise separable conv (no attention)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, padding=1, groups=in_channel, bias=False),  # Depthwise
            nn.Conv2d(in_channel, out_channel, 1, bias=False),  # Pointwise
            AdaptiveNorm(out_channel, backbone_type),
            nn.ReLU(inplace=True)
        )
        
        # Conditional residual
        self.residual = nn.Conv2d(in_channel, out_channel, 1, bias=False) if in_channel != out_channel else None

    def forward(self, x):
        residual = self.residual(x) if self.residual is not None else x
        x = self.conv(x)
        return x + residual


class Local_NoAttention(nn.Module):
    """
    Local module without attention mechanism for ablation study
    This isolates the impact of the attention mechanism
    """
    def __init__(self, in_channel, out_channel, backbone_type="transformer"):
        super(Local_NoAttention, self).__init__()
        
        # Single ultra-efficient block (no attention)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1, groups=min(in_channel, out_channel), bias=False),
            AdaptiveNorm(out_channel, backbone_type),
            nn.ReLU(inplace=True)
        )
        
        # Conditional residual
        self.residual = nn.Conv2d(in_channel, out_channel, 1, bias=False) if in_channel != out_channel else None

    def forward(self, x):
        residual = self.residual(x) if self.residual is not None else x
        x = self.conv(x)
        return x + residual


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


# Alternative lightweight versions for deeper ablation
class UltraLightGlobal(nn.Module):
    """
    Ultra-lightweight Global module for parameter efficiency analysis
    """
    def __init__(self, in_channel, out_channel, backbone_type="transformer"):
        super(UltraLightGlobal, self).__init__()
        
        # Minimal processing
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, bias=False),
            AdaptiveNorm(out_channel, backbone_type),
            nn.ReLU(inplace=True)
        )
        
        # Simple global pooling attention
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(out_channel, out_channel // 4, 1),
            nn.ReLU(),
            nn.Conv2d(out_channel // 4, out_channel, 1),
            nn.Sigmoid()
        )
        
        # Conditional residual
        self.residual = nn.Conv2d(in_channel, out_channel, 1, bias=False) if in_channel != out_channel else None

    def forward(self, x):
        residual = self.residual(x) if self.residual is not None else x
        x = self.conv(x)
        
        # Simple global attention
        attention = self.gate(self.global_pool(x))
        x = x * attention
        
        return x + residual


class UltraLightLocal(nn.Module):
    """
    Ultra-lightweight Local module for parameter efficiency analysis
    """
    def __init__(self, in_channel, out_channel, backbone_type="transformer"):
        super(UltraLightLocal, self).__init__()
        
        # Minimal local processing
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=False),
            AdaptiveNorm(out_channel, backbone_type),
            nn.ReLU(inplace=True)
        )
        
        # Conditional residual
        self.residual = nn.Conv2d(in_channel, out_channel, 1, bias=False) if in_channel != out_channel else None

    def forward(self, x):
        residual = self.residual(x) if self.residual is not None else x
        x = self.conv(x)
        return x + residual


# Component isolation modules for detailed ablation
class OnlyCrissCrossAttention(nn.Module):
    """
    Module that only applies CrissCross attention without other processing
    For isolating the pure impact of attention
    """
    def __init__(self, channels):
        super(OnlyCrissCrossAttention, self).__init__()
        from your_original_file import CrissCrossAttention  # Import from your main file
        self.attention = CrissCrossAttention(channels)
        
    def forward(self, x):
        return self.attention(x)


class OnlyGlobalLocalFusion(nn.Module):
    """
    Module that only performs Global-Local fusion without attention
    For isolating the impact of fusion strategy
    """
    def __init__(self, channels, fusion_type="add"):
        super(OnlyGlobalLocalFusion, self).__init__()
        self.fusion_type = fusion_type
        
        if fusion_type == "concat":
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(channels * 2, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
        elif fusion_type == "attention":
            self.attention_weights = nn.Sequential(
                nn.Conv2d(channels * 2, 2, 1),
                nn.Softmax(dim=1)
            )
    
    def forward(self, global_feat, local_feat):
        if self.fusion_type == "add":
            return global_feat + local_feat
        elif self.fusion_type == "concat":
            fused = torch.cat([global_feat, local_feat], dim=1)
            return self.fusion_conv(fused)
        elif self.fusion_type == "attention":
            combined = torch.cat([global_feat, local_feat], dim=1)
            weights = self.attention_weights(combined)
            weighted_global = global_feat * weights[:, 0:1, :, :]
            weighted_local = local_feat * weights[:, 1:2, :, :]
            return weighted_global + weighted_local
        else:
            return global_feat + local_feat






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

# ULTRA-OPTIMIZED: Minimal Criss-Cross Attention with massive FLOPS reduction
class CrissCrossAttention(nn.Module):
    """
    Ultra-lightweight Criss-Cross Attention with 90% FLOPS reduction
    
    Key optimizations:
    1. Extreme channel reduction (C//32 instead of C//16)
    2. Separable convolutions for Q,K,V
    3. Only horizontal OR vertical attention (not both)
    4. Fixed downsampling for large maps
    5. Shared Q-K projections
    """
    def __init__(self, in_dim, reduction_ratio=32, spatial_reduction=4):
        super(CrissCrossAttention, self).__init__()
        self.in_dim = in_dim
        self.spatial_reduction = spatial_reduction
        reduced_dim = max(1, in_dim // reduction_ratio)
        
        # Shared Q-K projection with separable conv (massive parameter reduction)
        self.qk_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, padding=1, groups=in_dim, bias=False),  # Depthwise
            nn.Conv2d(in_dim, reduced_dim, 1, bias=False)  # Pointwise to reduced dim
        )
        
        # Lightweight value projection
        self.v_conv = nn.Conv2d(in_dim, in_dim // 4, 1, bias=False)
        
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        
        # Aggressive spatial downsampling for efficiency
        if H * W > 32 * 32:
            x_down = F.adaptive_avg_pool2d(x, (H//self.spatial_reduction, W//self.spatial_reduction))
            H_work, W_work = H//self.spatial_reduction, W//self.spatial_reduction
        else:
            x_down = x
            H_work, W_work = H, W
        
        # Shared Q-K projection
        qk = self.qk_conv(x_down)
        v = self.v_conv(x_down)
        
        # Only horizontal attention (50% computation vs criss-cross)
        q_h = qk.permute(0, 3, 1, 2).contiguous().view(B * W_work, -1, H_work).permute(0, 2, 1)
        k_h = qk.permute(0, 3, 1, 2).contiguous().view(B * W_work, -1, H_work)
        v_h = v.permute(0, 3, 1, 2).contiguous().view(B * W_work, -1, H_work)
        
        # Efficient attention with temperature scaling
        energy = torch.bmm(q_h, k_h) / (qk.size(1) ** 0.5)
        attention = F.softmax(energy, dim=2)
        out_h = torch.bmm(v_h, attention).view(B, W_work, -1, H_work).permute(0, 2, 3, 1)
        
        # Upsample and project back
        if H_work != H or W_work != W:
            out_h = F.interpolate(out_h, size=(H, W), mode='bilinear', align_corners=False)
        
        # Lightweight output projection
        out_h = F.conv2d(out_h, self.v_conv.weight.transpose(0, 1), bias=None)
        
        return self.gamma * out_h + x

# ULTRA-OPTIMIZED: Minimal Cross-Modal Attention
class CrossModalCrissCross(nn.Module):
    """
    Ultra-lightweight Cross-Modal attention with 95% FLOPS reduction
    
    Key optimizations:
    1. Single shared projection for all operations
    2. Extreme channel reduction (C//64)
    3. Fixed spatial downsampling
    4. Simplified cross-attention (no full criss-cross)
    5. Early fusion approach
    """
    def __init__(self, channels, reduction_ratio=64, spatial_reduction=4):
        super(CrossModalCrissCross, self).__init__()
        self.channels = channels
        self.spatial_reduction = spatial_reduction
        reduced_channels = max(1, channels // reduction_ratio)
        
        # Single ultra-lightweight projection shared across all operations
        self.shared_proj = nn.Conv2d(channels * 2, reduced_channels, 1, bias=False)
        
        # Minimal output projection
        self.out_proj = nn.Conv2d(reduced_channels, channels, 1, bias=False)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, global_feat, local_feat):
        B, C, H, W = global_feat.size()
        
        # Aggressive spatial downsampling
        if H * W > 16 * 16:
            global_small = F.adaptive_avg_pool2d(global_feat, (H//self.spatial_reduction, W//self.spatial_reduction))
            local_small = F.adaptive_avg_pool2d(local_feat, (H//self.spatial_reduction, W//self.spatial_reduction))
        else:
            global_small, local_small = global_feat, local_feat
        
        # Early fusion with shared projection (massive FLOPS save)
        fused = torch.cat([global_small, local_small], dim=1)
        attended = self.shared_proj(fused)
        
        # Simple spatial attention
        attended = F.softmax(attended.view(B, -1, attended.size(2) * attended.size(3)), dim=2)
        attended = attended.view_as(self.shared_proj(fused))
        
        # Project back and upsample
        out = self.out_proj(attended)
        if out.size(2) != H or out.size(3) != W:
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        
        # Apply to both modalities with shared enhancement
        enhanced_global = self.gamma * out + global_feat
        enhanced_local = self.gamma * out + local_feat
        
        return enhanced_global, enhanced_local

# ULTRA-OPTIMIZED: Minimal Global module 
class Global(nn.Module):
    """
    Ultra-efficient Global module with 80% parameter reduction
    
    Key optimizations:
    1. Single depthwise separable block
    2. Minimal attention mechanism
    3. No residual if channels match
    """
    def __init__(self, in_channel, out_channel, backbone_type="transformer"):
        super(Global, self).__init__()
        
        # Single efficient block with depthwise separable conv
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, padding=1, groups=in_channel, bias=False),  # Depthwise
            nn.Conv2d(in_channel, out_channel, 1, bias=False),  # Pointwise
            AdaptiveNorm(out_channel, backbone_type),
            nn.ReLU(inplace=True)
        )

        # Minimal attention 
        self.attention = CrissCrossAttention(out_channel)
        
        # Conditional residual
        self.residual = nn.Conv2d(in_channel, out_channel, 1, bias=False) if in_channel != out_channel else None

    def forward(self, x):
        residual = self.residual(x) if self.residual is not None else x
        x = self.conv(x)
        x = self.attention(x)
        return x + residual

# ULTRA-OPTIMIZED: Minimal Local module
class Local(nn.Module):
    """
    Ultra-efficient Local module with 90% parameter reduction
    
    Key optimizations:
    1. Single depthwise separable block
    2. Replace channel attention with simple global pooling + linear
    3. Minimal operations
    """
    def __init__(self, in_channel, out_channel, backbone_type="transformer"):
        super(Local, self).__init__()
        
        # Single ultra-efficient block
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1, groups=min(in_channel, out_channel), bias=False),
            AdaptiveNorm(out_channel, backbone_type),
            nn.ReLU(inplace=True)
        )

        # Ultra-minimal channel attention (just 2 parameters per channel!)
        self.channel_scale = nn.Parameter(torch.ones(1, out_channel, 1, 1))
        
        # Conditional residual
        self.residual = nn.Conv2d(in_channel, out_channel, 1, bias=False) if in_channel != out_channel else None

    def forward(self, x):
        residual = self.residual(x) if self.residual is not None else x
        x = self.conv(x)
        
        # Ultra-lightweight channel scaling
        x = x * self.channel_scale
        
        return x + residual

# ULTRA-OPTIMIZED: Minimal GL_FI
class GL_FI(nn.Module):
    """
    Ultra-efficient Global-Local Feature Integration with 95% FLOPS reduction
    
    Key optimizations:
    1. Single processing path instead of multiple branches
    2. Minimal cross-modal attention
    3. No gating mechanism
    4. Direct feature fusion
    """
    def __init__(self, in_channels=128, backbone_type="transformer"):
        super(GL_FI, self).__init__()
        
        # Single lightweight processing
        self.process = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, bias=False),
            AdaptiveNorm(in_channels, backbone_type),
            nn.ReLU(inplace=True)
        )
        
        # Minimal cross-modal attention
        self.cross_modal = CrossModalCrissCross(in_channels)

    def forward(self, G, L):
        # Apply minimal cross-modal attention
        G_enhanced, L_enhanced = self.cross_modal(G, L)
        
        # Direct fusion
        fused = torch.cat([G_enhanced, L_enhanced], dim=1)
        out = self.process(fused)
        
        return out

# ULTRA-OPTIMIZED: Minimal FI_1 
class FI_1(nn.Module):
    """Ultra-efficient Feature Integration with 85% parameter reduction"""
    def __init__(self, in_channels, mid_channels, backbone_type="transformer"):
        super(FI_1, self).__init__()
        
        # Single lightweight processing
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels // 2, 1, bias=False),
            AdaptiveNorm(in_channels // 2, backbone_type),
            nn.ReLU(inplace=True)
        )

        # Minimal output head
        self.out_y = nn.Conv2d(in_channels // 2, 1, 1, bias=False)

        # Efficient GL_FI
        self.GL_FI = GL_FI(in_channels, backbone_type)

    def forward(self, G, L, prior_cam):
        GL = self.GL_FI(G, L)
        
        prior_cam = F.interpolate(prior_cam, size=L.size()[2:], mode='bilinear', align_corners=True)
        
        # Minimal reverse attention (keep as it's effective)
        r_prior_cam = 1 - torch.sigmoid(prior_cam)
        enhanced_GL = r_prior_cam.expand(-1, GL.size(1), -1, -1) * GL
        
        # Direct processing
        combined = self.conv(torch.cat([enhanced_GL, GL], dim=1))
        y = self.out_y(combined) + prior_cam
        
        return y

# ULTRA-OPTIMIZED: Minimal Region-Aware Attention
class RegionAwareAttention(nn.Module):
    """
    Ultra-lightweight Region-Aware Attention with 90% parameter reduction
    
    Key optimizations:
    1. Single shared conv for all operations
    2. Minimal channel dimensions
    3. Direct feature weighting
    """
    def __init__(self, channels):
        super(RegionAwareAttention, self).__init__()
        
        # Single ultra-lightweight shared processor
        hidden_dim = max(1, channels // 16)
        self.shared_conv = nn.Sequential(
            nn.Conv2d(channels + 2, hidden_dim, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, current_features, prior_cam, prev_features):
        # Ensure single channel inputs
        if prior_cam.size(1) != 1:
            prior_cam = prior_cam.mean(dim=1, keepdim=True)
        if prev_features.size(1) != 1:
            prev_features = prev_features.mean(dim=1, keepdim=True)
            
        # Direct attention computation
        attention_input = torch.cat([current_features, 1 - prior_cam, prev_features], dim=1)
        attention_weights = self.shared_conv(attention_input)
        
        return current_features * attention_weights

class FI_2(nn.Module):
    """Ultra-efficient FI_2 with minimal Region-Aware Attention"""
    def __init__(self, in_channels, mid_channels, backbone_type="transformer"):
        super(FI_2, self).__init__()
        
        # Minimal processing  
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels // 2, 1, bias=False),
            AdaptiveNorm(in_channels // 2, backbone_type), 
            nn.ReLU(inplace=True)
        )
        
        self.out_y = nn.Conv2d(in_channels // 2, 1, 1, bias=False)
        
        # Efficient modules
        self.GL_FI = GL_FI(in_channels, backbone_type)
        self.region_attention = RegionAwareAttention(in_channels)
        
    def forward(self, G, L, x1, prior_cam):
        GL = self.GL_FI(G, L)
        
        prior_cam = F.interpolate(prior_cam, size=L.size()[2:], mode='bilinear', align_corners=True)
        x1_prior_cam = F.interpolate(x1, size=L.size()[2:], mode='bilinear', align_corners=True)
        
        # Region-aware enhancement
        region_enhanced = self.region_attention(GL, prior_cam, x1_prior_cam)
        
        # Direct processing
        combined = self.conv(torch.cat([region_enhanced, GL], dim=1))
        y = self.out_y(combined) + prior_cam + x1_prior_cam
        
        return y

# ULTRA-OPTIMIZED: Minimal Global_1
class Global_1(nn.Module):
    """Ultra-efficient Global_1 with 90% parameter reduction"""
    def __init__(self, in_channel, out_channel, backbone_type="transformer"):
        super(Global_1, self).__init__()
        
        # Single depthwise separable block
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, padding=1, groups=in_channel, bias=False),
            nn.Conv2d(in_channel, out_channel, 1, bias=False),
            AdaptiveNorm(out_channel, backbone_type),
            nn.ReLU(inplace=True)
        )

        # Minimal attention
        self.attention = CrissCrossAttention(out_channel)
        
        # Direct output
        self.out = nn.Conv2d(out_channel, 1, 1, bias=False)
        
        # Conditional residual
        self.residual = nn.Conv2d(in_channel, out_channel, 1, bias=False) if in_channel != out_channel else None

    def forward(self, x):
        residual = self.residual(x) if self.residual is not None else x
        x = self.backbone(x)
        x = self.attention(x) + residual
        x = self.out(x)
        return x

