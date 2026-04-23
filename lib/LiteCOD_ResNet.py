import timm
import torch.nn as nn
import torch
import torch.nn.functional as F
from lib.LiteCOD_module import Local, Global, Global_1, FI_1, FI_2, GL_FI, BasicConv2d

'''
backbone: resnet50
Holistic arrangement of existing modules for camouflaged object detection
'''

class Network(nn.Module):
    # resnet based encoder decoder with holistic global-local integration
    def __init__(self, channels=64):
        super(Network, self).__init__()
        # Shared encoder backbone for feature extraction
        self.shared_encoder = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)
        
        # Upsampling operations
        self.dePixelShuffle = torch.nn.PixelShuffle(2)
        self.reduce = nn.Sequential(
            BasicConv2d(channels*2, channels, kernel_size=1),
            BasicConv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.up = nn.Sequential(
            BasicConv2d(channels//4, channels, kernel_size=1),
            BasicConv2d(channels, channels, kernel_size=3, padding=1)
        )
        
        # Holistic processing modules - combine global and local at each level
        # Level 5 (deepest): Process both global context and local details simultaneously
        self.Global5 = Global(2048, channels)                    # Global context extraction
        self.Local5 = Local(2048, channels)                      # Local detail extraction
        self.Holistic5 = GL_FI(channels)                         # Fuse global-local at level 5
        
        # Level 4: Holistic processing with skip connections
        self.Global4 = Global(1024+channels, channels)           # Global processing with skip
        self.Local4 = Local(1024+channels, channels)             # Local processing with skip  
        self.Holistic4 = FI_2(channels, channels)                # Holistic fusion at level 4
        
        # Level 3: Continue holistic processing
        self.Global3 = Global(512+channels, channels)            # Global branch
        self.Local3 = Local(512+channels, channels)              # Local branch
        self.Holistic3 = FI_2(channels, channels)                # Holistic fusion at level 3
        
        # Level 2: Finest level holistic processing  
        self.Global2 = Global(256+channels, channels)            # Global branch
        self.Local2 = Local(256+channels, channels)              # Local branch
        self.Holistic2 = FI_2(channels, channels)                # Holistic fusion at level 2
        
        # Enhanced context understanding for camouflaged objects
        self.Global6 = Global_1(2048 + channels, channels)       # Enhanced global context
        
        # Final holistic integration for camouflage detection
        self.FI_1 = FI_1(channels, channels)                     # Primary feature integration
        
    def forward(self, x):
        image = x
        # Extract multi-level features from encoder
        en_feats = self.shared_encoder(x)
        x0, x1, x2, x3, x4 = en_feats
        
        # Level 5 (deepest): Holistic processing of global context and local details
        g5 = self.Global5(x4)                                    # Extract global features
        l5 = self.Local5(x4)                                     # Extract local features  
        h5 = self.Holistic5(g5, l5)                              # Fuse global-local holistically
        h5_up = self.up(self.dePixelShuffle(h5))                 # Upsample for next level
        
        # Level 4: Holistic processing with encoder skip connection
        input4 = torch.cat((x3, h5_up), 1)                       # Combine encoder features with holistic features
        g4 = self.Global4(input4)                                # Global processing
        l4 = self.Local4(input4)                                 # Local processing
        h4 = self.Holistic4(g4, l4, h5, h5)                      # Holistic fusion with multi-level context
        h4_up = self.up(self.dePixelShuffle(h4))                 # Upsample for next level
        
        # Level 3: Continue holistic processing
        input3 = torch.cat((x2, h4_up), 1)                       # Combine features
        g3 = self.Global3(input3)                                # Global processing
        l3 = self.Local3(input3)                                 # Local processing  
        h3 = self.Holistic3(g3, l3, h4, h5)                      # Holistic fusion with hierarchical context
        h3_up = self.up(self.dePixelShuffle(h3))                 # Upsample for next level
        
        # Level 2: Finest level holistic processing
        input2 = torch.cat((x1, h3_up), 1)                       # Combine features
        g2 = self.Global2(input2)                                # Global processing
        l2 = self.Local2(input2)                                 # Local processing
        h2 = self.Holistic2(g2, l2, h3, h4)                      # Holistic fusion with multi-scale context
        
        # Enhanced global context for camouflaged object understanding
        enhanced_context = self.Global6(torch.cat((x4, h5), 1))  # Enhanced context with deepest features
        
        # Final holistic integration for comprehensive camouflage detection
        final_h5 = self.FI_1(h5, h5, enhanced_context)           # Refine deepest level
        final_h4 = self.FI_1(h4, h4, enhanced_context)           # Refine level 4
        final_h3 = self.FI_1(h3, h3, enhanced_context)           # Refine level 3  
        final_h2 = self.FI_1(h2, h2, enhanced_context)           # Refine level 2
        
        # Generate multi-scale predictions for camouflaged object detection
        f5 = F.interpolate(enhanced_context, size=image.size()[2:], mode='bilinear', align_corners=True)  # Context prediction
        f4 = F.interpolate(final_h5, size=image.size()[2:], mode='bilinear', align_corners=True)          # Coarse prediction
        f3 = F.interpolate(final_h4, size=image.size()[2:], mode='bilinear', align_corners=True)          # Medium prediction
        f2 = F.interpolate(final_h3, size=image.size()[2:], mode='bilinear', align_corners=True)          # Fine prediction
        f1 = F.interpolate(final_h2, size=image.size()[2:], mode='bilinear', align_corners=True)          # Finest prediction
        
        return f5, f4, f3, f2, f1

