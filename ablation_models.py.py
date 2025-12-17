import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from lib.GLCONet_module import BasicConv2d
import copy

class Network(nn.Module):
    """
    Baseline: Simple encoder-decoder without any attention or fusion modules
    This serves as our starting point for progressive ablation
    """
    def __init__(self, channels=32):
        super(BaselineNetwork, self).__init__()
        # Shared encoder backbone
        self.shared_encoder = timm.create_model('mobilevit_s.cvnets_in1k', pretrained=True, features_only=True)
        
        # Simple upsampling operations
        self.dePixelShuffle = torch.nn.PixelShuffle(2)
        self.reduce = nn.Sequential(
            BasicConv2d(channels*2, channels, kernel_size=1),
            BasicConv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.up = nn.Sequential(
            BasicConv2d(channels//4, channels, kernel_size=1),
            BasicConv2d(channels, channels, kernel_size=3, padding=1)
        )
        
        # Simple processing without attention
        self.process5 = nn.Sequential(
            BasicConv2d(640, channels, kernel_size=3, padding=1),
            BasicConv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.process4 = nn.Sequential(
            BasicConv2d(128+channels, channels, kernel_size=3, padding=1),
            BasicConv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.process3 = nn.Sequential(
            BasicConv2d(96+channels, channels, kernel_size=3, padding=1),
            BasicConv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.process2 = nn.Sequential(
            BasicConv2d(64+channels, channels, kernel_size=3, padding=1),
            BasicConv2d(channels, channels, kernel_size=3, padding=1)
        )
        
        # Output heads
        self.out5 = nn.Conv2d(channels, 1, 1)
        self.out4 = nn.Conv2d(channels, 1, 1)
        self.out3 = nn.Conv2d(channels, 1, 1)
        self.out2 = nn.Conv2d(channels, 1, 1)
        self.out1 = nn.Conv2d(channels, 1, 1)

    def forward(self, x):
        image = x
        out0_bk, x1, x2, x3, x4 = self.shared_encoder(x)

        # Simple processing without attention mechanisms
        h5 = self.process5(x4)
        h5_up = self.up(self.dePixelShuffle(h5))
        
        h4 = self.process4(torch.cat((x3, h5_up), 1))
        h4_up = self.up(self.dePixelShuffle(h4))
        
        h3 = self.process3(torch.cat((x2, h4_up), 1))
        h3_up = self.up(self.dePixelShuffle(h3))
        
        h2 = self.process2(torch.cat((x1, h3_up), 1))
        
        # Generate predictions
        f5 = F.interpolate(self.out5(h5), size=image.size()[2:], mode='bilinear', align_corners=True)
        f4 = F.interpolate(self.out4(h4), size=image.size()[2:], mode='bilinear', align_corners=True)
        f3 = F.interpolate(self.out3(h3), size=image.size()[2:], mode='bilinear', align_corners=True)
        f2 = F.interpolate(self.out2(h2), size=image.size()[2:], mode='bilinear', align_corners=True)
        f1 = F.interpolate(self.out1(h2), size=image.size()[2:], mode='bilinear', align_corners=True)
        
        return f5, f4, f3, f2, f1


class AblationNetwork_Step1(BaselineNetwork):
    """
    Step 1: Add Global and Local modules (without attention)
    This tests the impact of separate global-local processing
    """
    def __init__(self, channels=32):
        super(AblationNetwork_Step1, self).__init__(channels)
        
        # Replace simple processing with Global/Local modules (no attention)
        from lib.GLCONet_module import Global_NoAttention, Local_NoAttention
        
        self.Global5 = Global_NoAttention(640, channels)
        self.Local5 = Local_NoAttention(640, channels)
        
        self.Global4 = Global_NoAttention(128+channels, channels)
        self.Local4 = Local_NoAttention(128+channels, channels)
        
        self.Global3 = Global_NoAttention(96+channels, channels)
        self.Local3 = Local_NoAttention(96+channels, channels)
        
        self.Global2 = Global_NoAttention(64+channels, channels)
        self.Local2 = Local_NoAttention(64+channels, channels)

    def forward(self, x):
        image = x
        out0_bk, x1, x2, x3, x4 = self.shared_encoder(x)

        # Global-Local processing without attention
        g5 = self.Global5(x4)
        l5 = self.Local5(x4)
        h5 = g5 + l5  # Simple addition fusion
        h5_up = self.up(self.dePixelShuffle(h5))
        
        input4 = torch.cat((x3, h5_up), 1)
        g4 = self.Global4(input4)
        l4 = self.Local4(input4)
        h4 = g4 + l4
        h4_up = self.up(self.dePixelShuffle(h4))
        
        input3 = torch.cat((x2, h4_up), 1)
        g3 = self.Global3(input3)
        l3 = self.Local3(input3)
        h3 = g3 + l3
        h3_up = self.up(self.dePixelShuffle(h3))
        
        input2 = torch.cat((x1, h3_up), 1)
        g2 = self.Global2(input2)
        l2 = self.Local2(input2)
        h2 = g2 + l2
        
        # Generate predictions
        f5 = F.interpolate(self.out5(h5), size=image.size()[2:], mode='bilinear', align_corners=True)
        f4 = F.interpolate(self.out4(h4), size=image.size()[2:], mode='bilinear', align_corners=True)
        f3 = F.interpolate(self.out3(h3), size=image.size()[2:], mode='bilinear', align_corners=True)
        f2 = F.interpolate(self.out2(h2), size=image.size()[2:], mode='bilinear', align_corners=True)
        f1 = F.interpolate(self.out1(h2), size=image.size()[2:], mode='bilinear', align_corners=True)
        
        return f5, f4, f3, f2, f1


class AblationNetwork_Step2(AblationNetwork_Step1):
    """
    Step 2: Add CrissCross Attention to Global/Local modules
    This tests the impact of the attention mechanism
    """
    def __init__(self, channels=32):
        super(AblationNetwork_Step1, self).__init__(channels)
        
        # Replace with attention-enabled modules
        from lib.GLCONet_module import Global, Local
        
        self.Global5 = Global(640, channels)
        self.Local5 = Local(640, channels)
        
        self.Global4 = Global(128+channels, channels)
        self.Local4 = Local(128+channels, channels)
        
        self.Global3 = Global(96+channels, channels)
        self.Local3 = Local(96+channels, channels)
        
        self.Global2 = Global(64+channels, channels)
        self.Local2 = Local(64+channels, channels)


class AblationNetwork_Step3(AblationNetwork_Step2):
    """
    Step 3: Add GL_FI (Global-Local Feature Integration)
    This tests the impact of sophisticated fusion
    """
    def __init__(self, channels=32):
        super(AblationNetwork_Step3, self).__init__(channels)
        
        from lib.GLCONet_module import GL_FI
        
        self.GL_FI5 = GL_FI(channels)
        self.GL_FI4 = GL_FI(channels)
        self.GL_FI3 = GL_FI(channels)
        self.GL_FI2 = GL_FI(channels)

    def forward(self, x):
        image = x
        out0_bk, x1, x2, x3, x4 = self.shared_encoder(x)

        # Global-Local processing with sophisticated fusion
        g5 = self.Global5(x4)
        l5 = self.Local5(x4)
        h5 = self.GL_FI5(g5, l5)  # Sophisticated fusion
        h5_up = self.up(self.dePixelShuffle(h5))
        
        input4 = torch.cat((x3, h5_up), 1)
        g4 = self.Global4(input4)
        l4 = self.Local4(input4)
        h4 = self.GL_FI4(g4, l4)
        h4_up = self.up(self.dePixelShuffle(h4))
        
        input3 = torch.cat((x2, h4_up), 1)
        g3 = self.Global3(input3)
        l3 = self.Local3(input3)
        h3 = self.GL_FI3(g3, l3)
        h3_up = self.up(self.dePixelShuffle(h3))
        
        input2 = torch.cat((x1, h3_up), 1)
        g2 = self.Global2(input2)
        l2 = self.Local2(input2)
        h2 = self.GL_FI2(g2, l2)
        
        # Generate predictions
        f5 = F.interpolate(self.out5(h5), size=image.size()[2:], mode='bilinear', align_corners=True)
        f4 = F.interpolate(self.out4(h4), size=image.size()[2:], mode='bilinear', align_corners=True)
        f3 = F.interpolate(self.out3(h3), size=image.size()[2:], mode='bilinear', align_corners=True)
        f2 = F.interpolate(self.out2(h2), size=image.size()[2:], mode='bilinear', align_corners=True)
        f1 = F.interpolate(self.out1(h2), size=image.size()[2:], mode='bilinear', align_corners=True)
        
        return f5, f4, f3, f2, f1


class AblationNetwork_Step4(AblationNetwork_Step3):
    """
    Step 4: Add Enhanced Context Global (ECG) module - Global6
    This isolates the impact of the ECG module specifically
    """
    def __init__(self, channels=32):
        super(AblationNetwork_Step4, self).__init__(channels)
        
        from lib.GLCONet_module import Global_1
        
        self.Global6 = Global_1(640 + channels, channels)  # ECG module

    def forward(self, x):
        image = x
        out0_bk, x1, x2, x3, x4 = self.shared_encoder(x)

        # Global-Local processing
        g5 = self.Global5(x4)
        l5 = self.Local5(x4)
        h5 = self.GL_FI5(g5, l5)
        h5_up = self.up(self.dePixelShuffle(h5))
        
        # ECG module processing
        enhanced_context = self.Global6(torch.cat((x4, h5), 1))
        
        input4 = torch.cat((x3, h5_up), 1)
        g4 = self.Global4(input4)
        l4 = self.Local4(input4)
        h4 = self.GL_FI4(g4, l4)
        h4_up = self.up(self.dePixelShuffle(h4))
        
        input3 = torch.cat((x2, h4_up), 1)
        g3 = self.Global3(input3)
        l3 = self.Local3(input3)
        h3 = self.GL_FI3(g3, l3)
        h3_up = self.up(self.dePixelShuffle(h3))
        
        input2 = torch.cat((x1, h3_up), 1)
        g2 = self.Global2(input2)
        l2 = self.Local2(input2)
        h2 = self.GL_FI2(g2, l2)
        
        # Generate predictions (use ECG in context prediction)
        f5 = F.interpolate(enhanced_context, size=image.size()[2:], mode='bilinear', align_corners=True)
        f4 = F.interpolate(self.out4(h4), size=image.size()[2:], mode='bilinear', align_corners=True)
        f3 = F.interpolate(self.out3(h3), size=image.size()[2:], mode='bilinear', align_corners=True)
        f2 = F.interpolate(self.out2(h2), size=image.size()[2:], mode='bilinear', align_corners=True)
        f1 = F.interpolate(self.out1(h2), size=image.size()[2:], mode='bilinear', align_corners=True)
        
        return f5, f4, f3, f2, f1


class AblationNetwork_Step5(AblationNetwork_Step4):
    """
    Step 5: Add FI_2 (Enhanced Feature Integration with Region-Aware Attention)
    This tests the impact of region-aware processing
    """
    def __init__(self, channels=32):
        super(AblationNetwork_Step5, self).__init__(channels)
        
        from lib.GLCONet_module import FI_2
        
        self.FI_2_4 = FI_2(channels, channels)
        self.FI_2_3 = FI_2(channels, channels)
        self.FI_2_2 = FI_2(channels, channels)

    def forward(self, x):
        image = x
        out0_bk, x1, x2, x3, x4 = self.shared_encoder(x)

        # Global-Local processing
        g5 = self.Global5(x4)
        l5 = self.Local5(x4)
        h5 = self.GL_FI5(g5, l5)
        h5_up = self.up(self.dePixelShuffle(h5))
        
        # ECG module processing
        enhanced_context = self.Global6(torch.cat((x4, h5), 1))
        
        input4 = torch.cat((x3, h5_up), 1)
        g4 = self.Global4(input4)
        l4 = self.Local4(input4)
        h4 = self.FI_2_4(g4, l4, h5, enhanced_context)  # Region-aware fusion
        h4_up = self.up(self.dePixelShuffle(h4))
        
        input3 = torch.cat((x2, h4_up), 1)
        g3 = self.Global3(input3)
        l3 = self.Local3(input3)
        h3 = self.FI_2_3(g3, l3, h4, h5)
        h3_up = self.up(self.dePixelShuffle(h3))
        
        input2 = torch.cat((x1, h3_up), 1)
        g2 = self.Global2(input2)
        l2 = self.Local2(input2)
        h2 = self.FI_2_2(g2, l2, h3, h4)
        
        # Generate predictions
        f5 = F.interpolate(enhanced_context, size=image.size()[2:], mode='bilinear', align_corners=True)
        f4 = F.interpolate(self.out4(h4), size=image.size()[2:], mode='bilinear', align_corners=True)
        f3 = F.interpolate(self.out3(h3), size=image.size()[2:], mode='bilinear', align_corners=True)
        f2 = F.interpolate(self.out2(h2), size=image.size()[2:], mode='bilinear', align_corners=True)
        f1 = F.interpolate(self.out1(h2), size=image.size()[2:], mode='bilinear', align_corners=True)
        
        return f5, f4, f3, f2, f1


class AblationNetwork_Step6(AblationNetwork_Step5):
    """
    Step 6: Add FI_1 (Final Holistic Integration) - Full Model
    This is the complete model with all components
    """
    def __init__(self, channels=32):
        super(AblationNetwork_Step6, self).__init__(channels)
        
        from lib.GLCONet_module import FI_1
        
        self.FI_1 = FI_1(channels, channels)

    def forward(self, x):
        image = x
        out0_bk, x1, x2, x3, x4 = self.shared_encoder(x)

        # Global-Local processing
        g5 = self.Global5(x4)
        l5 = self.Local5(x4)
        h5 = self.GL_FI5(g5, l5)
        h5_up = self.up(self.dePixelShuffle(h5))
        
        # ECG module processing
        enhanced_context = self.Global6(torch.cat((x4, h5), 1))
        
        input4 = torch.cat((x3, h5_up), 1)
        g4 = self.Global4(input4)
        l4 = self.Local4(input4)
        h4 = self.FI_2_4(g4, l4, h5, enhanced_context)
        h4_up = self.up(self.dePixelShuffle(h4))
        
        input3 = torch.cat((x2, h4_up), 1)
        g3 = self.Global3(input3)
        l3 = self.Local3(input3)
        h3 = self.FI_2_3(g3, l3, h4, h5)
        h3_up = self.up(self.dePixelShuffle(h3))
        
        input2 = torch.cat((x1, h3_up), 1)
        g2 = self.Global2(input2)
        l2 = self.Local2(input2)
        h2 = self.FI_2_2(g2, l2, h3, h4)
        
        # Final holistic integration
        final_h5 = self.FI_1(h5, h5, enhanced_context)
        final_h4 = self.FI_1(h4, h5_up, enhanced_context)
        final_h3 = self.FI_1(h3, h4_up, enhanced_context)
        final_h2 = self.FI_1(h2, h3_up, enhanced_context)
        
        # Generate predictions
        f5 = F.interpolate(enhanced_context, size=image.size()[2:], mode='bilinear', align_corners=True)
        f4 = F.interpolate(final_h5, size=image.size()[2:], mode='bilinear', align_corners=True)
        f3 = F.interpolate(final_h4, size=image.size()[2:], mode='bilinear', align_corners=True)
        f2 = F.interpolate(final_h3, size=image.size()[2:], mode='bilinear', align_corners=True)
        f1 = F.interpolate(final_h2, size=image.size()[2:], mode='bilinear', align_corners=True)
        
        return f5, f4, f3, f2, f1


# Ablation Configuration Manager
class AblationConfig:
    """
    Configuration manager for ablation studies
    """
    
    ABLATION_STEPS = {
        'baseline': {
            'model_class': Network,
            'description': 'Simple encoder-decoder baseline without attention or fusion',
            'components': []
        },
        'step1_global_local': {
            'model_class': AblationNetwork_Step1,
            'description': 'Add Global/Local modules (no attention)',
            'components': ['Global_NoAttn', 'Local_NoAttn']
        },
        'step2_attention': {
            'model_class': AblationNetwork_Step2,
            'description': 'Add CrissCross Attention to Global/Local',
            'components': ['Global_NoAttn', 'Local_NoAttn', 'CrissCrossAttention']
        },
        'step3_gl_fusion': {
            'model_class': AblationNetwork_Step3,
            'description': 'Add GL_FI (Global-Local Feature Integration)',
            'components': ['Global', 'Local', 'CrissCrossAttention', 'GL_FI']
        },
        'step4_ecg': {
            'model_class': AblationNetwork_Step4,
            'description': 'Add ECG (Enhanced Context Global) module',
            'components': ['Global', 'Local', 'CrissCrossAttention', 'GL_FI', 'ECG(Global_1)']
        },
        'step5_region_aware': {
            'model_class': AblationNetwork_Step5,
            'description': 'Add FI_2 (Region-Aware Feature Integration)',
            'components': ['Global', 'Local', 'CrissCrossAttention', 'GL_FI', 'ECG(Global_1)', 'FI_2']
        },
        'step6_full': {
            'model_class': AblationNetwork_Step6,
            'description': 'Full model with FI_1 (Final Holistic Integration)',
            'components': ['Global', 'Local', 'CrissCrossAttention', 'GL_FI', 'ECG(Global_1)', 'FI_2', 'FI_1']
        }
    }
    
    @classmethod
    def get_model(cls, step_name, channels=32):
        """Get model for specific ablation step"""
        if step_name not in cls.ABLATION_STEPS:
            raise ValueError(f"Unknown ablation step: {step_name}")
        
        model_class = cls.ABLATION_STEPS[step_name]['model_class']
        return model_class(channels=channels)
    
    @classmethod
    def get_description(cls, step_name):
        """Get description for specific ablation step"""
        return cls.ABLATION_STEPS[step_name]['description']
    
    @classmethod
    def get_components(cls, step_name):
        """Get components included in specific ablation step"""
        return cls.ABLATION_STEPS[step_name]['components']
    
    @classmethod
    def print_ablation_plan(cls):
        """Print the complete ablation study plan"""
        print("\n" + "="*80)
        print("PROGRESSIVE ABLATION STUDY PLAN")
        print("="*80)
        
        for i, (step_name, config) in enumerate(cls.ABLATION_STEPS.items()):
            print(f"\n{i+1}. {step_name.upper()}")
            print(f"   Description: {config['description']}")
            print(f"   Components: {', '.join(config['components']) if config['components'] else 'None'}")
            
            if i < len(cls.ABLATION_STEPS) - 1:
                next_step = list(cls.ABLATION_STEPS.keys())[i+1]
                next_components = set(cls.ABLATION_STEPS[next_step]['components'])
                current_components = set(config['components'])
                new_components = next_components - current_components
                if new_components:
                    print(f"   → Next adds: {', '.join(new_components)}")
        
        print("\n" + "="*80)


# Example usage and testing
if __name__ == "__main__":
    # Print ablation plan
    AblationConfig.print_ablation_plan()
    
    # Test each ablation step
    print("\nTesting ablation models...")
    
    input_tensor = torch.randn(1, 3, 512, 512)
    
    for step_name in AblationConfig.ABLATION_STEPS.keys():
        print(f"\nTesting {step_name}...")
        model = AblationConfig.get_model(step_name, channels=32)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        
        # Test forward pass
        try:
            with torch.no_grad():
                outputs = model(input_tensor)
            print(f"  Forward pass: ✓ ({len(outputs)} outputs)")
            print(f"  Output shapes: {[out.shape for out in outputs]}")
        except Exception as e:
            print(f"  Forward pass: ✗ ({str(e)})")