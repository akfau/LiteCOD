## Architecture
 
<p align="center">
  <img src="framework/framework9.png" alt="LiteCOD Architecture" width="90%"/>
</p>
**Figure 1.** Overall architecture of LiteCOD featuring multi-scale feature extraction through hierarchical stages (S₁–S₄), Holistic Unification Modules (HUMs) for bilateral global–local feature enhancement, Enhanced Context Generation (ECG) for semantic guidance, multi-stage feature integration (MFI) for progressive feature refinement, and multi-level supervision with predictions (P₁ − P₄ and Pₓ) in each stage for comprehensive COD.


## Qualitative Comparison with State-of-the-Art Methods

<p align="center">
  <img src="Framework/Full_Comparison.png" alt="Qualitative Comparison with SOTA" width="95%"/>
</p>

**Figure 2.** Qualitative comparison of LiteCOD with recent COD methods across diverse challenging scenarios. The comparison includes SegMaR, ZoomNet, SINet-V2, FSPNet, FEDER, MRRNet, PUENet, and EVP across five different test cases showing various camouflaged objects (including what appears to be camouflaged animals and objects in natural environments). Each row shows the original image, ground truth mask, and segmentation results from LiteCOD (ours) and the comparison methods. The results demonstrate LiteCOD's superior performance in accurately detecting and segmenting camouflaged objects while maintaining better boundary preservation and structural fidelity compared to existing approaches.


## Qualitative Results Comparison with other Lightweight Methods

<p align="center">
  <img src="Framework/Lightweight_Comparison.png" alt="Qualitative Comparison" width="85%"/>
</p>

**Figure 3.** Qualitative comparison between our proposed method and contemporary lightweight COD approaches (TinyCOD, FINet, and DGNet-S) across diverse challenging scenarios. Our lightweight approach demonstrates superior boundary preservation and structural fidelity compared to other efficient methods, particularly excelling in cases involving both large-scale and minute camouflaged targets and complex textural patterns. These results validate the effectiveness of our proposed architecture in achieving high-quality detection while maintaining computational efficiency suitable for practical deployment.

## Quantitative Results
 
Performance comparison of LiteCOD against state-of-the-art methods across multiple COD benchmarks (CAMO, COD10K, NC4K). Our method achieves an optimal balance between detection accuracy and computational efficiency, outperforming lightweight techniques while maintaining competitive performance with heavyweight approaches at significantly reduced parameter count and computational overhead.
 
| Method | Publication | Param (M) | FLOPs (G) | FPS | CAMO S↑ | CAMO E↑ | CAMO F↑ | CAMO M↓ | COD10K S↑ | COD10K E↑ | COD10K F↑ | COD10K M↓ | NC4K S↑ | NC4K E↑ | NC4K F↑ | NC4K M↓ |
|--------|-------------|-----------|-----------|-----|---------|---------|---------|---------|-----------|-----------|-----------|-----------|---------|---------|---------|---------|
| SINet | CVPR'20 | 48.95 | 19.30 | 82 | 0.751 | 0.771 | 0.606 | 0.100 | 0.771 | 0.797 | 0.551 | 0.051 | 0.808 | 0.838 | 0.723 | 0.058 |
| PFNet | CVPR'21 | 46.50 | 26.39 | 106 | 0.782 | 0.855 | 0.695 | 0.085 | 0.800 | 0.868 | 0.660 | 0.040 | 0.829 | 0.894 | 0.745 | 0.053 |
| LSR | CVPR'21 | 50.94 | 17.36 | 150 | 0.793 | 0.859 | 0.743 | 0.080 | 0.804 | 0.883 | 0.678 | 0.037 | 0.840 | 0.904 | 0.666 | 0.048 |
| SINetV2 | TPAMI'22 | 26.98 | 12.17 | 130 | 0.820 | 0.884 | 0.743 | 0.070 | 0.815 | 0.864 | 0.689 | 0.037 | 0.847 | 0.901 | 0.770 | 0.044 |
| BGNet | ICCAI'22 | 79.85 | 58.24 | 88 | 0.812 | 0.876 | 0.749 | 0.073 | 0.831 | 0.892 | 0.722 | 0.033 | 0.851 | 0.911 | 0.788 | 0.044 |
| SegMaR | CVPR'22 | 56.97 | 33.49 | 85 | 0.815 | 0.881 | 0.753 | 0.071 | 0.833 | 0.869 | 0.724 | 0.034 | 0.861 | 0.905 | 0.781 | 0.046 |
| ZoomNet | CVPR'22 | 32.28 | 101.35 | 41 | 0.820 | 0.883 | 0.752 | 0.066 | 0.838 | 0.893 | 0.729 | 0.029 | 0.853 | 0.907 | 0.784 | 0.043 |
| FEDER | CVPR'23 | 44.13 | 35.80 | 42 | 0.802 | 0.877 | 0.738 | 0.071 | 0.822 | 0.901 | 0.716 | 0.032 | 0.847 | 0.913 | 0.789 | 0.044 |
| CamoFocus-P | WACV'24 | 73.20 | 44.00 | 45 | 0.817 | 0.884 | 0.752 | 0.067 | 0.838 | 0.900 | 0.724 | 0.029 | 0.865 | 0.913 | 0.788 | 0.042 |
| FSEL | ECCV'24 | 29.15 | 35.64 | - | 0.822 | 0.892 | 0.758 | 0.067 | 0.833 | 0.898 | 0.728 | 0.031 | 0.855 | 0.913 | 0.792 | 0.042 |
| CamoFormer | TPAMI'24 | 36.11 | 34.20 | 87 | 0.817 | 0.884 | 0.752 | 0.067 | 0.838 | 0.900 | 0.724 | 0.029 | 0.865 | 0.913 | 0.788 | 0.042 |
| ESNet | KBS'25 | 10.77 | 4.52 | 99 | 0.848 | - | - | 0.049 | 0.850 | - | - | 0.031 | 0.862 | - | - | 0.040 |
| **— Lightweight COD Methods —** | | | | | | | | | | | | | | | | |
| DGNet-S | MIR'23 | 7.02 | 1.14 | 153 | 0.826 | 0.896 | 0.754 | 0.063 | 0.810 | 0.869 | 0.672 | 0.036 | 0.845 | 0.902 | 0.764 | 0.047 |
| ASBI | CVIU'23 | 9.47 | 9.84 | 95 | 0.839 | 0.896 | 0.761 | 0.064 | 0.825 | 0.872 | 0.690 | 0.035 | 0.855 | 0.902 | 0.775 | 0.046 |
| ERRNET | PR | 9.47 | 9.84 | 95 | 0.839 | 0.896 | 0.761 | 0.064 | 0.825 | 0.872 | 0.690 | 0.035 | 0.855 | 0.902 | 0.775 | 0.046 |
| CamoFocus-E | WACV'24 | 4.76 | 5.54 | 78 | 0.817 | 0.884 | 0.752 | 0.067 | 0.838 | 0.900 | 0.724 | 0.029 | 0.865 | 0.913 | 0.788 | 0.042 |
| TinyCOD | ICASSP'23 | 4.72 | 1.40 | 60 | 0.822 | 0.890 | 0.752 | 0.066 | 0.831 | 0.877 | 0.678 | 0.036 | 0.843 | 0.903 | 0.766 | 0.047 |
| FINet | SPL'24 | 3.74 | 1.16 | 127 | 0.828 | 0.890 | 0.752 | 0.065 | 0.817 | 0.882 | 0.686 | 0.034 | 0.847 | 0.904 | 0.771 | 0.047 |
| **LiteCOD (Ours)** | — | **5.15** | **7.95** | **72** | **0.841** | **0.907** | **0.796** | **0.056** | **0.852** | **0.920** | **0.765** | **0.026** | **0.870** | **0.926** | **0.822** | **0.036** |
 
> ↑ higher is better, ↓ lower is better. `-` denotes unavailable results. **Bold** indicates best results among lightweight methods.
