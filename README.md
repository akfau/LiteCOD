## Quantitative Results
Performance comparison of LiteCOD against state-of-the-art methods across multiple COD benchmarks (CAMO, COD10K, NC4K). Our method achieves an optimal balance between detection accuracy and computational efficiency, outperforming lightweight techniques while maintaining competitive performance with heavyweight approaches at significantly reduced parameter count and computational overhead.
MethodPublicationParam (M)FLOPs (G)FPSCAMO S↑CAMO E↑CAMO F↑CAMO M↓COD10K S↑COD10K E↑COD10K F↑COD10K M↓NC4K S↑NC4K E↑NC4K F↑NC4K M↓SINetCVPR'2048.9519.30820.7510.7710.6060.1000.7710.7970.5510.0510.8080.8380.7230.058PFNetCVPR'2146.5026.391060.7820.8550.6950.0850.8000.8680.6600.0400.8290.8940.7450.053LSRCVPR'2150.9417.361500.7930.8590.7430.0800.8040.8830.6780.0370.8400.9040.6660.048SINetV2TPAMI'2226.9812.171300.8200.8840.7430.0700.8150.8640.6890.0370.8470.9010.7700.044BGNetICCAI'2279.8558.24880.8120.8760.7490.0730.8310.8920.7220.0330.8510.9110.7880.044SegMaRCVPR'2256.9733.49850.8150.8810.7530.0710.8330.8690.7240.0340.8610.9050.7810.046ZoomNetCVPR'2232.28101.35410.8200.8830.7520.0660.8380.8930.7290.0290.8530.9070.7840.043FEDERCVPR'2344.1335.80420.8020.8770.7380.0710.8220.9010.7160.0320.8470.9130.7890.044CamoFocus-PWACV'2473.2044.00450.8170.8840.7520.0670.8380.9000.7240.0290.8650.9130.7880.042FSELECCV'2429.1535.64-0.8220.8920.7580.0670.8330.8980.7280.0310.8550.9130.7920.042CamoFormerTPAMI'2436.1134.20870.8170.8840.7520.0670.8380.9000.7240.0290.8650.9130.7880.042ESNetKBS'2510.774.52990.848--0.0490.850--0.0310.862--0.040— Lightweight COD Methods —DGNet-SMIR'237.021.141530.8260.8960.7540.0630.8100.8690.6720.0360.8450.9020.7640.047ASBICVIU'239.479.84950.8390.8960.7610.0640.8250.8720.6900.0350.8550.9020.7750.046ERRNETPR9.479.84950.8390.8960.7610.0640.8250.8720.6900.0350.8550.9020.7750.046CamoFocus-EWACV'244.765.54780.8170.8840.7520.0670.8380.9000.7240.0290.8650.9130.7880.042TinyCODICASSP'234.721.40600.8220.8900.7520.0660.8310.8770.6780.0360.8430.9030.7660.047FINetSPL'243.741.161270.8280.8900.7520.0650.8170.8820.6860.0340.8470.9040.7710.047LiteCOD (Ours)—5.157.95720.8410.9070.7960.0560.8520.9200.7650.0260.8700.9260.8220.036

↑ higher is better, ↓ lower is better. - denotes unavailable results. Bold indicates best results among lightweight methods.

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

