# CalibPL Fix Log

## CRITICAL FIXES

### CRITICAL-01: YOLOv12 Backbone — CSPDarkNet → R-ELAN + Area Attention
- **Location**: Section 4.1 line 518, Supplementary B line 214
- **Verification**: arXiv:2502.12524 confirms YOLOv12 uses R-ELAN (Residual Efficient Layer Aggregation Network) with Area Attention, NOT CSPDarkNet (which is YOLOv4/v5)
- **Action**: Replace text throughout

### CRITICAL-02: 88% FP reduction is WRONG → correct value is ~70%
- **Calculation**: 
  - Baseline FP = (1-0.61)×14,194 = 5,535.7
  - CalibPL FP = (1-0.84)×10,494 = 1,679.0
  - Reduction = (5535.7-1679.0)/5535.7 = 69.7% ≈ **70%**
- **Location**: Abstract (line 83), Section 4.2 (line 583)

### CRITICAL-03: Sections 4.6 and 4.7 have tables/figures but NO narrative text
- **Action**: Write complete content for both subsections

### CRITICAL-04: D^val_L partitioning not explained
- **Finding**: "up to 300 images" but only 140 labeled images total.
  In supplementary, it says "300 validation images" for fitting.
  The 140 is labeled training images; D^val_L is a held-out split from D_L.
  Need explicit clarification.

## MAJOR FIXES

### MAJOR-01: Table 6 synergy claim mathematically wrong
- **Verification**: 1.20 < 0.78 + 0.76 = 1.54 → sub-additive (diminishing returns)
- **Location**: Table 6 caption lines 696-699

### MAJOR-02: κ=12 empirical, not analytic
- **Note**: Already partially addressed in Limitations and proof (lines 141-145 supplement)
  But Corollary 1 in main paper presents it ambiguously

### MAJOR-03: Supplementary Corollary 1 — relabel as Proposition 2
- **Location**: Supplementary line 148

### MAJOR-04: SKU-110K 10% protocol unclear
- **Finding**: Official SKU-110K: 8,233 train images. 10% = 823, but paper says 140.
  The "10%" likely refers to object-level annotation density, not image count.
  Or it is a custom 140-image labeled subset from a 1,400-image working subset.
  Need explicit clarification.

### MAJOR-05: Tesla K80 memory — 11 GB
- **Finding**: K80 is 12 GB/GPU nominal, but with ECC enabled = ~11.25 GB usable.
  The paper's "11 GB" is the practical ECC-enabled figure. ACCEPTABLE but needs note.

### MAJOR-06: Missing IoU-Net citation
- **Paper**: Jiang et al. ECCV 2018 "Acquisition of Localization Confidence for Accurate Object Detection"
- **Action**: Add to Related Work in Calibration in object detection paragraph

### MAJOR-07: No p-value for primary positive result SKU-110K
- **Action**: Compute from available std data: +1.20±0.11 vs 0
  t = 1.20/0.11 ≈ 10.9 for single estimate; with seeds {42,43,44} (n=3)
  t = 1.20/(0.11/sqrt(3)) = 1.20/0.0635 ≈ 18.9, df=2, p << 0.001
  Add this to Section 4.2

### MAJOR-08: CrowdHuman iterations 4-5 pending
- **Action**: Remove "camera-ready reservation" phrasing. State as limitation.

## MINOR FIXES
- ECE notation: abstract says <10^{-3}, table says <0.003 — these are identical
- 39% claim source vs 28% from table
- CGJS matching procedure undefined
- Bootstrap stratification on single-class dataset
- Table 7 missing AP50 column
- CGJS 0.45 vs 0.5 threshold difference
- i.i.d. assumption discussion
- Density notation ≈147 vs =141.95±32.4
