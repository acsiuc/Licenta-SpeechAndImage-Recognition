# Experiment Log — Cross-Modal Face-Voice Biometric Matching

## Ablation Table

| Configuration | Urdu Sim | Romanian Sim | Urdu EER |
|---|---|---|---|
| ResNet18 + OPL, 64 ids, English | — | 0.1377 | — |
| VGG16 + ECAPA + OPL, 64 ids, English | 0.0382 | 0.2532 | ~50% |
| VGG16 + ECAPA + SupCon, 64 ids, English | 0.0204 | 0.2252 | 47.06% |
| VGG16 + ECAPA + SupCon, 198 ids, multilingual | 0.0972 | 0.2265 | 43.60% |
| ArcFace + ECAPA + SupCon, 198 ids, multilingual | TBD | TBD | TBD |

## Experiment Details

### Exp 1 — ResNet18 + OPL
- Encoders: VGG16 128D + ResNet18 128D (both ImageNet)
- Loss: CE + OPL + InfoNCE
- Data: MAVCeleb v1, 64 ids, English, 32,000 samples
- Hardware: CPU local

### Exp 2 — VGG16 + ECAPA + OPL
- Encoders: VGG16 128D (ImageNet) + ECAPA-TDNN 192D (VoxCeleb)
- Loss: CE + OPL (×0.1) + InfoNCE (×10)
- Data: MAVCeleb v1, 64 ids, English, 32,000 samples
- Hardware: CPU local
- Closed-set: 99.11% accuracy, 3.44% EER, 0.65 separation gap

### Exp 3 — VGG16 + ECAPA + SupCon
- Encoders: VGG16 128D (ImageNet) + ECAPA-TDNN 192D (VoxCeleb)
- Loss: CE + SupCon (×0.5) + InfoNCE (×10)
- Data: MAVCeleb v1, 64 ids, English, 32,000 samples
- Hardware: CPU local
- Best val loss: 11.23
- Finding: SupCon alone with 64 ids performs slightly worse than OPL

### Exp 4 — Multilingual + SupCon (Colab)
- Encoders: VGG16 128D (ImageNet) + ECAPA-TDNN 192D (VoxCeleb, CUDA)
- Loss: CE + SupCon (×0.5) + InfoNCE (×10)
- Data: MAVCeleb v1+v2+v3, 198 ids, English+Urdu+Hindi+German
- Samples: 99,000 (500 per identity, cross-language pairs)
- Hardware: Google Colab Pro, NVIDIA Tesla T4
- Best val loss: 6.18, val accuracy: 99.78%
- Finding: Multilingual training improved Urdu similarity by 154%

### Exp 5 — ArcFace + ECAPA + SupCon (TBD)
- Encoders: ArcFace iResNet50 512D (VGGFace2) + ECAPA-TDNN 192D (VoxCeleb)
- Loss: CE + SupCon (×0.5) + InfoNCE (×10)
- Data: MAVCeleb v1+v2+v3, 198 ids, multilingual
- Hardware: Google Colab Pro, NVIDIA Tesla T4
- Results: TBD

## Key Findings
1. ECAPA-TDNN upgrade: +65% Romanian similarity over ResNet18
2. SupCon requires diverse multilingual data to outperform OPL
3. Multilingual training: +154% Urdu similarity improvement
4. Zero-shot Romanian Rank-1 remains 0% across all configurations
5. Linguistic proximity paradox: Romanian (0.2532) > Urdu (0.0972) despite being unseen