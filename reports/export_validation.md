# Model Export and Numerical Validation Report

## 1. Export Artifact Verification (Rule 56)

| Model | Parameters | PyTorch Size | ONNX Size | < 2 MB Limit | Max Parity Error | Verdict |
|---|---|---|---|---|---|---|
| **VelocityEstimatorNet** | 79,394 | 323.9 KB | 314.9 KB | ✅ PASS | 3.80e-06 | ✅ PASS |
| **IMUDenoiseNet** | 53,414 | 221.7 KB | 209.2 KB | ✅ PASS | 4.10e-06 | ✅ PASS |

## 2. TFLite Runtime Status (Rule 54)

- **TFLite Status**: UNSUPPORTED_IN_ENVIRONMENT (TensorFlow / tflite_runtime not installed)
- **Deployment Path**: Use ONNX Runtime Mobile for on-device Android/ARM deployment.
