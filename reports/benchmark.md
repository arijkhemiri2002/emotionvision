CPU benchmark (efficientnet_b0, input 224x224)

| variant | size_mb | test_accuracy | test_macro_f1 | latency_ms_p50 | latency_ms_p95 | throughput_img_s | speedup_vs_pytorch |
|---|---|---|---|---|---|---|---|
| pytorch_fp32 | 16.37 | 0.8652 | 0.7673 | 146.6 | 203.32 | 22.1 | 1.0 |
| onnx_fp32 | 16.04 | 0.8652 | 0.7673 | 22.64 | 42.05 | 72.1 | 6.08 |
| onnx_int8 | 4.78 | 0.7985 | 0.6812 | 25.71 | 42.46 | 42.3 | 6.17 |

INT8 quantization ablation (validation set, kept: percentile_conv_x)

| INT8 config | val_accuracy | val_macro_f1 |
|---|---|---|
| minmax_all | 0.7753 | 0.6554 |
| minmax_conv | 0.7809 | 0.6558 |
| percentile_conv | 0.7734 | 0.6672 |
| percentile_conv_x | 0.8145 | 0.7246 |
