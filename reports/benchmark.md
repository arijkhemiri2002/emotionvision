CPU benchmark (efficientnet_b0, input 112x112)

| variant | size_mb | test_accuracy | test_macro_f1 | latency_ms_p50 | latency_ms_p95 | throughput_img_s | speedup_vs_pytorch |
|---|---|---|---|---|---|---|---|
| pytorch_fp32 | 16.37 | 0.8559 | 0.7819 | 30.08 | 33.37 | 117.9 | 1.0 |
| onnx_fp32 | 16.04 | 0.8559 | 0.7819 | 3.59 | 4.39 | 342.5 | 7.82 |
| onnx_int8 | 4.78 | 0.8349 | 0.7334 | 5.42 | 9.05 | 260.4 | 4.85 |

INT8 quantization ablation (validation set, kept: percentile_conv_x)

| INT8 config | val_accuracy | val_macro_f1 |
|---|---|---|
| minmax_all | 0.785 | 0.6748 |
| minmax_conv | 0.7944 | 0.6917 |
| percentile_conv | 0.818 | 0.7245 |
| percentile_conv_x | 0.8475 | 0.769 |
