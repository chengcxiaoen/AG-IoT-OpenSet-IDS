# Results reported in the manuscript

All figures below are percentage mean ± sample standard deviation over the five predefined seeds. They are reported to describe the paper; they are not a guarantee that another environment will match every decimal.

## Frozen paired ablation: primary holdout

| Method | Open-set macro-F1 | Unknown recall | Benign FAR |
|---|---:|---:|---:|
| Closed-set MLP | 68.31 ± 0.84 | 0.00 ± 0.00 | 0.09 ± 0.06 |
| Context only | 71.72 ± 0.91 | 2.49 ± 0.95 | 0.21 ± 0.11 |
| Global MSP | 83.40 ± 6.49 | 68.00 ± 31.14 | 0.15 ± 0.06 |
| Class-specific MSP | 85.25 ± 6.07 | 73.04 ± 31.76 | 5.16 ± 0.54 |
| Global MSP + context | 86.35 ± 5.71 | 70.42 ± 30.54 | 0.26 ± 0.10 |
| Class-specific MSP + context | **86.99 ± 6.00** | **74.27 ± 31.76** | 5.21 ± 0.51 |

The best macro-F1 and unknown recall do not imply the best FAR. This is an explicit classification–rejection trade-off, not a universal superiority claim.

See the paper and generated aggregate files for the stress scenario and post-hoc comparison protocol.
