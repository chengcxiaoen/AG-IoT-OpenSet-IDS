# Reproducibility protocol

## Scope

The release uses exact equality over all observable fields except the target label and `is_attack` to form duplicate groups. Whole groups are assigned to exactly one partition. This prevents an exactly duplicated observable record from crossing train and test, but it is **not** a temporal, device-independent, or cross-dataset evaluation.

The paper configuration is `configs/paper_final_legacy8.json`:

- Dataset: Farm-Flow, with `BotNet DDoS` explicitly excluded because it has 64 rows.
- Primary unknown attacks: `Arp Spoofing`, `Port Scanning`.
- Stress unknown attacks: `Arp Spoofing`, `Port Scanning`, `ICMP Flood`.
- Seeds: 42, 52, 62, 72, 82.
- Known groups: nominal 64/8/8/20% train/validation/calibration/test partitions.
- Training uses no sampling or row cap; unknown attack groups enter test only.

## Installation

Use Python 3.10. The server setup installs pinned dependencies, then the native `libmr==0.1.9` dependency needed for OpenMax. A C++ compiler is required.

```bash
bash SETUP_SERVER.sh
```

## Run the grouped release protocol

```bash
bash RUN_GROUPED_SERVER.sh --dataset /absolute/path/to/Farm-Flows.csv \
  --output /absolute/path/to/ag_iot_grouped_results
```

The release driver validates the CSV checksum configured by the caller, trains each predefined scenario/seed in an isolated process, records group audits, and produces metrics and aggregate CSV files. It never averages selectively over only favorable or feasible seeds.

## Expected outputs

- `grouped_runs/<scenario>/seed<seed>/group_audit.json`: group counts, duplicate audit, class distributions, and partition assignments.
- `grouped_runs/<scenario>/seed<seed>/metrics.json`: frozen-threshold ablations.
- `matched/grouped/<scenario>/seed<seed>/metrics.json`: post-hoc comparisons calibrated only on known data.
- `aggregate_results.csv`, `grouped_frozen_ablation_per_seed.csv`, and `SUMMARY_CN.md`: machine-readable and human-readable summaries.

## Reproduction limits

Floating-point libraries, GPUs, TensorFlow kernels, and nondeterministic operations can cause small numerical differences. A successful rerun should be assessed from the recorded protocol, checksums, split audit, and broadly matching results, not a demand for bitwise-identical metrics. No claim is made for edge-device latency, temporal transfer, cross-device transfer, or cross-dataset transfer.
