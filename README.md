# AG-IoT Open-Set IDS

Reproducibility materials for **Lightweight Open-Set Intrusion Detection for Agricultural Internet of Things via Class-Specific Thresholds and Communication Context**.

> **Release status: pre-publication candidate.** Do not create a public release or add the URL to the paper until every item in [docs/PUBLICATION_CHECKLIST.md](docs/PUBLICATION_CHECKLIST.md) is confirmed by the corresponding authors.

The method combines a compact 256-128-64 multilayer perceptron (MLP), class-specific maximum softmax probability (MSP) rejection, and benign-gated communication context. It is evaluated on Farm-Flow with ARP spoofing and port scanning held out in the primary scenario, plus ICMP flooding in a stress scenario.

## What is included

- The frozen training, calibration, evaluation, and result-summary code.
- Exact-observable duplicate-group splitting used by the manuscript.
- Fixed paper configurations, five predefined seeds, unit tests, and server setup scripts.
- A GitHub Pages-ready project site in [`docs/`](docs/).

## What is deliberately not included

- The Farm-Flow CSV or any other dataset.
- Trained models, row-level predictions, server logs, cached features, and raw experiment outputs.
- External datasets or third-party source code.

Download Farm-Flow from its original [Zenodo record](https://zenodo.org/records/10964648), read its terms, and place `Farm-Flows.csv` at `datasets/Farm-Flow/Farm-Flows.csv`. See [docs/DATASET.md](docs/DATASET.md).

## Quick start

The tested environment is Linux, Python 3.10, TensorFlow 2.10.1, and an NVIDIA GPU. Install the pinned dependencies and LibMR:

```bash
bash SETUP_SERVER.sh
```

Then run the grouped protocol:

```bash
bash RUN_GROUPED_SERVER.sh --dataset /absolute/path/to/Farm-Flows.csv
```

This runs two pre-registered holdout scenarios and seeds 42, 52, 62, 72, and 82. It can take substantial GPU time. Results are written outside the repository when `--output` is supplied. Full instructions and limitations are in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Project website

`docs/index.html` is a static site ready for GitHub Pages. Push this folder to [chengxiaoen/AG-IoT-OpenSet-IDS](https://github.com/chengxiaoen/AG-IoT-OpenSet-IDS), then enable **Pages → Deploy from a branch → /docs**. See [docs/PROJECT_HOMEPAGE.md](docs/PROJECT_HOMEPAGE.md).

## Citation

Use [CITATION.cff](CITATION.cff). Replace the placeholder paper venue/DOI fields after acceptance or publication.

## License

The repository is prepared with an MIT license. The corresponding authors must confirm that all included code is author-owned or compatible with MIT before publication.
