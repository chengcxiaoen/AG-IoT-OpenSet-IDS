# Dataset access and use

This repository does **not** redistribute Farm-Flow. Obtain it only from the original record:

- Farm-Flow | AG-IoT Security: Intrusion Detection in Smart Agriculture Dataset, [Zenodo record 10964648](https://zenodo.org/records/10964648), DOI [10.5281/zenodo.10964648](https://doi.org/10.5281/zenodo.10964648).

Before use, read the dataset landing page, citation information, license, and any access terms supplied by the original authors. Cite the dataset paper and record in derivative work.

## Expected local layout

```text
AG-IoT-OpenSet-IDS/
  datasets/
    Farm-Flow/
      Farm-Flows.csv     # downloaded separately; ignored by Git
```

The paper configuration uses the `traffic` column as the label. It explicitly excludes the 64 `BotNet DDoS` rows and retains the other eight labels. No class is capped or resampled in the paper protocol.

The preprocessing and split audit are generated from the downloaded CSV. Do not upload raw data, derived row-level predictions, or logs to a public repository unless the applicable terms explicitly permit it.
