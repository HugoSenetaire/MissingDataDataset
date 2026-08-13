# MissingDataDataset

A PyTorch dataset library for benchmarking methods that handle **missing features** -- both artificially masked and real-world incomplete datasets.

## Overview

This package provides a unified interface for loading tabular and image datasets with missing data. It supports:

- **Fixed datasets** with pre-defined missingness patterns
- **Free datasets** where missingness masks are generated on-the-fly (MCAR, MAR, MNAR)
- YAML-based configuration for reproducible experimental setups
- Integration as a git submodule in other projects

## Structure

```
Dataloader/        # PyTorch DataLoader utilities
FixDatasets/       # Datasets with fixed missing patterns
FreeDatasets/      # Datasets with configurable missingness
MaskDataset/       # Missingness mask generation (MCAR, MAR, MNAR)
YAMLExamples/      # Example configuration files
prepare_data.py    # Script to download and prepare datasets
```

## Installation

```bash
# As a standalone package
git clone https://github.com/HugoSenetaire/MissingDataDataset.git

# As a submodule in another project
git submodule add https://github.com/HugoSenetaire/MissingDataDataset.git
```

## Usage

```python
from MissingDataDataset import prepare_data

# Load a dataset with MCAR missingness at 30%
dataset = prepare_data.get_dataset(
    dataset_name="UCI_boston",
    missing_rate=0.3,
    missing_mechanism="MCAR"
)
```

See `YAMLExamples/` for configuration templates.
