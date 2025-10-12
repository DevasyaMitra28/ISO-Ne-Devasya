# ISO-Ne-Devasya

This repository contains analysis of energy offer data for DA and RT markets from 06/25/2025 to 06/29/2025.
The goal is to clean the data, collate the long and wide formats and perform exploratory analysis.

## Setup Instructions

1. **Clone the repository:**
    ```
    git clone https://github.com/DevasyaMitra28/ISO-Ne-Devasya.git
    cd ISO-Ne-Devasya
    ```
2. **Create and activate the conda environment:**
    ```
    conda env create -f environment.yml
    conda activate iso-ne-devasya
    ```

## How to Use

- Run `notebooks/cleaning.ipynb` to clean and validate raw offer data.
- Run `notebooks/collate.ipynb` to combine intermediate files into master datasets.
- Use the scripts in `src/` for additional data processing and automation as needed.

## Data Flow

- **data/raw/**: Store original downloaded CSV files here.
- **data/interim/**: Intermediate cleaned files are outputted here (cleaned and cleaned_no_neg).
- **data/processed/**: Final aggregated and analysed datasets are saved here.

## Contributor

Devasya Mitra
