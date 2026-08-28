# Customer Journey Analytics Prototype

This project analyzes an e-commerce clickstream to discover behavioral segments of shoppers and identify where they drop off in the purchase funnel. It uses the real Retailrocket dataset from Kaggle.

## Prerequisites
- Python 3.9+
- A Kaggle account to download the dataset

## Setup and Installation

1. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the package in editable mode:
   ```bash
   pip install -e .
   ```

3. Download the dataset:
   - **Option A (Manual):** Download the zip from the [Kaggle dataset page](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset), unzip it, and place `events.csv` in `data/raw/`.
   - **Option B (CLI):** Install Kaggle CLI and download directly:
     ```bash
     pip install kaggle
     kaggle datasets download -d retailrocket/ecommerce-dataset -p data/raw --unzip
     ```

## Phases

This project is built phase by phase. Currently implemented:
- **Phase 1**: Acquire & prepare the Retailrocket data. Run with `python src/cja/acquisition.py`.
