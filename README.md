# AdFL In-browser Federated Learning for Online Advertisement - Model and Data samples

This repository contains the code and data sample for the AdFL research paper, published in ICWSM 2026. All feature names and data values have been obfuscated to protect privacy information while maintaining the scientific reproducibility of the results.

## Contents

- `model.py` - Main federated learning implementation
- `data_sample.csv` - Obfuscated data sample (2,000 records, 10 users, 200 samples each)
- `README.md` - This documentation file
- `requirements.txt` - Python package dependencies
- `LICENSE` - MIT License
- `CITATION.bib` - BibTeX citation for academic use
- `.gitignore` - Git ignore file for version control

## Features

The model uses **27 obfuscated features** plus a user identifier:

### Binary Features (4)
- `bin_1`, `bin_2`, `bin_3`, `bin_4` - Binary indicators (0/1)

### Numeric Features (14)
- `num_1` through `num_14` - Continuous numeric features
- All numeric values have been normalized to [0, 1] range
- Statistical properties are preserved while protecting original values

### Categorical Features (9)
- `cat_1` through `cat_9` - Categorical features
- Values have been hash-based obfuscated (e.g., `cat_1_a3f2b8c9`)

### Additional Fields
- `user_id` - Obfuscated user identifier, essential to split data per user. 
- `target` - Binary prediction target (0/1)

## Requirements

```bash
python >= 3.8
tensorflow >= 2.8
pandas >= 1.3
numpy >= 1.20
scikit-learn >= 1.0
```

Install dependencies:
```bash
pip install tensorflow pandas numpy scikit-learn
```

## Usage

### Quick Start

Default run will run the model for 5 users.
```bash
python model.py
```

Run the model with 2, 5, 10, 50, 100, or 500 users:
```bash
python model.py --num_users 2 --data_file data_sample.csv
python model.py --num_users 5 --data_file data_sample.csv
python model.py --num_users 10 --data_file data_sample.csv
```

Note that in AdFL paper, we run the model for 50, 100, and 500 users, however for demonstration here we provided 2, 5, and 10 users due to limited data samples publically available.

### User Configuration Options

The model file supports five user configurations:

| Users | Command | Early stopping patience | Early stopping start | Min samples/user |
|---:|---|---:|---:|---:|
| 5 | `python model.py --num_users 5 --data_file data_sample.csv --rounds 1000 --experiments 10` | 10 | Round 20 | 100 |
| 10 | `python model.py --num_users 10 --data_file data_sample.csv --rounds 1000 --experiments 10` | 15 | Round 30 | 100 |
| 50 | `python model.py --num_users 50 --data_file data_sample.csv --rounds 1000 --experiments 10` | 20 | Round 50 | 150 |
| 100 | `python model.py --num_users 100 --data_file data_sample.csv --rounds 1000 --experiments 10` | 30 | Round 100 | 110 |
| 500 | `python model.py --num_users 500 --data_file data_sample.csv --rounds 1000 --experiments 10` | 40 | Round 200 | 55 |

### Command Line Arguments

```
--num_users         Number of federated clients (2, 5, 10, 50, 100, or 500) [default: 5]
--data_file         Path to obfuscated CSV data file [default: data_sample.csv]
--rounds            Maximum training rounds [default: 1000]
--experiments       Number of independent experiments [default: 1]
--batch_size        Mini-batch size for training [default: 32]

# Differential Privacy (Optional)
--use_dp            Enable differential privacy protection
--l2_norm_clip      L2 norm clipping threshold [default: 1.0]
--noise_multiplier  Noise multiplier for DP [default: 0.1, higher=more privacy]
--num_microbatches  Number of microbatches for DP [default: 1]
```

## Differential Privacy Support

The model supports **standard DP-SGD (Differential Privacy Stochastic Gradient Descent)** to provide formal (ε, δ)-differential privacy guarantees during federated training.

**Implementation**: Uses `tensorflow-privacy` library (Abadi et al., 2016).  

### Installation

To use differential privacy, install tensorflow-privacy:
```bash
pip install tensorflow-privacy
# or
conda install -c conda-forge tensorflow-privacy
```

### Usage

```bash
# Run with standard DP-SGD enabled
python model.py \
    --num_users 50 \
    --rounds 1000 \
    --use_dp \
    --noise_multiplier 0.5 \
    --l2_norm_clip 1.0
```

**Important**: This uses the **standard DP-SGD algorithm** which provides formal privacy guarantees. The model automatically uses `model.fit()` for DP compatibility.

### DP Parameters

- **`--use_dp`**: Enables differential privacy (requires tensorflow-privacy)
- **`--l2_norm_clip`**: Maximum L2 norm of gradients (default: 1.0)
  - Clips gradients to bound sensitivity
  - Smaller values = more privacy but may slow convergence
- **`--noise_multiplier`**: Amount of noise added to gradients (default: 0.1)
  - Higher values = stronger privacy guarantees
  - Typical range: 0.1 to 2.0

### Privacy Budget

Differential privacy provides (ε, δ)-DP guarantees. The privacy budget depends on:
- Noise multiplier
- Number of training steps
- Dataset size

Lower ε means stronger privacy (typical target: ε < 10)

### Output Files

The model file generates:
- `{num_users}users_experiment_{id}.txt` - Final metrics for each experiment
- `training_YYYYMMDD_HHMMSS.log` - Detailed training logs

## Model Architecture

### Input Processing
1. **Categorical features** → Hash layer → Embedding layer (dim = min(50, bins/2))
2. **Numeric features** → Dense layer (64 units, ReLU)
3. **Binary features** → Dense layer (200 units, ReLU)

### Neural Network
- Concatenated embeddings
- Dense layers: 500 → 250 → 100 → 50 → 30 neurons (all ReLU)
- Output: Single sigmoid neuron (binary classification)

### Federated Learning
- **Algorithm:** Federated Averaging (FedAvg)
- **Client training:** One epoch per round on local data
- **Aggregation:** Simple weight averaging across all clients
- **Evaluation:** Validation loss for early stopping, test metrics (Loss, Accuracy, AUC) for reporting

## Data Format

The `data_sample.csv` file contains 1000 records with the following structure:

```csv
bin_1,bin_2,bin_3,bin_4,num_1,num_2,...,cat_1,cat_2,...,user_id,target
0,1,1,0,0.523,0.891,...,cat_1_a3f2b8c9,cat_2_7e4d1f5a,...,user_3a4e5f1e,0
1,1,0,1,0.234,0.456,...,cat_1_b8c2a9d3,cat_2_9f3e7b1c,...,user_b544f557,1
...
```

## Data Obfuscation

All data has been obfuscated for publication:

1. **Feature Names:** Replaced with generic names (bin_X, num_X, cat_X)
2. **Numeric Values:** Factorized to [0,1] range with small Gaussian noise
3. **Categorical Values:** Hash-based anonymization (MD5 truncated)
4. **User IDs:** Hash-based anonymization
5. **Target Labels:** Preserved as binary (0/1)

The obfuscation preserves:
- Statistical distributions
- Correlations between features
- Relative relationships within features
- Model training dynamics

## Example Results

After training, you will see output like:

```
Round 173, Global Model Metrics - Loss: 0.3227, Accuracy: 0.8848, AUC: 0.9301
```

Each experiment saves final metrics to a text file for analysis.

## Data Sample Statistics

- **Total Records:** 2,000
- **Unique Users:** 10
- **Samples per User:** 200 (balanced distribution)
- **Target Distribution:** 
  - Class 0: 611 records (30.5%)
  - Class 1: 1,389 records (69.5%)
- **Features:** 27 (4 binary + 14 numeric + 9 categorical) + 1 user_id + 1 target

**Note:**

1. The `user_id` column is included for proper user-based data splitting but is not used as a model feature.

2. The provided dataset contains 2,000 records from 10 users. Configurations with more than 10 users (50, 100, 500) would require a larger dataset and are included in the code for completeness but cannot be tested with the provided sample data.

## Privacy and Ethics

- All data has been anonymized and obfuscated
- No personally identifiable information (PII) is included
- Feature names and values do not reveal privacy information
- The obfuscation maintains scientific validity while protecting privacy
- Differential privacy support available for additional privacy guarantees

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{alemari_adfl_2026,
  title = {AdFL: In-Browser Federated Learning for Online Advertisement},
  author = {Ahmad Alemari, Pritam Sen, and Cristian Borcea},
  journal = {AAAI/ICWSM},
  year = {2026},
  note = {Paper associated with this code repository}
}
```

See [CITATION.bib](CITATION.bib) for the complete citation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Installation

For easy installation of all dependencies:

```bash
pip install -r requirements.txt
```

See [requirements.txt](requirements.txt) for the complete list of dependencies.

## Contact

For questions about the code or data, please open an issue on the GitHub repository.

---

**Note:** The provided data sample (2,000 records, 10 users) is for demonstration, validation, and reproducibility purposes. All data has been obfuscated to protect privacy information while maintaining the statistical properties necessary for scientific reproducibility.
