# Time Series Transformer for Performance Monitoring

This project implements a **Time Series Transformer** model for processing and analyzing performance monitoring data, specifically focusing on CPU and GPU usage patterns. The implementation includes both feature extraction using a Transformer and unsupervised clustering to group similar performance trends.

## Project Structure

# Time Series Transformer for Performance Monitoring

This project implements a **Time Series Transformer** model for processing and analyzing performance monitoring data, specifically focusing on CPU and GPU usage patterns. The implementation includes both feature extraction using a Transformer and unsupervised clustering to group similar performance trends.

## Project Structure

```plaintext
project_root/
│
├── src/
│   ├── dataloader.py          # Data loading and preprocessing
│   ├── transformer.py         # Transformer model definition
│   ├── train.py               # Training script
│   ├── inference.py           # Inference script
│
├── data/
│   ├── train/                 # Training data folder
│   │   ├── user1/
│   │   │   ├── Log20240801/
│   │   │   │   └── performance_monitor.csv
│   │   │   ├── Log20240802/
│   │   │   │   └── performance_monitor.csv
│   │   │   └── ...
│   │   └── user2/
│   │       ├── Log20240801/
│   │       │   └── performance_monitor.csv
│   │       └── ...
│   └── test/                  # Test data folder (similar structure as train)
│
├── README.md                  # Project documentation
└── requirements.txt           # Required Python packages
```
## Requirements
To set up the environment, you need to install the required Python packages:
pip install -r requirements.txt

## Data Preparation
The data should be organized in a multi-level folder structure. Each user's data is stored in separate folders, with daily logs further subdivided into subfolders containing performance_monitor.csv files. Each CSV file contains CPU and GPU usage data sampled every 3 seconds.

CSV File Format
Column 0: CPU utilization (0-100%)
Column 1: Internet usage (ignored during training)
Column 2: GPU utilization (0-100%)
Ensure that all training data is placed under the data/train directory, while test data should be under data/test.

## Training
To train the Time Series Transformer model, use the following command:
python train.py --n_clusters <number_of_clusters> --data_path <path_to_train_data> --debug
##  License
This project is open-source and available under the MIT License.
