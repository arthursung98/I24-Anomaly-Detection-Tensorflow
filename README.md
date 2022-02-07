# I24 Trajectory Quality Model

## 1. Local Directory & File Setup
1. Access the Box folder at https://vanderbilt.app.box.com/folder/155574990234
2. Download the **TM_chunk0_ratio0_GT.csv** file.
3. Your directory should look like
```
YOUR LOCAL FOLDER
└─── csv_data
|   |   TM_chunk0_ratio0_GT.csv
|   |   TM_chunk10_ratio0_GT.csv (If using chunk 10 version)
|   |   ...
│
└─── src
		└─  __pycache__
		└─  .ipynb_checkpoints
    |   ANN_test.py
    |   ANN_train.py
    |   preprocess.py
		│   README.md
    |   ...
```

## 2. Understanding the Training Data