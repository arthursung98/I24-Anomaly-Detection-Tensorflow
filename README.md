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

## 2. The Training Data
As of now, we define a single **trajectory** to be *n* number of consecutive x axis coordinates from a single vehicle.
For example, let *n* be 100 as it is in the codebase. The trajectory will then become 100 consecutive coordinates of a car.
Therefore each trajectory is a numpy array of shape (100), with each element being a floating number that represents the x position.

Each trajectory is also associated with a quality rating. If the trajectory is perfect and without any missing coordinates, the quality number will be 1.0.
For now, if 10% of the trajectory data is missing, then we let the quality number to be 0.9.

With this association of trajectory and quality rating, we will train a supervised learning model that can rate any given trajectory in the range of 0 ~ 1.

To load the data, use the **load_data()** method in preprocess.py. Similar to tensorflow.keras.MNIST.load_data(), **load_data()** will return a tuple of 2 numpy arrays 
of size ( ? , 100), and ( ? ). The size of the arrays can vary depending on which files we process, which is why it has been denoted ?.