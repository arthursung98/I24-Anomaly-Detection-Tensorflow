# import tensorflow as tf
import preprocess_data

# (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
# print('Shape of X_train: ', X_train.shape)
# print('Shape of y_train: ', y_train.shape)
# print('Shape of X_test: ', X_test.shape)
# print('Shape of y_test: ', y_test.shape)

dataset = preprocess_data.get_x_train('TM_1000_GT.csv')
print('Shape of my dataset : ', dataset.shape)
