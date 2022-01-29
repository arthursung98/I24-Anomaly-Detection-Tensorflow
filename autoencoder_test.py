import tensorflow as tf
from autoencoder_train import *

# The size of the x_test data represent the number of cars.
# Load the train data & test data by calling the data processing helper file.
# TODO Create a helpfer file for data processing.
x_train, x_test = 0, 0
training_epochs = 50


# Train Autoencoder model.
AutoEncoder_model = AutoEncoder(x_test.__sizeof__())

train_data = tf.data.Dataset.from_tensor_slices(x_train)
train_data = train_data.shuffle(x_train_size).batch(batch_size)

for epoch in range(training_epochs):
    for batch_x in train_data:
        _, current_loss = train_step(AutoEncoder_model, batch_x), mse_loss(
            AutoEncoder_model(batch_x), batch_x)

    if epoch % display_step == 0:
        print(f"Epoch #{epoch + 1} : Loss = {current_loss}")

# Test using Autoencoder model.
# TODO Need to create a function that determines the statistic quality of the test data and
# whether to disregard or not.
compute_accuracy(AutoEncoder_model(x_test), x_test)
