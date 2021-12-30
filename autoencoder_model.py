import tensorflow as tf
import math

x_train = 0 # The x_train data will be DF that only holds true value.

# Variable setup - Many are subject to change with experimentation
entry_size, batch_size, input_size, hidden1_size, hidden2_size = 0,0,0,0,0

learning_rate = 0.02
display_step = 1

def random_normal_initializer() :
	return tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None)

# Define AutoEncoder model using keras.
class AutoEncoder(tf.keras.Model) :
	def __init__(self, num_cars) :
		super(AutoEncoder, self).__init__()

		self.entry_size = num_cars
		self.input_size = 15 * num_cars
		self.batch_size = 1
		hidden1_size = math.ceil(math.sqrt(input_size))
		hidden2_size = hidden1_size / 2

		# Encoding Layers
		self.hidden_layer_1 = tf.keras.layers.Dense(hidden1_size,
																								activation='sigmoid',
																								kernel_initializer=random_normal_initializer(),
																								bias_initializer=random_normal_initializer())
		self.hidden_layer_2 = tf.keras.layers.Dense(hidden2_size,
																								activation='sigmoid',
																								kernel_initializer=random_normal_initializer(),
																								bias_initializer=random_normal_initializer())
		# Decoding Layers
		self.hidden_layer_3 = tf.keras.layers.Dense(hidden1_size,
																								activation='sigmoid',
																								kernel_initializer=random_normal_initializer(),
																								bias_initializer=random_normal_initializer())
		self.output_layer = tf.keras.layers.Dense(input_size,
                                                activation='sigmoid',
                                                kernel_initializer=random_normal_initializer(),
                                                bias_initializer=random_normal_initializer())

	def call(self, x) :
		H1_output = self.hidden_layer_1(x)
		H2_output = self.hidden_layer_2(H1_output)
		H3_output = self.hidden_layer_3(H2_output)
		reconstructed_x = self.output_layer(H3_output)
		
		return reconstructed_x

# Define MSE Loss Function calculating cost.
@tf.function
def mse_loss(y_pred, y_true):
  return tf.reduce_mean(tf.pow(y_true - y_pred, 2))

# Define Optimizer & Training Step
optimizer = tf.optimizers.RMSprop(learning_rate)

@tf.function
def train_step(model, x) :
	y_true = x
	with tf.GradientTape() as tape :
		y_pred = model(x)
		loss = mse_loss(y_pred, y_true)
	gradients = tape.gradient(loss, model.trainable_variables)
	optimizer.apply_gradients(zip(gradients, model.trainable_variables))