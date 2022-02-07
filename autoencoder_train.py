import tensorflow as tf
import preprocess

# 1. Load data from Ground Truth data.
x_train, y_train = preprocess.load_data('TM_1000_GT.csv')

# 2. Variable setup - Many are subject to change with experimentation
input_size = 50
output_size = 1
data_size = int(x_train.size / input_size)
batch_size = 100
num_epochs = 50
hidden1_size = 30
hidden2_size = 10
learning_rate = 0.02
display_step = 1

# 3. Transform x_train, y_train into tuple and shuffle the tuple.
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.shuffle(data_size).batch(batch_size)

def random_normal_initializer() :
	return tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None)

# 4. Define Artificial Neural Network model using keras.
class ANN(tf.keras.Model) :
	def __init__(self) :
		super(ANN, self).__init__()
		# Encoding Layers
		self.hidden_layer_1 = tf.keras.layers.Dense(hidden1_size,
																								activation='sigmoid',
																								kernel_initializer=random_normal_initializer(),
																								bias_initializer=random_normal_initializer())
		self.hidden_layer_2 = tf.keras.layers.Dense(hidden2_size,
																								activation='sigmoid',
																								kernel_initializer=random_normal_initializer(),
																								bias_initializer=random_normal_initializer())
		self.output_layer = tf.keras.layers.Dense(output_size,
																							activation=None,
																							kernel_initializer=random_normal_initializer(),
																								bias_initializer=random_normal_initializer())

	def call(self, x) :
		H1_output = self.hidden_layer_1(x)
		H2_output = self.hidden_layer_2(H1_output)
		logits = self.output_layer(H2_output)
		
		return tf.nn.softmax(logits)

# 5. Define the Loss Function(Mean Squared Error) for calculating cost.
@tf.function
def mse_loss(y_pred, y_true):
  return tf.reduce_mean(tf.pow(y_true - y_pred, 2))

# 6. Define Optimizer & Training Step
optimizer = tf.optimizers.RMSprop(learning_rate)

@tf.function
def train_step(model, x) :
	y_true = x
	with tf.GradientTape() as tape :
		y_pred = model(x)
		loss = mse_loss(y_pred, y_true)
	gradients = tape.gradient(loss, model.trainable_variables)
	optimizer.apply_gradients(zip(gradients, model.trainable_variables))