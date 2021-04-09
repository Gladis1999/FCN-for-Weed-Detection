class NeuralNetwork:
	def __init__(self, x, y):
		self.input      = x
		self.weights1   = np.random.rand(self.input.shape[1],4) 
		self.weights2   = np.random.rand(4,1)                 
		self.y          = y
		self.output     = np.zeros(self.y.shape)

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def sigmoid_derivative(self, x):
		return x * (1 - x)

	def feedforward(self,inputs):
		inputs = inputs.astype(float)
		self.layer1 = self.sigmoid(np.dot(inputs, self.weights1))
		self.output = self.sigmoid(np.dot(self.layer1, self.weights2))
		return self.output


	def backprop(self):
		d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * self.sigmoid_derivative(self.output)))
		d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * self.sigmoid_derivative(self.output), self.weights2.T) * self.sigmoid_derivative(self.layer1)))
		self.weights1 += d_weights1
		self.weights2 += d_weights2

	def train(self,iterations):

		for i in range(iterations):
			output=self.feedforward(self.input)
			self.backprop()


if __name__ == "__main__":
	train_inputs = np.array([[0,0,1],
								[1,1,1],
								[1,0,1],
								[0,1,1]])

	train_outputs = np.array([[0,1,1,0]]).T

	neural_network = NeuralNetwork(train_inputs, train_outputs)

	neural_network.train(1500)

	input_one = str(input("Input One: "))
	input_two = str(input("Input Two: "))
	input_three = str(input("Input Three: "))

	print("New Output data: ")
	output=neural_network.feedforward(np.array([input_one,input_two,input_three]))
	print(output)
	if(output>0.5):
		print(1)
	else:
		print(0)
