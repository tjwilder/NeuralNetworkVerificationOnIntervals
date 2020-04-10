import functools
import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Setting the random seed for uniformity - used for all algorithms invoked
# in pytorch
torch.manual_seed(2)

PATH = './data'

# Fully connected neural network with one hidden layer


class NeuralNet(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(NeuralNet, self).__init__()
    # initializes a weight matrix of shape (hidden_size * input_size)
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU()

    # initializes a weight matrix of shape (num_classes * hidden_size)
    self.fc2 = nn.Linear(hidden_size, num_classes)

  # invoke all functions in order to compute the final outcome/digit class
  # in a forward pass
  def forward(self, x):
    # applies linear transformation to the input data, y = wx + b
    out = self.fc1(x)
    # applies the RELU activation function on the hidden layer
    out = self.relu(out)

    # applies linear transformation to the hidden layer to map to the
    # output layer
    out = self.fc2(out)
    return out


class Range():
  def __init__(self, lower_bound, upper_bound):
    self.lower = lower_bound
    self.upper = upper_bound

  def __repr__(self):
    return f'[{self.lower:.4f}, {self.upper:.4f}]'

  def add(self, other):
    lower = self.lower + other.lower
    upper = self.upper + other.upper
    return Range(lower, upper)

  def add_scalar(self, scalar):
    lower = self.lower + scalar
    upper = self.upper + scalar
    return Range(lower, upper)

  def scale(self, factor):
    min_num = self.lower * factor
    max_num = self.upper * factor
    # Account for negative factors
    lower = min(min_num, max_num)
    upper = max(min_num, max_num)
    return Range(lower, upper)

  def relu(self):
    if self.upper < 0:
      return Range(0, 0)
    if self.lower > 0:
      return Range(self.lower, self.upper)

    return Range(0, self.upper)


def ranged_dot(ranges, w, b):
  # TODO: reduce redundancy here with one loop that calculates
  #   the final min/max of each
  # Scale all ranges by appropriate weights
  y = [[ranges[i].scale(w[j][i]) for i in range(len(ranges))]
       for j in range(len(w))]
  # Sum them together to complete the dot product
  y = [functools.reduce(lambda a, b: a.add(b), y[j])
       for j in range(len(y))]
  # Add bias
  y = [y[i].add_scalar(b[i]) for i in range(len(y))]
  return y


def forward_range(epsilon, x):
  ranged_x = [Range(p - epsilon, p + epsilon) for p in x]
  # applies linear transformation to the input data, y = wx + b
  w = fetch_weights(1)
  b = fetch_bias(1)
  y_1 = ranged_dot(ranged_x, w, b)

  # applies the RELU activation function on the hidden layer
  y_2 = [y.relu() for y in y_1]

  # applies linear transformation to the hidden layer to map to the
  # output layer
  w = fetch_weights(2)
  b = fetch_bias(2)
  # Scale all ranges by appropriate weights
  y_3 = ranged_dot(y_2, w, b)

  return y_3


def range_matches(ranges, label):
  label_min = ranges[label].lower
  return all([ranges[r].upper < label_min
              for r in range(len(ranges)) if r != label])


def init_model_parameters():
  global num_epochs, batch_size, model, criterion, optimizer
  # Hyper-parameters
  # The dataset contains gray-scale images of pixel dimensions 28*28
  # Input layer contains 784 nodes, one node for each input feature
  #   (or pixel in the image)
  # Hidden layer contains 500 nodes
  # Output layer contains 10 nodes (one for each class representing a digit
  # between 0-9)
  input_size = 784
  hidden_size = 10
  num_classes = 10
  num_epochs = 5
  batch_size = 100
  learning_rate = 0.001
  model = NeuralNet(input_size, hidden_size, num_classes).to(device)

  # Loss and optimizer
  # Cross Entropy loss for the predicted outcome vs actual result is
  # evaluated in the training phase after the forward pass
  criterion = nn.CrossEntropyLoss()
  # Adam stochastic optimization algorithm implements an adaptive learning
  # rate
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Load the MNIST dataset
def load_dataset():
  global train_dataset, test_dataset, train_loader, test_loader

  # MNIST dataset contains 60000 images in the training data and 10000 test
  # data images
  train_dataset = torchvision.datasets.MNIST(root='../../data',
                                             train=True,
                                             transform=transforms.ToTensor(),
                                             download=True)

  test_dataset = torchvision.datasets.MNIST(root='../../data',
                                            train=False,
                                            transform=transforms.ToTensor())

  # Data loader divides the dataset into batches of batch_size=100 that
  # can be used for parallel computation on multi-processors
  train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                             batch_size=batch_size,
                                             shuffle=True)

  test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)


def save_model():
  torch.save(model.state_dict(), PATH)


def load_model():
  model.load_state_dict(torch.load(PATH))
  model.eval()


def train():
  total_step = len(train_loader)
  # Iterate over all training data (600 images in each of the 100 batches)
  # in every epoch(5)
  for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
      # Move tensors to the configured device
      images = images.reshape(-1, 28 * 28).to(device)
      labels = labels.to(device)

      # Forward pass
      outputs = model(images)
      loss = criterion(outputs, labels)

      # manually set the gradients to zero
      optimizer.zero_grad()
      # compute the new gradients based on the loss likelihood
      loss.backward()
      # propagate the new gradients back into NN parameters
      optimizer.step()

      if (i + 1) % 100 == 0:
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(
            epoch + 1, num_epochs, i + 1, total_step, loss.item()))
    torch.save(model.state_dict(), 'model.ckpt')


def test():
  with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
      images = images.reshape(-1, 28 * 28).to(device)
      labels = labels.to(device)
      # pass the test images batch to the trained model to compute
      # outputs
      outputs = model(images)
      # fetching the class with maximum probability for every image in
      # the batch as the predicted label
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      # compute the total correctly predicted outcomes (when test image
      # label = predicted)
      correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(
        100 * correct / total))


# fetch the weights of the neural network returned as tensors
# pass layer as int (1: input to hidden or 2: hidden to output)
def fetch_weights(layer):
  # Weights between input layer and hidden layer (Tensor of shape: [#hidden,
  # 784] i.e. hidden layer size * input layer size)
  if layer == 1:
    return model.fc1.weight.data
  # Weights between input layer and hidden layer (Tensor of shape: [10,
  # #hidden] i.e. output layer size * hidden layer size)
  if layer == 2:
    return model.fc2.weight.data


def fetch_bias(layer):
  # Fetch bias term
  if layer == 1:
    return model.fc1.bias
  # Weights between input layer and hidden layer (Tensor of shape: [10,
  # #hidden] i.e. output layer size * hidden layer size)
  if layer == 2:
    return model.fc2.bias


def plot_dataset(image):
  plt.imshow(image[0][0], cmap='gray')
  plt.show()


def test_ranged(epsilon):
  robust = 0
  for i in range(len(test_dataset)):
    image = test_dataset[i][0]
    image = image.reshape(-1, 28 * 28).to(device)[0]
    label = test_dataset[i][1]
    ranges = forward_range(epsilon, image)
    # print(ranges)
    if range_matches(ranges, label):
      robust += 1
    # print(model.forward(image))
  print(robust, len(test_dataset))


def main():
  init_model_parameters()
  load_dataset()
  if len(sys.argv) > 1:
    if sys.argv[1] == 'train':
      train()
      print(fetch_weights(1))
      save_model()
    else:
      load_model()
  else:
    load_model()
  # test()
  test_ranged(0.001)

  # Print first image in the test dataset
  # for i, image in enumerate(test_dataset):
  #   if i == 0:
  #     plot_dataset(image)


if __name__ == '__main__':
  main()
