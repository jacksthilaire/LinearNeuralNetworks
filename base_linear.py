import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import random
import csv

#########################################################################
# Generic neural net - input an array of [data_tensor, correct_answer]  #
# where data_tensor is a tensor full of doubles (statistical values),   #
# correct_answer is an integer representing a classification            #
#########################################################################
# Jack St. Hilaire 2020

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Using GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

# DATA LOADING #
def load_data_from_csv(filename):
  data_list = []
  data_file = csv.reader(open(filename), delimiter=',')

  # iterate over each line and read it in column by column
  skip_lines = 50
  for line in data_file:
    if skip_lines >= 0:
      skip_lines = skip_lines - 1
    else:
      the_data = []
      the_result = []

      # load the data and result (the last line of the csv)
      for i in range(1103):
        if i >= 5:
          the_data.append(float(line[i]))

      the_result.append(int(line[1103]))
      
      #create tensors from data and apppend the data point to the list
      train_tensor = torch.tensor(the_data)
      correct = torch.tensor(the_result)
      data_list.append([train_tensor, correct])

  return data_list


# attempt to fill training and testing datasets from csv
complete_data = []             
set1 = []
set2 = []     
set3 = []
set4 = []
set5 = []

try:
  set1 = load_data_from_csv("14-15-dataset.csv")
  set2 = load_data_from_csv("15-16-dataset.csv")
  set3 = load_data_from_csv("16-17-dataset.csv")
  set4 = load_data_from_csv("18-19-dataset.csv")
  # set5 = load_data_from_csv("19-20-dataset.csv")
  
  for i in range(0, len(set1)):
    complete_data.append(set1[i])
  for i in range(0, len(set2)):
    complete_data.append(set2[i])
  for i in range(0, len(set3)):
    complete_data.append(set3[i])
  for i in range(0, len(set4)):
    complete_data.append(set4[i])
  for i in range(0, len(set5)):
    complete_data.append(set5[i])

  random.shuffle(complete_data)
  print("Data loaded.")
except:
  print("Error loading data.")

# split
trainset = []
testset = []

print("Data count: ", len(complete_data))
data_index = 0
for data_point in complete_data:
  if (data_index % 2) == 0:
    trainset.append(data_point)
  else:
    testset.append(data_point)
  data_index += 1


# neural network class
class Net(nn.Module):

    # initialization of the layers
    def __init__(self):
        super().__init__()

        # 4 fully connected linear layers
        self.fc1 = nn.Linear(1098, 32) # input = 1098 float values
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 4) # output = [1,2,3]

    # how data flows through the layers
    # rectified linear activation function

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        # after a forward pass return the softmax for the output layer (prediction)
        return F.log_softmax(x, dim=1)


# initialize net and optimizer
net = Net().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.002)  # loss rate default = 0.001


# TRAINING
def train_net(epochs):

    # set to train mode and determine epoch count
    net.train() 
    EPOCHS = epochs

    for epoch in range(EPOCHS):
        for data in trainset:
            # data = data and a label
            X, y = data
            X, y = X.to(device), y.to(device)

            # start gradient at 0, which contains loss
            net.zero_grad()
            output = net(X.view(-1, 1098).float())
            loss = F.nll_loss(output, y) # calculate loss, using nll because it is an integer 'guess'

            loss.backward()  # back-propagation
            optimizer.step()  # adjust the weights with optimizer
        
        # save the model, print the loss
        model_save_name = 'nba001.pt'
        path = F"{model_save_name}" 
        torch.save(net.state_dict(), path)

       #  loss_output = loss

        print(loss)  # print the loss each epoch

        # also save to drive
        drive_save_name = 'nba001.pt'
        drive_path = F"/content/gdrive/My Drive/{drive_save_name}" 
        torch.save(net.state_dict(), drive_path)

# TESTING
def test_net():
    # counters
    correct = 0
    total = 0

    net.eval() # eval mode

    # don't use gradient, don't want to adjust weights when testing

    with torch.no_grad():
      for data in testset:
        X, y = data
            
        # iterate though 'guesses'
        output = net(X.view(-1, 1098).float().to(device))

        for idx, i in enumerate(output):
          # if the class with the highest guess is equal to the y value (actual) at that index
          if torch.argmax(i) == y[idx]:
              correct += 1
          total += 1

    accuracy = round(correct / total, 3) * 100
    # print("Accuracy: %", accuracy)
    return accuracy

# attempt to load the model
try:
  model_save_name = 'nba001.pt' # input("Enter model name: ")
  path = F"{model_save_name}"
  net.load_state_dict(torch.load(path))
  print("Model loaded.")
except:
  print("Unable to load model.")

# user selection of task
choice = input("Enter '1' for training or '2' for testing: ")
if choice == "2":

  accuracy = 0.0
  for i in range(0, 10):
    accuracy += test_net()
  accuracy = accuracy / 10
  print("Accuracy: ", accuracy)
    
else:
    epoch_num = int(input("Enter number of epochs: "))
    train_net(epoch_num)


# epoch training strategy --> 64, test, 32, test, 16, test, 8, test, 4/2, test, 1, 1,1....

#-------------------------------------------------------------------------------------------------------------#
# code for loading a standard dataset (mnist)

# train = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
# test = datasets.MNIST("", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

# trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
# testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

from google.colab import drive
drive.mount('/content/drive')

from google.colab import drive
drive.mount('/content/gdrive')