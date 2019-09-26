import numpy as np
import pandas as pd 
import torch

df_train = pd.read_csv('mnist_train.csv', sep=',')
train_data = df_train.to_numpy()

learning_rate = 0.01
D_in, H1, H2, D_out = 784, 100, 50, 10
model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H1),
        torch.nn.ReLU(),
        torch.nn.Linear(H1, H2),
        torch.nn.ReLU(),
        torch.nn.Linear(H2, D_out)
    )

# Train the model
for row in train_data:
  label, data = row[0], row[1:]/255
  label_as_array = np.zeros(10)
  label_as_array[label] = 1

  # Make a prediction and compare to target
  target = torch.from_numpy(label_as_array).float()
  prediction = model(torch.from_numpy(data).float())
  loss = ((prediction - target)**2).mean()

  # Backward propagation
  model.zero_grad()
  loss.backward()
  with torch.no_grad():
    for param in model.parameters():
      param -= learning_rate * param.grad

# Test the predictive powers
df_test = pd.read_csv('mnist_test.csv', sep=',')
test_data = df_test.to_numpy()

correct = 0
attempts = 0

for row in test_data:
  label, data = row[0], row[1:]/255

  # Make a prediction and compare to target
  prediction = model(torch.from_numpy(data).float())
  best_guess = torch.argmax(prediction)

  attempts += 1
  if label == best_guess:
    correct += 1

print(attempts)
print(correct)
print(correct/attempts * 100)
  

