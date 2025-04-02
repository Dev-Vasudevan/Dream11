
import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from DataLoader_Model import FantasyData, BattingModel
# dataset_path = kagglehub.dataset_download('sukhdayaldhanday/dream-11-fantasy-points-data-of-ipl-all-seasons')
dataset_path = '/home/dev/.cache/kagglehub/datasets/sukhdayaldhanday/dream-11-fantasy-points-data-of-ipl-all-seasons/versions/1'

print('Data source import complete.')
dataset_path

def train(model, train_data_loader, optimizer, loss_fn):
    total_loss = 0.0
    model.train()  # Set the model to training mode

    for data in train_data_loader:
        inputs, labels = data  # Unpack inputs and labels
        inputs = inputs.detach()
        labels = labels.detach()
        optimizer.zero_grad()  # Clear gradients from previous step

        outputs = model(inputs)  # Forward pass
        loss = loss_fn(outputs, labels)  # Compute loss

        loss.backward()  # Backward pass (no need for retain_graph=True)
        optimizer.step()  # Update weights

        total_loss += loss.item()  # Accumulate loss

    # Compute average loss over all batches
    avg_loss = total_loss / len(train_data_loader)

    return avg_loss
def evaluate(model, eval_data_loader ):
    model.eval()  # Set model to evaluation mode
    total_error = 0  # Assuming 4 output columns
    total_samples = 0

    with torch.no_grad():  # Disable gradient computation
        for data in eval_data_loader:
            inputs, labels = data
            inputs = inputs.detach()
            labels = labels.detach()

            # Forward pass
            outputs = model(inputs)

            # Calculate squared error for each sample and each output
            error = (outputs - labels)**2

            # Sum errors across batches
            total_error += error.sum(dim=0)
            total_samples += labels.size(0)

    # Calculate average error for each column
    avg_error = total_error / total_samples

    return torch.sqrt(avg_error)

df = pd.read_csv(f'{dataset_path}/Batting_data.csv')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = FantasyData(device,df)
# data.preprocess_batting(df)
train_data_loader = DataLoader(data, batch_size=1024,shuffle=True)
criterion  = nn.MSELoss()
model = BattingModel().to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.01)
loss_list = []

model.load_state_dict(torch.load("../../models/batting_1.pth"))


epoch = 650
count = 0
prev = float('inf')
model.train()
for iter in range(epoch):

    avg_loss = train(model, train_data_loader, optimizer, criterion)

    if iter%20 ==0 :
        if prev<avg_loss:
            count+=1
        # print(evaluate(model, train_data_loader) , avg_loss , iter )
        prev = avg_loss
    if iter%100==0:
        print(avg_loss , iter )
    loss_list.append(avg_loss)
    if count > 30:
        break
print(avg_loss)
plt.plot(loss_list)
plt.title('Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

