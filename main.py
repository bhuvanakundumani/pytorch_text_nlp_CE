import pandas as pd 
import numpy as np
import torch
import torchtext
import nltk
import json
from tqdm import tqdm
from models.LSTM import LstmClassifier
from models.LSTM_Attn import LstmAttentionModel
from models.LSTM_MHAttn import LstmMultiAttentionModel
from preprocess import preproc_load
from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from plotloss import plot_function

file_train = "data/processed_data/train.csv"
file_test ="data/processed_data/test.csv"

data_dict = preproc_load(file_train, file_test)
train_data = data_dict["data_train"]
test_data = data_dict["data_test"]
valid_data = data_dict["data_valid"]
epochs = 4
lr = 0.01

output_size = 2
hidden_size = 100
embedding_length = 100
batch_size = 64
model = LstmMultiAttentionModel(batch_size, output_size, hidden_size, data_dict["vocab_size"], embedding_length, True, data_dict["pretrained_embeddings"])
#model = LstmClassifier(batch_size, output_size, hidden_size, data_dict["vocab_size"], embedding_length, True, data_dict["pretrained_embeddings"])
#model = LstmAttentionModel(batch_size, output_size, hidden_size, data_dict["vocab_size"], embedding_length, True, data_dict["pretrained_embeddings"])
optimizer = torch.optim.Adam(model.parameters(), lr)
loss_function = F.cross_entropy
train_loss_plot = []
valid_loss_plot = []
valid_loss_optim = float('inf')

def train_model(model, train_iter, optimizer):  
    train_epoch_loss = 0
    train_true = []
    train_pred = []
    optimizer = optimizer
    steps = 0
    model.train()

    for idx, batch in tqdm(enumerate(train_iter)):
        text, target = batch.text, batch.target 
        optimizer.zero_grad()
        output = model(text)
        loss = loss_function(output,target.long())
        loss.backward()
        optimizer.step()
        train_epoch_loss  += loss.item()
        print(train_epoch_loss)
        pred = torch.max(output,1)[1]
        train_pred.extend(pred.tolist())
        train_true.extend(target.float().tolist())
    
    accuracy_train = accuracy_score(train_true, train_pred)
    #print(f"train loss {train_epoch_loss}, and train_accuracy {accuracy_train}")
    return train_epoch_loss, accuracy_train

def valid_model(model, valid_iter):
    valid_epoch_loss = 0
    valid_true = []
    valid_pred = []
    model.eval()
    for idx,batch in tqdm(enumerate(valid_iter)):
        text = batch.text
        target = batch.target
        with torch.no_grad():
            valid_true.extend(target.float().tolist())
            output = model(text)
            loss = loss_function(output, target.long())
            valid_epoch_loss  += loss.item()
            pred = torch.max(output,1)[1]
            valid_pred.extend(pred.tolist())

    accuracy_valid = accuracy_score(valid_true,valid_pred)
    return valid_epoch_loss, accuracy_valid

def test_model(model, test_iter):
    test_epoch_loss = 0
    test_true = []
    test_pred = []
    model.eval()
    for idx,batch in tqdm(enumerate(test_iter)):
        text = batch.text
        target = batch.target
        target = target.float()
        with torch.no_grad():
            test_true.extend(target.float().tolist())
            output = model(text)
            loss = loss_function(output, target.long())
            test_epoch_loss += loss.item()
            pred = torch.max(output,1)[1]
            test_pred.extend(pred.tolist())

    accuracy_test = accuracy_score(test_true,test_pred)
    return test_epoch_loss, accuracy_test

train_iter = torchtext.data.Iterator(dataset=train_data, batch_size=batch_size, train=True, shuffle=True, sort=False)
valid_iter = torchtext.data.Iterator(dataset=valid_data, batch_size=batch_size, train=True, shuffle=True, sort=False)
test_iter = torchtext.data.Iterator(dataset=test_data, batch_size=batch_size, train=True, shuffle=True, sort=False)

for epoch in range(epochs):
    print("********************************************************")
    print(f"epoch number {epoch}")
    train_loss, train_accu  = train_model(model, train_iter,optimizer)
    valid_loss, valid_accu = valid_model(model, valid_iter)
    train_loss_plot.append(train_loss)
    valid_loss_plot.append(valid_loss)
    if valid_loss < valid_loss_optim:
        valid_loss_optim = valid_loss
        
        torch.save(model.state_dict(), 'model_files/model.pkl' )
    print(f"Train loss:{train_loss:.3f}, Train accu:{train_accu:.3f}, Valid loss:{valid_loss:.3f}, Valid accu:{valid_accu:.3f}")
    
# Plot Loss vs epochs and store the graph in models/images/Loss_graph.png
plot_function(train_loss_plot, valid_loss_plot)


model.load_state_dict(torch.load('model_files/model.pkl'))
test_loss, test_accu = test_model(model, test_iter)
print("___________________________________________________")
print(f"Test Loss:{test_loss:.3f}, Test acc:{test_accu:.3f}")

# Model metric details stored in models/metrics/model_metrics.json
metrics_details = {
    'Epochs' : epochs,
    'Train Loss' : train_loss_plot,
    'Valid Loss' : valid_loss_plot,
    'Test Loss' : test_loss,
    }

with open('model_files/metrics/model_metrics.json', 'w') as json_file:
    json.dump(metrics_details, json_file)

with open('model_files/metrics/model_metrics.json') as json_file:
    data = json.load(json_file)
    print(data)



