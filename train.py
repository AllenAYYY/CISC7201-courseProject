# -*- encoding: utf-8 -*-
"""

@File    :   train.py  
@Modify Time : 2023/12/11 22:36 
@Author  :  Allen.Yang  
@Contact :   MC36514@um.edu.mo        
@Description  : Model Train File

"""
# import package begin
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from data_loader import data_loader
from model import StockPredictionModel
# import package end

# file path begin
excel_file_path = './datasets/ChinaUnicom2023.xlsx'
model_path = './model/stock_model.pth'
# file path end

# size begin
input_size = 1
hidden_size = 64
output_size = 1
# size end

# data loader begin
df_selected = data_loader(excel_file_path)
dates = df_selected['date']
prices = df_selected['price']
# data loader end
def train_model(model, train_data, num_epochs=20, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_data:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_data)
        tqdm.tqdm.write(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    print('Training finished.')
    torch.save(model.state_dict(), model_path)

def predict(model_path,input=prices):
    model = StockPredictionModel(input_size, hidden_size, output_size)

    # 加载保存的模型参数
    model.load_state_dict(torch.load(model_path))
    model.eval()


    predict_data_tensor = torch.tensor(input.values, dtype=torch.float32).view(-1, 1, 1)

    predicted_prices = model(predict_data_tensor)
    return predicted_prices


if __name__ == '__main__':
    model = StockPredictionModel(input_size, hidden_size, output_size)
    train_prices = prices[:int(len(prices) * 0.9)]  # 使用前百分之90的数据作为训练数据
    train_prices_tensor = torch.tensor(train_prices.values[:-1], dtype=torch.float32).view(-1, 1, 1)
    train_labels_tensor = torch.tensor(train_prices.values[1:], dtype=torch.float32).view(-1, 1, 1)
    train_data = list(zip(train_prices_tensor, train_labels_tensor))
    train_model(model, train_data)
