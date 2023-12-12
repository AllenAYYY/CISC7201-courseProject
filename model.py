# -*- encoding: utf-8 -*-
"""

@File    :   model.py.py  
@Modify Time : 2023/12/11 22:31 
@Author  :  Allen.Yang  
@Contact :   MC36514@um.edu.mo        
@Description  : Model File

"""
# import package begin
import torch
import torch.nn as nn
# import package end

class StockPredictionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(StockPredictionModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        _, hidden = self.gru(input)
        out = self.fc(hidden.squeeze(0))
        return out