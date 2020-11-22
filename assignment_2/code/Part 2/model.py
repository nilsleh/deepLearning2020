# MIT License
#
# Copyright (c) 2019 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
###############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(TextGenerationModel, self).__init__()
        self.seq_length = seq_length
        self.embedding_size = int(lstm_num_hidden / 2)
        
        self.embedding = nn.Embedding(vocabulary_size, self.embedding_size)

        self.lstm = nn.LSTM(input_size=self.embedding_size, hidden_size = lstm_num_hidden, num_layers=lstm_num_layers)

        self.linear = nn.Linear(lstm_num_hidden, vocabulary_size)

         
    def forward(self, x):
        embed_x = self.embedding(x)

        output, (h_state, c_state) = self.lstm(embed_x)
        output = self.linear(output)

        # y_hat_t = []
        
        # # for time_step in range(self.seq_length): 
        # for time_step in range(output.shape[0]):
            
        #     seq_output = output[time_step, :, :]
        #     y_hat_seq = torch.nn.functional.log_softmax(seq_output, dim=1)
        #     y_hat_t.append(y_hat_seq)

        # y_hat_t = torch.stack(y_hat_t)

        return output
