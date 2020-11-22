"""
This module implements a LSTM model in PyTorch.
You should fill in code into indicated sections.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch


class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, num_classes,
                 batch_size, device):

        super(LSTM, self).__init__()
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        # assuming that input data is sequence_length x batch_size so transposed input
        # self.h_init = torch.zeros(hidden_dim, batch_size)
        print("Input DIM")
        print(input_dim)
        self.seq_length = seq_length - 1
        self.embedding_size = input_dim
        self.embedding = nn.Embedding(seq_length, self.embedding_size)

        self.W_gx = nn.Parameter(torch.empty(self.embedding_size, hidden_dim))
        nn.init.kaiming_normal_(self.W_gx)
        self.W_gh = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        nn.init.kaiming_normal_(self.W_gh)
        self.b_g = nn.Parameter(torch.zeros(1, hidden_dim))
        
        self.W_ix = nn.Parameter(torch.empty(self.embedding_size, hidden_dim))
        nn.init.kaiming_normal_(self.W_ix)
        self.W_ih = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        nn.init.kaiming_normal_(self.W_ih)
        self.b_i = nn.Parameter(torch.zeros(1, hidden_dim))

        self.W_fx = nn.Parameter(torch.empty(self.embedding_size, hidden_dim))
        nn.init.kaiming_normal_(self.W_fx)
        self.W_fh = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        nn.init.kaiming_normal_(self.W_fh)
        self.b_f = nn.Parameter(torch.zeros(1, hidden_dim))

        self.W_ox = nn.Parameter(torch.empty(self.embedding_size, hidden_dim))
        nn.init.kaiming_normal_(self.W_ox)
        self.W_oh = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        nn.init.kaiming_normal_(self.W_oh)
        self.b_o = nn.Parameter(torch.zeros(1, hidden_dim))

        self.W_ph = nn.Parameter(torch.empty(hidden_dim, num_classes))
        nn.init.kaiming_normal_(self.W_ph)
        self.b_p = nn.Parameter(torch.zeros(1, num_classes))

        self.h_init = nn.Parameter(torch.zeros(1, hidden_dim))
        self.c_init = nn.Parameter(torch.zeros(1, hidden_dim))

        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        ########################
        # PUT YOUR CODE HERE  #
        #######################

        # make embedding of input
        ### RANDOM COMBINATIONS ###
      
        #x = torch.LongTensor(x).type(torch.LongTensor)
        # x_embedded = self.embedding(x.type(torch.LongTensor))
        # print("Batch input shape")
        # print(x.shape)
        # print("batch input")
        # print(x)

        hidden_state = self.h_init
        cell_state = self.c_init

        embed_x = self.embedding(x.type(torch.LongTensor))
        # print("Embed Full")
        # print(embed_x.shape)
        for time_step in range(self.seq_length):
            # print(time_step)

            embed_seq = embed_x[:, time_step, :]
            # print("Embed shape")
            # print(embed_seq.shape)
            # print("W_gx")
            # print(self.W_gx.shape)
            
            g_t = torch.tanh(torch.mm(embed_seq , self.W_gx) + torch.mm(hidden_state, self.W_gh) + self.b_g)

            i_t = torch.sigmoid(torch.mm(embed_seq, self.W_ix) + torch.mm(hidden_state, self.W_ih) + self.b_i)

            f_t = torch.sigmoid(torch.mm(embed_seq, self.W_fx) + torch.mm(hidden_state, self.W_fh) + self.b_f)

            o_t = torch.sigmoid(torch.mm(embed_seq, self.W_ox) + torch.mm(hidden_state, self.W_oh) + self.b_o)

            cell_state = g_t * i_t + cell_state * f_t

            hidden_state = torch.tanh(cell_state) * o_t
    
            p_t = torch.mm(hidden_state, self.W_ph) + self.b_p

        y_hat_t = torch.nn.functional.log_softmax(p_t, dim=1)
        
        return y_hat_t
        ########################
        # END OF YOUR CODE    #
        #######################
