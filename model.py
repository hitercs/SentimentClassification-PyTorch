#-*- encoding: utf-8 -*-
import settings
import torch
import torch.nn as nn
torch.manual_seed(settings.seed)

class LSTM(nn.Module):
    """
        Implementation of Single-Directional LSTM for Sentiment Classification Task
        Final Representation is Upper most Layer LSTM at final timestep
        Then the representation -> Linear -> Softmax 
    """
    def __init__(self, vocab_size, input_size, hidden_size, num_layers, out_dim, dropout=0.50):
        super(LSTM, self).__init__()

        self.word_embed = nn.Embedding(num_embeddings=vocab_size,
                                       embedding_dim=input_size,
                                       padding_idx=settings.pad_idx)
        self.input_size = input_size

        # sentence encoder
        self.encoder = nn.LSTM(input_size=input_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               dropout=dropout,
                               batch_first=True,
                               bidirectional=False)
        self.linear = nn.Linear(hidden_size, out_dim)
        self.softmax = nn.Softmax()

    def _fetch(self, encoder_outputs, batch_sen_mask):
        """
        
        :param encoder_outputs:  (batch_size, sen_len, hidden_size), LSTM encoding results
        :param batch_sen_mask: (batch_size, sen_len, hidden_size): sentence length for each sentence
        :return: encode_result: (batch_size, hidden_size)
        """
        batch_size = encoder_outputs.size()[0]
        hidden_size = encoder_outputs.size()[2]
        return torch.masked_select(encoder_outputs, batch_sen_mask).view(batch_size, hidden_size)



    def forward(self, batch_sen_id, batch_sen_mask):
        """
        
        :param batch_sen_id: (batch_size, max_sen_length), Tensor for sentence sequence id
        :param batch_sen_mask: (batch_size, seq_len, hidden_size)
        :return: 
        """
        '''  Embedding Layer | Padding '''
        batch_word_embedding = self.word_embed(batch_sen_id)                    # (batch_size, sen_len, input_size)

        '''  LSTM Encoding '''
        encoder_outputs, _= self.encoder(batch_word_embedding)                  # (batch_size, sen_len, hidden_size)

        batch_sen_representation = self._fetch(encoder_outputs, batch_sen_mask) # (batch_size, hidden_size)
        class_prob = self.softmax(self.linear(batch_sen_representation))        # (batch_size, out_dim)
        return class_prob



