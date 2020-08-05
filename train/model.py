import torch.nn as nn

class LSTMRegressor(nn.Module):
    """
    LSTM network that we use to perform regression on financial news data
    """
    
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_layers=1):
        """
        Model initialization: initializing layers
        """
        super(LSTMRegressor, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = 0)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers)
        self.linear = nn.Linear(in_features = hidden_dim, out_features = 1)
        
        self.word_dict = None
        
    def forward(self, x):
        """
        perform a forward propagation with this LSTM network
        """
        x = x.t()
        lengths = x[0,:]
        reviews = x[1:,:]
        embeds = self.embedding(reviews)
        lstm_out, _ = self.lstm(embeds)
        output = self.linear(lstm_out)
        output = output[lengths-1, range(len(lengths))]
        
        return output.squeeze()