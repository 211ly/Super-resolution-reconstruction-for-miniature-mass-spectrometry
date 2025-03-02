import torch
import torch.nn as nn

class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_len, split_size):
        super(Seq2SeqLSTM, self).__init__()
        self.split_size = split_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True, num_layers=2)
        self.fc = nn.Linear(hidden_size*2, output_size)

    def forward(self, x):

        batch_size = x.size(0)
        # Reshape input from (batch, channels, length) to (batch, seq_len, split_size)
        x = x.view(batch_size, self.seq_len, self.split_size)

        # Pass through LSTM
        lstm_out, _ = self.lstm(x)

        # Pass through fully connected layer
        output = self.fc(lstm_out)
        output = output.view(batch_size, -1, self.seq_len * self.split_size)

        return output
class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=2, bidirectional=True, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=bidirectional)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        outputs, (hidden, cell) = self.lstm(x, (hidden, cell))
        res = self.fc(outputs)
        return res


class EncDecLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, seq_len, split_size, bidirectional=True):
        self.split_size = split_size
        self.output_size = output_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.enc = Encoder(input_size, hidden_size, bidirectional)
        self.dec = Decoder(output_size, hidden_size)

    def forward(self, src, trg):
        batch_size = src.size(0)
        src = src.view(batch_size, self.seq_len, self.split_size)
        trg = trg.view(batch_size, self.seq_len, self.split_size)
        outputs = torch.zeros(batch_size, self.seq_len, self.output_size)

        enc_output, hidden, cell = self.enc(src)

        



