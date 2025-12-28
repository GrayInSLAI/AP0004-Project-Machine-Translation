import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random


class Encoder(nn.Module):
    """
    Encoder: Encodes the source sentence into context vectors.
    """

    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, rnn_type="gru"):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.rnn_type = rnn_type.lower()

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

        # Handle dropout warning for single layer RNN
        rnn_dropout = dropout if n_layers > 1 else 0

        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                emb_dim, hid_dim, n_layers, dropout=rnn_dropout, batch_first=True
            )
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(
                emb_dim, hid_dim, n_layers, dropout=rnn_dropout, batch_first=True
            )
        else:
            raise ValueError("rnn_type must be 'lstm' or 'gru'")

    def forward(self, src, src_len):
        """
        src: (batch_size, seq_len)
        src_len: (batch_size) - Tensor containing actual lengths
        """
        # embedded: (batch_size, seq_len, emb_dim)
        embedded = self.dropout(self.embedding(src))

        # Pack sequence for efficiency (ignore pads)
        # src_len must be on CPU for pack_padded_sequence
        packed_embedded = pack_padded_sequence(
            embedded, src_len.cpu(), batch_first=True, enforce_sorted=False
        )

        # Forward pass through RNN
        packed_outputs, hidden = self.rnn(packed_embedded)

        # Unpack
        outputs, _ = pad_packed_sequence(
            packed_outputs, batch_first=True, total_length=src.shape[1]
        )

        return outputs, hidden


class Attention(nn.Module):
    """
    Attention Mechanism: Dot, General, or Additive.
    """

    def __init__(self, enc_hid_dim, dec_hid_dim, method="dot"):
        super().__init__()
        self.method = method
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        if method == "general":
            self.W = nn.Linear(enc_hid_dim, dec_hid_dim)
        elif method == "additive":
            self.W = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
            self.v = nn.Linear(dec_hid_dim, 1, bias=False)
        elif method != "dot":
            raise ValueError("Attention method must be 'dot', 'general', or 'additive'")

    def forward(self, hidden, encoder_outputs, mask=None):
        """
        hidden: (batch_size, dec_hid_dim)
        encoder_outputs: (batch_size, src_len, enc_hid_dim)
        mask: (batch_size, src_len)
        """
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        # Calculate Energy (Scores)
        if self.method == "dot":
            # (batch, 1, dec_hid) @ (batch, enc_hid, src_len) -> (batch, 1, src_len)
            score = torch.bmm(hidden.unsqueeze(1), encoder_outputs.permute(0, 2, 1))

        elif self.method == "general":
            x = self.W(encoder_outputs)
            score = torch.bmm(hidden.unsqueeze(1), x.permute(0, 2, 1))

        elif self.method == "additive":
            hidden_expanded = hidden.unsqueeze(1).repeat(1, src_len, 1)
            combined = torch.cat((hidden_expanded, encoder_outputs), dim=2)
            energy = torch.tanh(self.W(combined))
            score = self.v(energy).permute(0, 2, 1)

        attention = score.squeeze(1)  # (batch, src_len)

        if mask is not None:
            # Mask logic: mask is 1 for valid, 0 for pad.
            # We want to fill 0 positions with -inf
            attention = attention.masked_fill(mask == 0, -1e10)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    """
    Decoder with Attention.
    """

    def __init__(
        self,
        output_dim,
        emb_dim,
        enc_hid_dim,
        dec_hid_dim,
        n_layers,
        dropout,
        attention,
        rnn_type="gru",
    ):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.rnn_type = rnn_type.lower()
        self.n_layers = n_layers

        # output_dim here is the size of the SHARED vocabulary
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

        self.rnn_input_dim = emb_dim + enc_hid_dim

        rnn_dropout = dropout if n_layers > 1 else 0

        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                self.rnn_input_dim,
                dec_hid_dim,
                n_layers,
                dropout=rnn_dropout,
                batch_first=True,
            )
        else:
            self.rnn = nn.GRU(
                self.rnn_input_dim,
                dec_hid_dim,
                n_layers,
                dropout=rnn_dropout,
                batch_first=True,
            )

        self.fc_out = nn.Linear(emb_dim + dec_hid_dim + enc_hid_dim, output_dim)

    def forward(self, input_token, hidden, encoder_outputs, mask):
        # input_token: (batch_size)
        input_token = input_token.unsqueeze(1)
        embedded = self.dropout(self.embedding(input_token))

        # Calculate Attention
        if self.rnn_type == "lstm":
            query = hidden[0][-1]
        else:
            query = hidden[-1]

        a = self.attention(query, encoder_outputs, mask)

        # Context Vector
        a = a.unsqueeze(1)
        weighted = torch.bmm(a, encoder_outputs)

        # RNN Step
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, new_hidden = self.rnn(rnn_input, hidden)

        # Prediction
        embedded = embedded.squeeze(1)
        output = output.squeeze(1)
        weighted = weighted.squeeze(1)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))

        return prediction, new_hidden, a.squeeze(1)


class Seq2Seq(nn.Module):
    """
    Seq2Seq Wrapper with Shared Embeddings Support.
    """

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        self.decoder.embedding = self.encoder.embedding

        assert (
            encoder.hid_dim == decoder.attention.enc_hid_dim
        ), "Encoder hidden dim must match Attention enc dim"

    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        encoder_outputs, hidden = self.encoder(src, src_len)

        # Create mask (1 for valid, 0 for pad)
        mask = src != 0

        input_token = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, _ = self.decoder(input_token, hidden, encoder_outputs, mask)
            outputs[:, t] = output

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = trg[:, t] if teacher_force else top1

        return outputs
