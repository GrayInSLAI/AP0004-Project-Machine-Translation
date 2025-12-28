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

        # Select RNN type
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True
            )
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(
                emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True
            )
        else:
            raise ValueError("rnn_type must be 'lstm' or 'gru'")

    def forward(self, src, src_len):
        """
        src: (batch_size, seq_len)
        src_len: (batch_size)
        """
        # embedded: (batch_size, seq_len, emb_dim)
        embedded = self.dropout(self.embedding(src))

        # Pack sequence
        # src_len must be on CPU for pack_padded_sequence
        packed_embedded = pack_padded_sequence(
            embedded, src_len.cpu(), batch_first=True, enforce_sorted=False
        )

        # Forward RNN
        packed_outputs, hidden = self.rnn(packed_embedded)

        # Unpack
        # outputs: (batch_size, seq_len, hid_dim)
        outputs, _ = pad_packed_sequence(
            packed_outputs, batch_first=True, total_length=src.shape[1]
        )

        return outputs, hidden


class Attention(nn.Module):
    """
    Attention Mechanism: Calculates context vector based on encoder outputs.
    Supports: dot, multiplicative (general), additive (concat/bahdanau)
    """

    def __init__(self, enc_hid_dim, dec_hid_dim, method="dot"):
        super().__init__()
        self.method = method
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        if method == "general":  # Multiplicative
            self.W = nn.Linear(enc_hid_dim, dec_hid_dim)
        elif method == "additive":  # Bahdanau
            self.W = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
            self.v = nn.Linear(dec_hid_dim, 1, bias=False)
        elif method != "dot":
            raise ValueError("Attention method must be 'dot', 'general', or 'additive'")

    def forward(self, hidden, encoder_outputs, mask=None):
        """
        hidden: (batch_size, dec_hid_dim) - Query (Decoder current state)
        encoder_outputs: (batch_size, src_len, enc_hid_dim) - Keys/Values
        mask: (batch_size, src_len) - 1 for valid, 0 for pad
        """
        src_len = encoder_outputs.shape[1]

        # Calculate Energy
        if self.method == "dot":
            # (batch, 1, dec_hid) @ (batch, enc_hid, src_len) -> (batch, 1, src_len)
            # Assumption: dec_hid_dim == enc_hid_dim for dot product
            score = torch.bmm(hidden.unsqueeze(1), encoder_outputs.permute(0, 2, 1))

        elif self.method == "general":
            # energy = hidden @ W @ encoder_outputs.T
            # W(enc_out) -> (batch, src_len, dec_hid)
            x = self.W(encoder_outputs)
            score = torch.bmm(hidden.unsqueeze(1), x.permute(0, 2, 1))

        elif self.method == "additive":
            # hidden expanded: (batch, src_len, dec_hid)
            hidden_expanded = hidden.unsqueeze(1).repeat(1, src_len, 1)
            # concat: (batch, src_len, enc_hid + dec_hid)
            combined = torch.cat((hidden_expanded, encoder_outputs), dim=2)
            # energy: (batch, src_len, 1)
            energy = torch.tanh(self.W(combined))
            score = self.v(energy).permute(0, 2, 1)  # -> (batch, 1, src_len)

        # Squeeze: (batch, src_len)
        attention = score.squeeze(1)

        # Apply Mask
        if mask is not None:
            # Mask logic: mask is 1 for valid, 0 for pad.
            # We want to fill 0 positions with -inf
            attention = attention.masked_fill(mask == 0, -1e10)

        # Softmax to get probabilities
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    """
    Decoder: Decodes the context vector into target sentence step-by-step.
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

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

        # RNN Input: Embedding + Weighted Source Context
        self.rnn_input_dim = emb_dim + enc_hid_dim

        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                self.rnn_input_dim,
                dec_hid_dim,
                n_layers,
                dropout=dropout,
                batch_first=True,
            )
        else:
            self.rnn = nn.GRU(
                self.rnn_input_dim,
                dec_hid_dim,
                n_layers,
                dropout=dropout,
                batch_first=True,
            )

        self.fc_out = nn.Linear(emb_dim + dec_hid_dim + enc_hid_dim, output_dim)

    def forward(self, input_token, hidden, encoder_outputs, mask):
        """
        Single step decoding.

        input_token: (batch_size) - 1D tensor of token indices
        hidden: Previous hidden state
        encoder_outputs: (batch_size, src_len, enc_hid_dim)
        mask: (batch_size, src_len)
        """
        # 1. Embed input: (batch, 1, emb_dim)
        input_token = input_token.unsqueeze(1)
        embedded = self.dropout(self.embedding(input_token))

        # 2. Calculate Attention Weights based on PREVIOUS hidden state
        # Note: For LSTM, hidden is (h, c), we usually use h[-1] for attention query
        if self.rnn_type == "lstm":
            # hidden[0] is h_n: (n_layers, batch, hid) -> take last layer: (batch, hid)
            query = hidden[0][-1]
        else:
            # GRU: hidden is (n_layers, batch, hid)
            query = hidden[-1]

        # a: (batch, src_len)
        a = self.attention(query, encoder_outputs, mask)

        # 3. Calculate Context Vector (Weighted Sum)
        # a: (batch, 1, src_len)
        a = a.unsqueeze(1)
        # weighted: (batch, 1, enc_hid) = (batch, 1, src_len) @ (batch, src_len, enc_hid)
        weighted = torch.bmm(a, encoder_outputs)

        # 4. RNN Step
        # Input: Concat(Embedding, Context) -> (batch, 1, emb + enc_hid)
        rnn_input = torch.cat((embedded, weighted), dim=2)

        # Output: (batch, 1, dec_hid), new_hidden
        output, new_hidden = self.rnn(rnn_input, hidden)

        # 5. Prediction
        # Concat(Embedding, RNN_Output, Context) to allow skip-connections
        embedded = embedded.squeeze(1)
        output = output.squeeze(1)
        weighted = weighted.squeeze(1)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))

        return prediction, new_hidden, a.squeeze(1)


class Seq2Seq(nn.Module):
    """
    Main Seq2Seq Model Wrapper.
    """

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert (
            encoder.hid_dim == decoder.attention.enc_hid_dim
        ), "Encoder hidden dim must match Attention enc dim"

    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):
        """
        src: (batch, src_len)
        src_len: (batch)
        trg: (batch, trg_len)
        teacher_forcing_ratio: Probability of using ground truth as input
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # 1. Encode source
        encoder_outputs, hidden = self.encoder(src, src_len)

        # Create mask for attention (1 for valid, 0 for pad)
        mask = src != 0

        # 2. Initialize First Input (SOS)
        input_token = trg[:, 0]

        # 3. Decoding Loop
        for t in range(1, trg_len):
            # Step forward
            output, hidden, _ = self.decoder(input_token, hidden, encoder_outputs, mask)

            # Store prediction
            outputs[:, t] = output

            # 4. Teacher Forcing Logic
            # Decide whether to use ground truth or model's own prediction
            teacher_force = random.random() < teacher_forcing_ratio

            # Get the highest predicted token
            top1 = output.argmax(1)

            # Next input is either Ground Truth or Prediction
            input_token = trg[:, t] if teacher_force else top1

        return outputs


# ==========================================
# Unit Test
# ==========================================
if __name__ == "__main__":
    print("=== Testing models.py ===")

    # 1. Hyperparameters
    INPUT_DIM = 1000
    OUTPUT_DIM = 1000
    ENC_EMB_DIM = 64
    DEC_EMB_DIM = 64
    HID_DIM = 128
    N_LAYERS = 2
    DROPOUT = 0.5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Initialize Components (Test GRU + Dot Attention)
    attn = Attention(HID_DIM, HID_DIM, method="dot")
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, DROPOUT, rnn_type="gru")
    dec = Decoder(
        OUTPUT_DIM,
        DEC_EMB_DIM,
        HID_DIM,
        HID_DIM,
        N_LAYERS,
        DROPOUT,
        attn,
        rnn_type="gru",
    )
    model = Seq2Seq(enc, dec, device).to(device)

    # 3. Create Dummy Data
    BATCH_SIZE = 4
    SRC_LEN = 15
    TRG_LEN = 12

    # Random tokens
    src = torch.randint(0, INPUT_DIM, (BATCH_SIZE, SRC_LEN)).to(device)
    src_len = torch.tensor([15, 12, 10, 8]).to(device)  # Variable lengths
    trg = torch.randint(0, OUTPUT_DIM, (BATCH_SIZE, TRG_LEN)).to(device)

    # 4. Forward Pass
    print("\n[Forward Pass Test]")
    outputs = model(src, src_len, trg, teacher_forcing_ratio=0.5)

    print(f"Input src shape: {src.shape}")
    print(f"Input trg shape: {trg.shape}")
    print(f"Output predictions shape: {outputs.shape}")  # Should be (4, 12, 1000)

    expected_shape = (BATCH_SIZE, TRG_LEN, OUTPUT_DIM)
    assert outputs.shape == expected_shape
    print("Shape check passed!")
