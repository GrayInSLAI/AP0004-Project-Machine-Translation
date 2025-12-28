import torch
import torch.nn as nn
import math


# ==========================================
# 1. Normalization Layers
# ==========================================
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Often used in modern LLMs (e.g., LLaMA) for stability.
    """

    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Calculate RMS
        norm_x = torch.mean(x**2, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.scale * x_normed


def get_norm_layer(type_name, dim):
    """
    Factory function to select normalization type.
    """
    if type_name.lower() == "rms":
        return RMSNorm(dim)
    else:
        # Default to standard LayerNorm
        return nn.LayerNorm(dim)


# ==========================================
# 2. Positional Embeddings
# ==========================================
class PositionalEncoding(nn.Module):
    """
    Standard Sinusoidal Positional Encoding (Absolute).
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create constant 'pe' matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a learnable parameter)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Dim]
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable Absolute Positional Encoding.
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embeddings = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Dim]
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.embeddings(positions)
        return self.dropout(x)


# ==========================================
# 3. Transformer Layers (Pre-Norm)
# ==========================================
class CustomEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, norm_type="layer"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # Feed Forward Network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Normalization
        self.norm1 = get_norm_layer(norm_type, d_model)
        self.norm2 = get_norm_layer(norm_type, d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()  # Modern activation

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Pre-Norm Architecture: Norm -> Attn -> Add

        # 1. Self Attention
        src2 = self.norm1(src)
        src2, _ = self.self_attn(
            src2, src2, src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(src2)

        # 2. Feed Forward
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)

        return src


class CustomDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, norm_type="layer"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # Feed Forward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Norms
        self.norm1 = get_norm_layer(norm_type, d_model)
        self.norm2 = get_norm_layer(norm_type, d_model)
        self.norm3 = get_norm_layer(norm_type, d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        # Pre-Norm Architecture

        # 1. Self Attention (Masked)
        tgt2 = self.norm1(tgt)
        tgt2, _ = self.self_attn(
            tgt2, tgt2, tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout1(tgt2)

        # 2. Cross Attention (Source-Target)
        tgt2 = self.norm2(tgt)
        tgt2, _ = self.multihead_attn(
            tgt2,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout2(tgt2)

        # 3. Feed Forward
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt


# ==========================================
# 4. Main Transformer Model
# ==========================================
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.3,
        norm_type="layer",  # 'layer' or 'rms'
        pos_scheme="sinusoidal",  # 'sinusoidal' or 'learnable'
        max_len=512,
    ):
        super().__init__()

        self.d_model = d_model

        # Token Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(trg_vocab_size, d_model)

        # Positional Embeddings
        if pos_scheme == "learnable":
            self.pos_encoder = LearnablePositionalEncoding(d_model, max_len, dropout)
            self.pos_decoder = LearnablePositionalEncoding(d_model, max_len, dropout)
        else:
            self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
            self.pos_decoder = PositionalEncoding(d_model, max_len, dropout)

        # Encoder & Decoder Stacks
        self.encoder_layers = nn.ModuleList(
            [
                CustomEncoderLayer(d_model, nhead, dim_feedforward, dropout, norm_type)
                for _ in range(num_encoder_layers)
            ]
        )

        self.decoder_layers = nn.ModuleList(
            [
                CustomDecoderLayer(d_model, nhead, dim_feedforward, dropout, norm_type)
                for _ in range(num_decoder_layers)
            ]
        )

        # Final Output Layer
        self.final_norm = get_norm_layer(norm_type, d_model)
        self.output_layer = nn.Linear(d_model, trg_vocab_size)

        print(
            f"Model Initialized: {norm_type.upper()} Norm, {pos_scheme.upper()} Pos Enc."
        )

    def generate_square_subsequent_mask(self, sz, device):
        """
        Generate a mask to prevent attention to future tokens.
        Returns: [sz, sz] upper triangular matrix with -inf.
        """
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def encode(self, src, src_key_padding_mask=None):
        # src: [Batch, Seq_Len]

        # 1. Embedding + Scaling
        x = self.src_embedding(src) * math.sqrt(self.d_model)

        # 2. Positional Encoding
        x = self.pos_encoder(x)

        # 3. Encoder Layers
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)

        return self.final_norm(x)

    def decode(
        self,
        tgt,
        memory,
        tgt_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        # tgt: [Batch, Seq_Len]
        # memory: [Batch, Seq_Len, Dim] (Encoder Output)

        # 1. Embedding + Scaling
        x = self.trg_embedding(tgt) * math.sqrt(self.d_model)

        # 2. Positional Encoding
        x = self.pos_decoder(x)

        # 3. Decoder Layers
        for layer in self.decoder_layers:
            x = layer(
                x,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        return self.final_norm(x)

    def forward(
        self,
        src,
        tgt,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        """
        Full Seq2Seq Forward Pass.
        """
        # 1. Encode
        memory = self.encode(src, src_key_padding_mask=src_key_padding_mask)

        # 2. Create Target Mask (Causal Mask)
        tgt_seq_len = tgt.size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len, tgt.device)

        # 3. Decode
        output = self.decode(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        # 4. Project to Vocab
        logits = self.output_layer(output)
        return logits


# ==========================================
# Unit Test
# ==========================================
if __name__ == "__main__":
    print("=== Testing transformers_models.py ===")

    # Configuration
    SRC_VOCAB = 5000
    TRG_VOCAB = 5000
    BATCH_SIZE = 2
    SEQ_LEN = 10

    # 1. Instantiate Model (Test RMSNorm + Learnable Pos)
    model = Transformer(
        src_vocab_size=SRC_VOCAB,
        trg_vocab_size=TRG_VOCAB,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        norm_type="rms",  # Testing RMSNorm
        pos_scheme="learnable",  # Testing Learnable PE
    )

    # 2. Create Dummy Data
    src = torch.randint(0, SRC_VOCAB, (BATCH_SIZE, SEQ_LEN))
    tgt = torch.randint(0, TRG_VOCAB, (BATCH_SIZE, SEQ_LEN))

    # Create padding masks (1=not padded, 0=padded? PyTorch attention usually needs bool: True=Ignored)
    # PyTorch MultiheadAttention key_padding_mask: True for values to be ignored (pad tokens)
    src_padding_mask = torch.zeros((BATCH_SIZE, SEQ_LEN), dtype=torch.bool)
    tgt_padding_mask = torch.zeros((BATCH_SIZE, SEQ_LEN), dtype=torch.bool)

    # 3. Forward Pass
    print("\n[Running Forward Pass]")
    try:
        logits = model(
            src,
            tgt,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )
        print(
            f"Logits Shape: {logits.shape} (Expected: [{BATCH_SIZE}, {SEQ_LEN}, {TRG_VOCAB}])"
        )
        print("Forward pass successful.")
    except Exception as e:
        print(f"Forward pass failed: {e}")
