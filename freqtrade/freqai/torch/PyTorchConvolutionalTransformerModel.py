import logging
import math

import torch
from torch import nn

"""
The architecture is based on the paper “Attention Is All You Need”.
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
Lukasz Kaiser, and Illia Polosukhin. 2017.
"""
logger = logging.getLogger(__name__)


class PyTorchConvolutionalTransformerModel(nn.Module):
    """
    A transformer approach to time series modeling using positional encoding.
    The architecture is based on the paper “Attention Is All You Need”.
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017.
    """

    def __init__(
            self,
            input_dim: int = 7,
            output_dim: int = 7,
            hidden_dim=1024,
            n_layer=2,
            dropout_percent=0.1,
            time_window=10,
            nhead=8,
            d_model=128,
            cnn_filters=64,
            kernel_size=3,
    ):
        super().__init__()
        self.time_window = time_window
        logger.info(f'Setting up model with input_dim {input_dim}, outputdim: {output_dim}, '
                    f'hiddendim: {hidden_dim}, n_layer: {n_layer}, droupout: {dropout_percent}, '
                    f'time_window: {time_window}, nhead: {nhead}, d_model: {d_model}, cnn_filters: {cnn_filters},'
                    f'kernel_size: {kernel_size}')
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=cnn_filters,
                      kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU(),
            nn.Conv1d(cnn_filters, d_model, kernel_size=kernel_size,
                      padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
        )

        # Encode the timeseries with Positional encoding
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=d_model)

        # Define the encoder block of the Transformer
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout_percent, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layer)

        # the pseudo decoding FC
        self.output_net = nn.Sequential(
            nn.Linear(d_model , int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(dropout_percent),
            nn.Linear(int(hidden_dim), output_dim),
        )
        logger.info('DONE Setting up model')

    def forward(self, x, mask=None, add_positional_encoding=True):
        """
        Args:
            x: Input features of shape [Batch, SeqLen, input_dim] aka (batch, time_steps, num_features)
            mask: Mask to apply on the attention outputs (optional)
            add_positional_encoding: If True, we add the positional encoding to the input.
                                      Might not be desired for some tasks.
        """
        # logger.info(f'Model Input columns before transposing: {x.shape}')
        # logger.info('Starting forwarding')
        x = x.transpose(1, 2)  # → (batch, features, time_steps)
        # logger.info(f'Model Input columns after transposing: {x.shape}')
        x = self.conv(x)
        # logger.info(f'X columns before second transposing: {x.shape}')
        x = x.transpose(1, 2)  # → (batch, time_steps, d_model)
        # logger.info(f'X columns after second transposing: {x.shape}')

        if add_positional_encoding:
            x = self.positional_encoding(x)
        # logger.info(f'X columns after positional encoding: {x.shape}')
        # x = x.permute(1, 0, 2)
        # logger.info(f'X columns after permuting: {x.shape}')
        x = self.transformer(x, mask=mask)
        # logger.info(f'X columns after transformer: {x.shape}')
        # x = x.reshape(-1, 1, self.time_window * x.shape[-1])  # from the original model
        x = x[:, -1, :]  # or x.mean(dim=1)
        # logger.info(f'X columns after reshaping: {x.shape}')
        # x = x[-1] from the proposition to take only recent time????
        x = self.output_net(x)
        # logger.info(f'X columns after output net: {x.shape}')
        res = x.unsqueeze(1)
        # logger.info(f'Finishing forwarding. Shape of x before squeeze: {x.shape}, after squeeze: {res.shape}')
        return res


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Args
            d_model: Hidden dimensionality of the input.
            max_len: Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding
        # for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x
