import os
import math
import time
from tqdm import tqdm

import torch
import torch.nn as nn


########
# Taken from:
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# or also here:
# https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # shape (max_len, 1, dim)
        self.register_buffer('pe', pe)  # Will not be trained.

    def forward(self, x):
        """Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        assert x.size(0) < self.max_len, (
            f"Too long sequence length: increase `max_len` of pos encoding")
        # shape of x (len, B, dim)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, source_vocabulary_size, target_vocabulary_size,
                 d_model=256, pad_id=0, encoder_layers=3, decoder_layers=2,
                 dim_feedforward=1024, num_heads=8):
        # all arguments are (int)
        super().__init__()
        self.pad_id = pad_id

        self.embedding_src = nn.Embedding(source_vocabulary_size, d_model, padding_idx=pad_id)
        self.embedding_tgt = nn.Embedding(target_vocabulary_size, d_model, padding_idx=pad_id)

        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model, num_heads, encoder_layers, decoder_layers, dim_feedforward)
        self.encoder = self.transformer.encoder
        self.decoder = self.transformer.decoder
        self.linear = nn.Linear(d_model, target_vocabulary_size)

    def create_src_padding_mask(self, src):
        # input src of shape ()
        src_padding_mask = src.transpose(0, 1) == 0
        return src_padding_mask

    def create_tgt_padding_mask(self, tgt):
        # input tgt of shape ()
        tgt_padding_mask = tgt.transpose(0, 1) == 0
        return tgt_padding_mask


    def greedy_decode(self, target, max_len, memory, memory_key_padding_mask):
        """
        :param target: decoder target
        :param max_len:  maximum length for the algorithm to run
        :param memory: encoder output
        :param memory_key_padding_mask: encoder output mask
        :return:
        """
        ys = torch.ones(1, 1).fill_(3).type_as(target.data).to(DEVICE)
        for i in range(max_len - 1):
            tgt_key_padding_mask = self.create_tgt_padding_mask(ys).to(DEVICE)
            tgt_mask = (nn.Transformer.generate_square_subsequent_mask(ys.size(0))
                        .type(torch.bool)).to(DEVICE)
            tgt = self.embedding_tgt(ys)
            tgt = self.pos_encoder(tgt)

            out = self.decoder(tgt, memory, tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask)
            # shift the target by one
            out = out.transpose(0, 1)
            prob = self.linear(out[:, -1])

            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

            ys = torch.cat([ys, torch.ones(1, 1).type_as(target.data).fill_(next_word)], dim=0)

            # stop decoding when we get an end of sentence
            if next_word == 2:
                break

        if ys.shape[0] < max_len:
            new_seq = ys.data.new(max_len, 1).fill_(0)
            new_seq[:ys.shape[0], :] = ys
            ys = new_seq
        return ys

    def greedy_search(self, src, tgt):
        """
        Greedy search algorithm
        :param src: input sequence
        :param tgt: target sequence
        :return: decoded output
        """
        src_key_padding_mask = self.create_src_padding_mask(src).to(DEVICE)
        out = self.embedding_src(src)
        out = self.pos_encoder(out)
        encoder_out = self.encoder(out, src_key_padding_mask=src_key_padding_mask)
        results = torch.ones(tgt.shape[1], tgt.shape[0]).type(torch.long).to(DEVICE)
        for i in range(encoder_out.shape[1]):
            memory = encoder_out[:, i, :].unsqueeze(dim=1)
            memory_key_padding_mask = src_key_padding_mask[i, :].unsqueeze(dim=0)
            result = self.greedy_decode(tgt[:, i], tgt.size()[0] + 1, memory, memory_key_padding_mask)
            result = result.permute(1, 0)
            results[i, :] = result[:, 1:]
        return results


    def forward_separate(self, src, tgt):
        """
        Forward function that forward decoder and encoder separately
        :param src: tensor of shape (sequence_length, batch, data dim)
        :param tgt: tensor of shape (sequence_length, batch, data dim)
        :return: tensor of shape (sequence_length, batch, data dim)
        """
        src_key_padding_mask = self.create_src_padding_mask(src).to(DEVICE)
        tgt_key_padding_mask = self.create_tgt_padding_mask(tgt).to(DEVICE)
        memory_key_padding_mask = src_key_padding_mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt.shape[0]).to(DEVICE)

        tgt = self.embedding_tgt(tgt)
        tgt = self.pos_encoder(tgt)
        out = self.embedding_src(src)
        out = self.pos_encoder(out)

        encoder_out = self.encoder(out, src_key_padding_mask=src_key_padding_mask)
        decoder_out = self.decoder(tgt, encoder_out, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                                   memory_key_padding_mask=memory_key_padding_mask)

        out = self.linear(decoder_out)
        return out

    def forward(self, src, tgt):
        """
        Forword function
        :param src: tensor of shape (sequence_length, batch, data dim)
        :param tgt: tensor of shape (sequence_length, batch, data dim)
        :return: tensor of shape (sequence_length, batch, data dim)
        """
        src_key_padding_mask = self.create_src_padding_mask(src).to(DEVICE)
        tgt_key_padding_mask = self.create_tgt_padding_mask(tgt).to(DEVICE)
        memory_key_padding_mask = src_key_padding_mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt.shape[0]).to(DEVICE)

        tgt = self.embedding_tgt(tgt)
        tgt = self.pos_encoder(tgt)
        out = self.embedding_src(src)
        out = self.pos_encoder(out)
        out = self.transformer(
            out, tgt, src_key_padding_mask=src_key_padding_mask,
            tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask)
        out = self.linear(out)

        return out