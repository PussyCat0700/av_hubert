# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
from itertools import groupby
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import (
    FairseqIncrementalDecoder,
)
# from fairseq.models.wav2vec.wav2vec2 import MASKING_DISTRIBUTION_CHOICES
from fairseq.modules import (
    LayerNorm,
    PositionalEmbedding,
)
from fairseq.modules.transformer_layer import TransformerDecoderLayer
import pdb


class TransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self,
        cfg,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
    ):
        super().__init__(dictionary)

        self.dropout = cfg.decoder_dropout
        self.share_input_output_embed = cfg.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = cfg.decoder_embed_dim
        self.output_embed_dim = cfg.decoder_embed_dim

        self.layerdrop = cfg.decoder_layerdrop

        padding_idx = embed_tokens.padding_idx
        self.max_target_positions = cfg.max_target_positions

        self.embed_tokens = embed_tokens
        # self.embed_scale = math.sqrt(embed_dim)  # todo: try with input_embed_dim
        self.embed_scale = 1.0 if cfg.no_scale_embedding else math.sqrt(embed_dim)

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )

        self.embed_positions = (
            PositionalEmbedding(
                cfg.max_target_positions,
                embed_dim,
                padding_idx,
                learned=cfg.decoder_learned_pos,
            )
            if not cfg.no_token_positional_embeddings
            else None
        )

        # TODO: update this when transformer gets converted to dataclass configs
        transformer_cfg = copy.deepcopy(cfg)
        # with open_dict(transformer_cfg):
        transformer_cfg.dropout = transformer_cfg.decoder_dropout
        transformer_cfg.attention_dropout = (
            transformer_cfg.decoder_attention_dropout
        )
        transformer_cfg.activation_dropout = (
            transformer_cfg.decoder_activation_dropout
        )

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                TransformerDecoderLayer(transformer_cfg, no_encoder_attn)
                for _ in range(transformer_cfg.decoder_layers)
            ]
        )

        if not self.share_input_output_embed:
            self.embed_out = nn.Parameter(
                torch.Tensor(len(dictionary), self.output_embed_dim)
            )
            nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim ** -0.5)

        if transformer_cfg.decoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def forward(
        self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        prev_output_tokens = prev_output_tokens.long()
        x, extra = self.extract_features(
            prev_output_tokens, encoder_out, incremental_state
        )
        x = self.output_layer(x) # B,T,EED->B,T,V
        return x, extra

    def extract_features(
        self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused
    ):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions.
        # prev_output_tokens=(B, T=42)
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        # (B, T=42, EED=768)
        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]

        # decoder layers
        for layer in self.layers:
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, attn, _ = layer(
                    x,  # (B, T=42)
                    encoder_out["encoder_out"] if encoder_out is not None else None,  #(T'=138, B=7, EED=768)
                    encoder_out["padding_mask"] if encoder_out is not None else None,  #(B=7, T'=138)
                    incremental_state,
                    self_attn_mask=self.buffered_future_mask(x)
                    if incremental_state is None
                    else None,
                )
                inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, {"attn": attn, "inner_states": inner_states}

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        # project back to size of vocabulary
        emb_mat = self.embed_tokens.weight if self.share_input_output_embed else self.embed_out
        return torch.matmul(features, emb_mat.transpose(0, 1))
        # if self.share_input_output_embed:
        #     return F.linear(features, self.embed_tokens.weight)
        # else:
        #     return F.linear(features, self.embed_out)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
            not hasattr(self, "_future_mask")
            or self._future_mask is None
            or self._future_mask.device != tensor.device
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(tensor.new(dim, dim)), 1
            )
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict


"""
    Adopted from "Leveraging Self-supervised Learning for AVSR"
    author: Xichen Pan
"""
def ctc_greedy_decode(outputBatch, inputLenBatch, eosIx, blank=0):
    """
    Greedy search technique for CTC decoding.
    This decoding method selects the most probable character at each time step. This is followed by the usual CTC decoding
    to get the predicted transcription.
    Note: The probability assigned to <EOS> token is added to the probability of the blank token before decoding
    to avoid <EOS> predictions in middle of transcriptions. Once decoded, <EOS> token is appended at last to the
    predictions for uniformity with targets.
    """

    outputBatch[:, :, blank] = torch.log(torch.exp(outputBatch[:, :, blank]) + torch.exp(outputBatch[:, :, eosIx]))
    reqIxs = np.arange(outputBatch.shape[2])
    reqIxs = reqIxs[reqIxs != eosIx]  # (V-1)
    outputBatch = outputBatch[:, :, reqIxs]  # (T, B, V-1), deleting eos = 2 from [0, 999]
    
    outputBatch = outputBatch.cpu()
    inputLenBatch = inputLenBatch.cpu()
    
    predCharIxs = torch.argmax(outputBatch, dim=2).T.numpy()  # (B, T)
    inpLens = inputLenBatch.numpy()  # (B)
    preds = list()
    predLens = list()
    for i in range(len(predCharIxs)):
        pred = predCharIxs[i]
        ilen = inpLens[i]
        pred = pred[:ilen]
        pred = np.array([x[0] for x in groupby(pred)])  # get rid of repetitive items in CTC
        pred = pred[pred != blank]  # get rid of blank symbols in CTC
        pred = list(pred)
        pred.append(eosIx)
        preds.extend(pred)
        predLens.append(len(pred))
    predictionBatch = torch.tensor(preds).int()
    predictionLenBatch = torch.tensor(predLens).int()
    return predictionBatch, predictionLenBatch

"""
    Adopted from "Leveraging Self-supervised Learning for AVSR"
    author: Xichen Pan
"""
def compute_CTC_prob(h, alpha, CTCOutLogProbs, T, gamma_n, gamma_b, numBeam, numClasses, blank, eosIx):
    batch = h.shape[0]  # numClasses = V - 1
    g = h[:, :, :, :-1]  # torch.Size([6, 50, 999, l-1]), 0 being eos
    c = h[:, :, :, -1]  # torch.Size([6, 50, 999]), 0 being eos
    alphaCTC = torch.zeros_like(alpha)  # torch.Size([6, 50, 999]), 0 being eos
    eosIxMask = c == eosIx  # torch.Size([6, 50, 999])
    eosIxIndex = eosIxMask.nonzero()  # torch.Size(300, 3]), last 3 dims make up a coordinate to locate eosTokens in c.3 dims:(b, beamWidth, V-1)
    eosIxIndex = torch.cat((eosIxIndex[:, :1], torch.repeat_interleave((T - 1).unsqueeze(-1), numBeam, dim=0), eosIxIndex[:, 1:]), dim=-1).long()  # torch.Size(300, 4]), represents four dims:(b, targetLength, beamWidth, V-1)
    eosIxIndex[:, -1] = 0  # (eosIx is forced to be set from eos() to eos=0 for every batch/beam in gamma.)
    gamma_eosIxMask = torch.zeros_like(gamma_n).bool()  # torch.Size([6, Tmax, 50, 1000])
    gamma_eosIxMask.index_put_(tuple(map(torch.stack, zip(*eosIxIndex))), torch.tensor(True))  # gamma_eosIxMask[eosIxIndex[:]]=True. i.e., eosIxIndex[0]=[0, 31, 0, 0], then gamma_eosIxMask[0, 31, 0, 0]=True.
    alphaCTC[eosIxMask] = np.logaddexp(gamma_n[gamma_eosIxMask], gamma_b[gamma_eosIxMask])  # logpctc

    if g.shape[-1] == 1:
        gamma_n[:, 1, 0, 1:-1] = CTCOutLogProbs[:, 1, 1:-1]  # b, tao, blank, ignoring eos(CTC=-1, gamma=0) and blank(CTC=0, gamma=-1)
    else:
        gamma_n[:, 1, :numBeam, 1:-1] = -np.inf
    gamma_b[:, 1, :numBeam, 1:-1] = -np.inf

    psi = gamma_n[:, 1, :numBeam, 1:-1]  # (6, 50, 998)
    for t in range(2, T.max()):
        activeBatch = t < T
        gEndWithc = (g[:, :, :, -1] == c)[:, :, :-1].nonzero()  # [300, 3]
        if numBeam>1:
            pdb.set_trace()
        added_gamma_n = torch.repeat_interleave(gamma_n[:, t - 1, :numBeam, None, 0], numClasses - 1, dim=-1)  # torch.Size([6, beamWidth, 998])
        if len(gEndWithc):
            added_gamma_n.index_put_(tuple(map(torch.stack, zip(*gEndWithc))), torch.tensor(-np.inf).float())  # endwithc的元素被置零，只留下不endwithc的
        phi = np.logaddexp(torch.repeat_interleave(gamma_b[:, t - 1, :numBeam, None, 0], numClasses - 1, dim=-1), added_gamma_n)  # gamma_b无论如何都是要加的
        expandShape = [batch, numBeam, numClasses - 1]
        gamma_n[:, t, :numBeam, 1:-1][activeBatch] = np.logaddexp(gamma_n[:, t - 1, :numBeam, 1:-1][activeBatch], phi[activeBatch]) \
                                                     + CTCOutLogProbs[:, t, None, 1:-1].expand(expandShape)[activeBatch]
        gamma_b[:, t, :numBeam, 1:-1][activeBatch] = \
            np.logaddexp(gamma_b[:, t - 1, :numBeam, 1:-1][activeBatch], gamma_n[:, t - 1, :numBeam, 1:-1][activeBatch]) \
            + CTCOutLogProbs[:, t, None, None, blank].expand(expandShape)[activeBatch]
        psi[activeBatch] = np.logaddexp(psi[activeBatch], phi[activeBatch] + CTCOutLogProbs[:, t, None, 1:-1].expand(phi.shape)[activeBatch])
    return torch.cat((psi, alphaCTC[:, :, -1:]), dim=-1)  # (998[no blank], 1[eos])