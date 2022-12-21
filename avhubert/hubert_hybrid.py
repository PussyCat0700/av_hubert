# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import sys,logging
import contextlib
from argparse import Namespace

import torch
import torch.nn as nn
from avhubert.hubert_asr import AVHubertSeq2SeqConfig, HubertEncoderWrapper, Embedding
from fairseq import checkpoint_utils, tasks, metrics, utils
from fairseq.utils import log_softmax, get_perplexity, item as get_item
from fairseq.criterions.fairseq_criterion import FairseqCriterion
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import BaseFairseqModel, register_model
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion, LabelSmoothedCrossEntropyCriterionConfig, label_smoothed_nll_loss
import editdistance

DBG=True if len(sys.argv) == 1 else False

if DBG:
    from hubert import AVHubertModel
    from decoder import TransformerDecoder
else:
    from .hubert import AVHubertModel
    from .decoder import TransformerDecoder

logger = logging.getLogger(__name__)

class SmoothedCELoss(nn.Module):
    """
    Adopted From fairseq's LabelSmoothedCrossEntropyCriterion
    """
    def __init__(
        self,
        padding_idx, # task.target_dictionary.pad()
        label_smoothing,
        ignore_prefix_size=0,
        ):
        super(SmoothedCELoss, self).__init__()
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.padding_idx = padding_idx
    
    def forward(self, attn_out, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(attn_out, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss
    
    def get_lprobs_and_target(self, attn_out, sample):
        lprobs = log_softmax(attn_out, dim=-1)  # (7, 42, 1000)
        target = sample['target']  #(7, 42)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

class CTCLoss(nn.CTCLoss):
    def forward(self, log_probs, concat_targets, input_lengths, target_lengths):
        """CTC Loss For AV-HuBERT Hybrid

        Args:
            log_probs (Tensor): (T, B, num_classes)

        Returns:
            Tensor: CTC Loss
        """
        return super(CTCLoss, self).forward(log_probs, concat_targets, input_lengths, target_lengths)
        
@register_criterion(
    "my_ce_loss", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class MyCELoss(LabelSmoothedCrossEntropyCriterion):
    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = log_softmax(net_output, dim=-1)  # (7, 42, 1000)
        target = sample['target']  #(7, 42)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

@register_criterion(
    "hybrid_attn_ctc_loss", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class HybridAttentionCTCCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        label_smoothing,
        ignore_prefix_size=0,
        blank=0):
        super().__init__(task)
        self.eos = task.target_dictionary.eos()
        self.pad = task.target_dictionary.pad()
        self.blank = blank
        self.tgt_dict = task.target_dictionary
        self.ctc_loss = CTCLoss(reduction='mean', blank=self.blank, zero_infinity=True)
        self.ce_loss = SmoothedCELoss(self.pad, label_smoothing, ignore_prefix_size)
        
    def forward(self, model, sample, reduce=True):
        ctc_out, attn_out = model(**sample['net_input'])  # (B, num_classes, T), (B, T, num_classes)
        
        # attention loss
        attn_loss, nll_loss = self.ce_loss(attn_out, sample, reduce)
        
        # ctc loss
        ctc_out = ctc_out.transpose(1, 2).transpose(0, 1)  # (T, B, num_classes)
        ctc_out = log_softmax(ctc_out, dim=-1)
        ctc_out_lengths = ctc_out.new(ctc_out.size(1)).fill_(ctc_out.size(0)).int()  # (b, sLen)
        target = sample['target']  # (b, tLen)
        sample_size = target.size(0)
        target_lengths = sample['target_lengths']  # (tLen)
        # input, target, input_lengths, target_lengths
        ctc_loss = self.ctc_loss(ctc_out, target, ctc_out_lengths, target_lengths)
        
        # checkpoint save metric computation
        def get_pred(e):
            toks = e.argmax(dim=-1).unique_consecutive()
            return toks[toks != self.blank]
        def post_process_to_fit_spm(sentence):
            return sentence.replace(" ", "").replace("\u2581", " ").strip()
        hypos = [get_pred(x).int().cpu() for x in ctc_out.transpose(0, 1)]
        w_edits = 0
        w_counts = 0
        for batch_id in range(len(sample["id"].tolist())):
            if "target_label" in sample:
                toks = sample["target_label"]
            else:
                toks = sample["target"]
            toks = toks[batch_id, :]
            hypo = hypos[batch_id]
            # Processes hypothesis.
            hyp_pieces = self.tgt_dict.string(hypo)
            hyp_words = post_process_to_fit_spm(hyp_pieces)

            # Processes target.
            target_tokens = utils.strip_pad(toks, self.tgt_dict.pad())
            tgt_pieces = self.tgt_dict.string(target_tokens.int().cpu())
            tgt_words = post_process_to_fit_spm(tgt_pieces)
            hyp_words, tgt_words = hyp_words.split(), tgt_words.split()
            errs, length = editdistance.eval(hyp_words, tgt_words), len(tgt_words)
            w_edits += errs
            w_counts += length
        
        # logger output dict
        tot_loss = 0.8*attn_loss + 0.2*ctc_loss
        logging_outputs = {
            "ctc_loss": ctc_loss,
            "attn_loss": attn_loss,
            "nll_loss": nll_loss,
            "tot_loss": tot_loss,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "w_edits": w_edits,
            "w_counts": w_counts,
        }
        return tot_loss, sample_size, logging_outputs
    
    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("tot_loss", 0) for log in logging_outputs)
        ctc_loss_sum = sum(log.get("ctc_loss", 0) for log in logging_outputs)
        attn_loss_sum = sum(log.get("attn_loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        w_edits = get_item(sum(log.get("w_edits", 0) for log in logging_outputs))
        w_counts = get_item(sum(log.get("w_counts", 0) for log in logging_outputs))

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "ctc_loss", ctc_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "attn_loss", attn_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: get_perplexity(meters["nll_loss"].avg)
        )
        metrics.log_scalar(
            "w_counts", w_counts
        )
        metrics.log_scalar(
            "w_edits", w_edits
        )
        metrics.log_derived(
            "wer",
            lambda meters: round(
                meters["w_edits"].sum * 100.0 / meters["w_counts"].sum, 3
            )
        )
    
    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
        

@register_model("av_hubert_hybrid", dataclass=AVHubertSeq2SeqConfig)
class AVHubertHybrid(BaseFairseqModel):
    def __init__(
        self,
        cfg,
        task
    ) -> None:
        super().__init__()
        
        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
        }
        # argument preparation
        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(
                cfg.w2v_path, arg_overrides
            )
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(
                    w2v_args
                )

        assert cfg.normalize == w2v_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for "
            "both pre-training and here"
        )
        
        # task loading
        w2v_args.task.data = cfg.data
        task_pretrain = tasks.setup_task(w2v_args.task)  # pretraining task from hubert_pretraining.py
        if state is not None:
            task_pretrain.load_state_dict(state['task_state'])
        
        # encoder will be shared for both attention and ctc.
        encoder_ = task_pretrain.build_model(w2v_args.model)  # wav2vec model pretrained from hubert_pretraining.py
        self.encoder = HubertEncoderWrapper(encoder_)
        if state is not None and not cfg.no_pretrained_weights:
            # set strict=False because we omit some modules
            del state['model']['mask_emb']
            self.encoder.w2v_model.load_state_dict(state["model"], strict=False)
        self.encoder.w2v_model.remove_pretraining_modules()
        
        # attention decoder part
        tgt_dict = task.target_dictionary
        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx=padding_idx)
            return emb
        decoder_embed_tokens = build_embedding(tgt_dict, cfg.decoder_embed_dim)
        self.decoder = TransformerDecoder(cfg, tgt_dict, decoder_embed_tokens)
        
        # ctc decoder part
        self.ctc_output_conv = outputConv("LN", cfg.decoder_embed_dim, len(tgt_dict))
        
        # cfg setting
        self.cfg = cfg
        self.freeze_finetune_updates = cfg.freeze_finetune_updates

    @classmethod
    def build_model(cls, cfg, pretrain_task):
        """Build a new model instance by just invoking __init__().
            task: pretrain task 
            cfg: config loaded by fairseq
        """
        return AVHubertHybrid(cfg=cfg, task=pretrain_task)


    def forward(self, **kwargs):  # kwargs:['source', 'padding_mask', 'prev_output_tokens']
        ft = self.freeze_finetune_updates <= self.num_updates
        with torch.no_grad() if not ft else contextlib.ExitStack():
            output = self.encoder(**kwargs)  # (T, B, C), (B, T), (B, T)
            encoder_out = output['encoder_out']
            
        # ctc decoder
        ctc_out = encoder_out.transpose(0, 1).transpose(1, 2)  # (T, B, C)->(B, C, T)
        ctc_out = self.ctc_output_conv(ctc_out)
        
        # attention decoder
        #[0]=(B, outL, V), [1]={attn, inner_states=B*[outL, B, EED]}
        attn_out = self.decoder(prev_output_tokens=kwargs['prev_output_tokens'], encoder_out=output)
        attn_out = attn_out[0]
        
        return ctc_out, attn_out  # (B, num_classes, T), (B, T, num_classes)

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

"""
convolution used before ctc

cloned from "Leveraging Self-supervised Learning for AVSR"
author: Xichen Pan
"""
class TransposeLayer(nn.Module):
    def __init__(self, dim1, dim2):
        super(TransposeLayer, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, inputBatch):
        return inputBatch.transpose(self.dim1, self.dim2)

class outputConv(nn.Module):
    def __init__(self, MaskedNormLayer, dModel, numClasses):
        super(outputConv, self).__init__()
        if MaskedNormLayer == "LN":
            self.outputconv = nn.Sequential(
                nn.Conv1d(dModel, dModel, kernel_size=(1,), stride=(1,), padding=(0,)),
                TransposeLayer(1, 2),
                nn.LayerNorm(dModel),
                TransposeLayer(1, 2),
                nn.ReLU(True),
                nn.Conv1d(dModel, dModel // 2, kernel_size=(1,), stride=(1,), padding=(0,)),
                TransposeLayer(1, 2),
                nn.LayerNorm(dModel // 2),
                TransposeLayer(1, 2),
                nn.ReLU(True),
                nn.Conv1d(dModel // 2, dModel // 2, kernel_size=(1,), stride=(1,), padding=(0,)),
                TransposeLayer(1, 2),
                nn.LayerNorm(dModel // 2),
                TransposeLayer(1, 2),
                nn.ReLU(True),
                nn.Conv1d(dModel // 2, numClasses, kernel_size=(1,), stride=(1,), padding=(0,))
            )
        else:
            self.outputconv = nn.Sequential(
                nn.Conv1d(dModel, dModel, kernel_size=(1,), stride=(1,), padding=(0,)),
                TransposeLayer(1, 2),
                MaskedNormLayer,
                TransposeLayer(1, 2),
                nn.ReLU(True),
                nn.Conv1d(dModel, dModel // 2, kernel_size=(1,), stride=(1,), padding=(0,)),
                TransposeLayer(1, 2),
                MaskedNormLayer,
                TransposeLayer(1, 2),
                nn.ReLU(True),
                nn.Conv1d(dModel // 2, dModel // 2, kernel_size=(1,), stride=(1,), padding=(0,)),
                TransposeLayer(1, 2),
                MaskedNormLayer,
                TransposeLayer(1, 2),
                nn.ReLU(True),
                nn.Conv1d(dModel // 2, numClasses, kernel_size=(1,), stride=(1,), padding=(0,))
            )

    def forward(self, inputBatch):
        return self.outputconv(inputBatch)