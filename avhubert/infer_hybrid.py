"""
cloned from "Leveraging Self-supervised Learning for AVSR"
author: Xichen Pan
"""
import argparse
import copy
import hashlib
import json
import logging
import os
import pdb
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
import sys
sys.path.append("/home/yfliu/av_hubert/fairseq")
from viterbi import ViterbiDecoder
from fairseq.data import data_utils
import editdistance
from decoder import compute_CTC_prob

from fairseq import tasks, utils, checkpoint_utils
from fairseq.logging import progress_bar

def inference_hybrid(model, inputBatch, Lambda, beamWidth, eosIx, blank=0, device="cuda"):
    """
    Args:
        blank (int): blank symbol must be set to zero. Defaults to 0.
    """
    encoderOutput = model.encoder(**inputBatch['net_input'])
    # (L,B,dmodel)
    CTCOutputConv = model.ctc_output_conv
    attentionDecoder = model.decoder

    CTCOutputBatch = encoderOutput['encoder_out'].transpose(0, 1).transpose(1, 2)  # (L,B,dmodel)->(B,dmodel,L)
    CTCOutputBatch = CTCOutputConv(CTCOutputBatch)  # (B,V,L)
    CTCOutputBatch = CTCOutputBatch.transpose(1, 2)  # (B,L,V)
    # claim batch and time step
    batch = CTCOutputBatch.shape[0]  # B
    T = torch.tensor(batch*[CTCOutputBatch.shape[1]]).cpu()  # (T)
    numClasses = CTCOutputBatch.shape[-1]
    # claim CTClogprobs and Length
    CTCOutputBatch = CTCOutputBatch.cpu()
    CTCOutLogProbs = torch.nn.functional.log_softmax(CTCOutputBatch, dim=-1) # (B,L,V)
    # init Omega and Omegahat for attention beam search
    Omega = [[[(torch.tensor([eosIx]), torch.tensor(0), torch.tensor(0))]] for i in
             range(batch)]  # B*[[(tensor([eos]), 0, 0)]]
    Omegahat = [[] for i in range(batch)]  # B*[]
    # init
    gamma_n = torch.full((batch, T.max(), beamWidth, numClasses), -np.inf).float()  # (B, Tmax, bw, V)
    gamma_b = torch.full((batch, T.max(), beamWidth, numClasses), -np.inf).float()  # (B, Tmax, bw, V)
    for b in range(batch):
        gamma_b[b, 0, 0, 0] = 0  # gamma_b[0]=1, and gamma_n[0] has been set to 0.
        for t in range(1, T[b]):
            gamma_n[b, t, 0, 0] = -np.inf  # gamma_n[t]=0
            gamma_b[b, t, 0, 0] = 0  # gamma_b[t]=1
            for tao in range(1, t + 1):
                gamma_b[b, t, 0, 0] += gamma_b[b, tao - 1, 0, 0] + CTCOutLogProbs[
                    b, tao, blank]  # gamma_b[t] *= gamma_b[tao-1]*p[tao, blank]

    newhypo = torch.arange(1, numClasses).unsqueeze(-1).unsqueeze(0).unsqueeze(0)  # (1, 1, V-1, 1)

    for l in tqdm(range(1, T.max() + 1), leave=False, desc="Regression", ncols=75):
        predictionBatch = []
        for i in range(batch):
            predictionBatch += [x[0] for x in Omega[i][-1][:beamWidth]]  # list +=
            Omega[i].append([])
        predictionBatch = torch.stack(predictionBatch).long().to(device)  # (B, T')
        if not predictionBatch.shape[0] == encoderOutput['encoder_out'].shape[1]:  # B
            encoderIndex = [i for i in range(batch) for _ in range(beamWidth)]  # beam=3, [0,0,0,1,1,1,...,batch-1,batch-1,batch-1]
            encoderOutput['encoder_out'] = encoderOutput['encoder_out'][:, encoderIndex, :]
            encoderOutput['padding_mask'] = encoderOutput['padding_mask'][encoderIndex, :]
            encoderOutput['encoder_padding_mask'] = encoderOutput['encoder_padding_mask'][encoderIndex, :]
        # predictionBatch=(B, 1), encoderOutput=(T, B, EED)
        attentionOutputBatch = attentionDecoder(predictionBatch, encoder_out=encoderOutput)
        attentionOutputBatch = attentionOutputBatch[0]  # (B, L, V)
        attentionOutputBatch = torch.nn.functional.log_softmax(attentionOutputBatch[:, -1, 1:], dim=-1)  # (B, V-1), ignoring blank
        attentionOutLogProbs = attentionOutputBatch.unsqueeze(1).cpu()  # (B, 1, V-1)

        # Decode
        h = []
        alpha = []
        for b in range(batch):
            h.append([])
            alpha.append([])
            for o in Omega[b][l - 1][:beamWidth]:
                h[b].append([o[0].tolist()])  # [[[l]]]
                alpha[b].append([[o[1], o[2]]])  #[[[tensor(-14.9989) b, tensor(-11.4477) nb]]]
        h = torch.tensor(h)  # (B, beamWidth, 1, l)
        alpha = torch.tensor(alpha).float()  # (B, beamWidth, 1, 2)
        numBeam = alpha.shape[1]
        recurrnewhypo = torch.repeat_interleave(torch.repeat_interleave(newhypo, batch, dim=0), numBeam, dim=1)  # [B, beamWidth, V-1, 1]
        h = torch.cat((torch.repeat_interleave(h, numClasses - 1, dim=2), recurrnewhypo), dim=-1)  # [B, beamWidth, V-1, l+1]
        alpha = torch.repeat_interleave(alpha, numClasses - 1, dim=2)  # [B, beamWidth, V-1, 2]
        alpha[:, :, :, 1] += attentionOutLogProbs.reshape(batch, numBeam, -1)  # [300, 1, 999], excluding blank=0

        # h = (batch * beam * (V-1) * hypoLength)
        # CTCOutLogProbs = (batch * sequence length * V)
        # gamma_n or gamma_b = (batch * max time length * beamwidth * V <which is max num of candidates in one time step>)
        CTCHypoLogProbs = compute_CTC_prob(h, CTCOutLogProbs, T, gamma_n, gamma_b, numBeam,
                                           numClasses - 1, blank, eosIx)
        alpha[:, :, :, 0] = Lambda * CTCHypoLogProbs + (1 - Lambda) * alpha[:, :, :, 1]
        hPaddingShape = list(h.shape)
        hPaddingShape[-2] = 1
        h = torch.cat((torch.zeros(hPaddingShape), h), dim=-2)  # (batch * beam * V * hypoLength), idx 0 of V is padded and will not be used.
        alpha = torch.cat((torch.full((batch, numBeam, 1, 2), -np.inf), alpha), dim=-2)  # (batch * beam * V * 2)
        
        activeBatch = (l < T).nonzero().squeeze(-1).tolist()
        for b in activeBatch:
            for i in range(numBeam):
                Omegahat[b].append((h[b, i, eosIx], copy.deepcopy(alpha[b, i, eosIx, 0])))  # c=eos
        
        alpha[:, :, eosIx, 0] = -np.inf
        predictionRes = alpha[:, :, :, 0].reshape(batch, -1).topk(beamWidth, -1).indices  # (batch * beamWidth) from V*beam, length regradless.
        # gamma=(B,T,beamWidth,V)
        for b in range(batch):
            for pos, c in enumerate(predictionRes[b]):
                beam = c // numClasses   # floordiv V
                c = c % numClasses  # mod v
                Omega[b][l].append((h[b, beam, c], copy.deepcopy(alpha[b, beam, c, 0]), copy.deepcopy(alpha[b, beam, c, 1])))
                gamma_n[b, :, pos, 0] = gamma_n[b, :, beam, c]
                gamma_b[b, :, pos, 0] = gamma_b[b, :, beam, c]
        gamma_n[:, :, :, 1:] = -np.inf
        gamma_b[:, :, :, 1:] = -np.inf

    predictionBatch = [sorted(Omegahat[b], key=lambda x: x[1], reverse=True)[0][0] for b in range(batch)]  #[([hypo], score)]
    return predictionBatch


def decode_fn(spm, indexBatch, symbols_to_ignore):
    return spm.decode(indexBatch, symbols_to_ignore)


def num_params(model):
    """
    Function that outputs the number of total and trainable paramters in the model.
    """
    numTotalParams = sum([params.numel() for params in model.parameters()])
    numTrainableParams = sum([params.numel() for params in model.parameters() if params.requires_grad])
    return numTotalParams, numTrainableParams


def inference(model, evalProgress, logger, task, cfg, use_cuda):
    num_sentences = 0
    result_dict = {'utt_id': [], 'ref': [], 'hypo': []}
    
    target_dictionary = task.target_dictionary
    spm = task.datasets[cfg.dataset.gen_subset].label_processors[0]
    symbols_to_ignore_spm = {target_dictionary.eos(), target_dictionary.pad()}
    decode_type = cfg.generation.decodeType
    if decode_type == 'hybrid':
        Lambda = cfg.generation.Lambda
    elif decode_type == 'greedy':
        decoder = ViterbiDecoder(target_dictionary)
    else:
        raise NotImplementedError(f"decode type {decode_type} not supported.")
    
    model.eval()
    for inputBatch in evalProgress:
        inputBatch = utils.move_to_cuda(inputBatch) if use_cuda else inputBatch
        targetBatch = inputBatch['target']
        if "net_input" not in inputBatch:
            continue
        with torch.no_grad():
            if decode_type == 'hybrid':
                predictionBatch = inference_hybrid(model, inputBatch, Lambda, cfg.generation.beamWidth,
                                    target_dictionary.eos(), 0, device="cuda" if use_cuda else "cpu")
            elif decode_type == 'greedy':
                predictionBatch = decoder.generate([model], inputBatch)
                predictionBatch = [x[0]['tokens'] for x in predictionBatch]
            predictionBatch = data_utils.collate_tokens(predictionBatch, pad_idx=target_dictionary.pad(), 
                                                        eos_idx=target_dictionary.eos(), left_pad=False)
            for i in range(len(inputBatch["id"])):
                result_dict['utt_id'].append(inputBatch['utt_id'][i])
                ref_sent = decode_fn(spm, targetBatch[i].int().cpu(), symbols_to_ignore_spm)
                result_dict['ref'].append(ref_sent)
                hypo_str = decode_fn(spm, predictionBatch[i].int().cpu(), symbols_to_ignore_spm)
                result_dict['hypo'].append(hypo_str)
                logger.info(f"\nREF:{ref_sent}\nHYP:{hypo_str}\n")
            num_sentences += inputBatch["nsentences"] if "nsentences" in inputBatch else inputBatch["id"].numel()
        logger.info("Recognized {:,} utterances".format(num_sentences))
    yaml_str = OmegaConf.to_yaml(cfg.generation)
    fid = int(hashlib.md5(yaml_str.encode("utf-8")).hexdigest(), 16)
    fid = fid % 1000000
    result_fn = f"{cfg.common_eval.results_path}/hypo-{fid}.json"
    json.dump(result_dict, open(result_fn, 'w'), indent=4)
    n_err, n_total = 0, 0
    assert len(result_dict['hypo']) == len(result_dict['ref'])
    for hypo, ref in zip(result_dict['hypo'], result_dict['ref']):
        hypo, ref = hypo.strip().split(), ref.strip().split()
        n_err += editdistance.eval(hypo, ref)
        n_total += len(ref)
    wer = 100 * n_err / n_total
    wer_fn = f"{cfg.common_eval.results_path}/wer.{fid}"
    with open(wer_fn, "a") as fo:
        fo.write(f'\n=================={time.asctime()}=================\n')
        fo.write(f"WER: {wer}\n")
        fo.write(f"err / num_ref_words = {n_err} / {n_total}\n\n")
        fo.write(f"{yaml_str}")
    logger.info(f"WER: {wer}%")
    return wer

def load_and_ensemble_args(args):
    conf = OmegaConf.load(args.yaml_path)
    conf.common.user_dir = args.user_dir
    conf.common_eval.results_path = os.path.join(args.finetuned_dir, 'decode', conf.generation.decodeType, conf.dataset.gen_subset)
    conf.common_eval.path = args.ckpt_path
    conf.override.modalities = []
    if args.modalities != 'AO':
        conf.override.modalities.append('video')
    if args.modalities != 'VO':
        conf.override.modalities.append('audio')
    conf.override.data = conf.override.label_dir = args.tsv_dir
    return conf
    
def main(args):
    utils.import_user_module(args.common)  # load task from current working dir
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([args.common_eval.path])
    model = models[0]
    cfg = OmegaConf.merge(args, cfg, args)
    del args
    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        torch.manual_seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)
    # logger config settings
    os.makedirs(cfg.common_eval.results_path, exist_ok=True)
    log_path = os.path.join(cfg.common_eval.results_path, "decode.log")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S',
                        filename=log_path, filemode='w')
    logger = logging.getLogger(__name__)
    # task setup
    use_cuda = torch.cuda.is_available()
    cfg.task.modalities = cfg.override.modalities
    task = tasks.setup_task(cfg.task)
    task.build_tokenizer(cfg.tokenizer)
    task.build_bpe(cfg.bpe)
    logger.info(f'Use CUDA={use_cuda}')
    logger.info(cfg)
    # loading the dataset should happen after the checkpoint has been loaded so we can give it the saved task config
    if cfg.override.data is not None:
        task.cfg.data = cfg.override.data
    if cfg.override.label_dir is not None:
        task.cfg.label_dir = cfg.override.label_dir
    task.load_dataset(cfg.dataset.gen_subset, task=cfg.task)
    # hardware settings
    # if cfg.common.fp16:
    #     model.half()
    if use_cuda:
        model.cuda()
    model.prepare_for_inference_(cfg)

    # Load dataset
    itr = task.get_batch_iterator(
        dataset=task.dataset(cfg.dataset.gen_subset),  # test
        max_tokens=cfg.dataset.max_tokens,  # 1000
        max_sentences=cfg.dataset.batch_size,  # None
        max_positions=utils.resolve_max_positions(
            task.max_positions(), model.max_positions()
        ),  # 9223372036854775807
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,  # False
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,  # 8
        seed=cfg.common.seed,  # 1
        num_shards= 1,
        shard_id=cfg.distributed_training.distributed_rank,  # 0
        num_workers=cfg.dataset.num_workers,  # 0
        data_buffer_size=cfg.dataset.data_buffer_size,  # 10
    ).next_epoch_itr(shuffle=False)
    logger.info(f'eval on set {cfg.dataset.gen_subset}, iteration length={len(itr)}.')
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )

    logger.info("\nTesting the trained model .... \n")

    testWER = inference(model, progress, logger, task, cfg, True)

    logger.info("Test WER: %.3f" % (testWER))

    logger.info("\nTesting Done.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-dir', help='Config directory', default="/home/yfliu/av_hubert/avhubert/conf")
    parser.add_argument('--config-name', help='Name of config file with .postfix', default="hybrid_decode.yaml")
    parser.add_argument('--tsv-dir', help="to override task.label_dir and task.data", default="/home/yfliu/datasets/lrs3/30h_data")
    parser.add_argument('--finetuned-dir', help='finetune dir where decoding result will be saved', default="/home/yfliu/output/finetune_hybrid_spm100")
    parser.add_argument('--user-dir', help='command-line pwd result')
    parser.add_argument('--modalities', help='shuold be one of "AO", "VO" or "AV".', default="VO")
    args = parser.parse_args()
    args.yaml_path = os.path.join(args.config_dir, args.config_name)
    args.ckpt_path = os.path.join(args.finetuned_dir, "checkpoints", "checkpoint_best.pt")
    # load args from yaml file using OmegaConf
    args = load_and_ensemble_args(args)
    main(args)
