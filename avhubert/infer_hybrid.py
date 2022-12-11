"""
cloned from "Leveraging Self-supervised Learning for AVSR"
author: Xichen Pan
"""
import argparse
import logging
import os
from itertools import chain
import pdb

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from error_rate import compute_error_ch, compute_error_word
from decoder import ctc_greedy_decode, compute_CTC_prob

from fairseq import tasks, utils, checkpoint_utils
from fairseq.logging import progress_bar

def inference_hybrid(model, inputBatch, Lambda, beamWidth, eosIx, blank, device="cuda"):
    inputLenBatch = inputBatch['target_lengths']
    encoderOutput = model.encoder(**inputBatch['net_input'])
    # (L,B,dmodel)
    CTCOutputConv = model.ctc_output_conv
    attentionDecoder = model.decoder

    CTCOutputBatch = encoderOutput['encoder_out'].transpose(0, 1).transpose(1, 2)  # (L,B,dmodel)->(B,dmodel,L)
    CTCOutputBatch = CTCOutputConv(CTCOutputBatch)  # (B,num_classes,L)
    CTCOutputBatch = CTCOutputBatch.transpose(1, 2)  # (B,L,num_classes)
    # claim batch and time step
    batch = CTCOutputBatch.shape[0]  # B
    T = inputLenBatch.cpu()  # (T)
    numClasses = CTCOutputBatch.shape[-1]
    # claim CTClogprobs and Length
    CTCOutputBatch = CTCOutputBatch.cpu()
    CTCOutLogProbs = torch.nn.functional.log_softmax(CTCOutputBatch, dim=-1) # (B,L,num_classes)
    predictionLenBatch = torch.ones(batch, device=device).long()
    # init Omega and Omegahat for attention beam search
    Omega = [[[(torch.tensor([eosIx]), torch.tensor(0), torch.tensor(0))]] for i in
             range(batch)]  # B*[[(tensor([eos]), 0, 0)]]
    Omegahat = [[] for i in range(batch)]  # B*[]
    # init
    gamma_n = torch.full((batch, T.max(), beamWidth, numClasses), -np.inf).float()  # (B, Tmax, bw, num_classes)
    gamma_b = torch.full((batch, T.max(), beamWidth, numClasses), -np.inf).float()  # (B, Tmax, bw, num_classes)
    for b in range(batch):
        gamma_b[b, 0, 0, 0] = 0  # gamma_b[0]=1, and gamma_n[0] has been set to 0.
        for t in range(1, T[b]):
            gamma_n[b, t, 0, 0] = -np.inf  # gamma_n[t]=0
            gamma_b[b, t, 0, 0] = 0  # gamma_b[t]=1
            for tao in range(1, t + 1):
                gamma_b[b, t, 0, 0] += gamma_b[b, tao - 1, 0, 0] + CTCOutLogProbs[
                    b, tao, blank]  # gamma_b[t] *= gamma_b[tao-1]*p[tao, blank]

    newhypo = torch.arange(1, numClasses).unsqueeze(-1).unsqueeze(0).unsqueeze(0)  # (1, 1, num_classes-1, 1)

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
            predictionLenBatch = predictionLenBatch[encoderIndex]
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
        # alpha = (batch * beam * (V-1))
        # CTCOutLogProbs = (batch * sequence length * V)
        # gamma_n or gamma_b = (batch * max time length * beamwidth * V <which is max num of candidates in one time step>)
        CTCHypoLogProbs = compute_CTC_prob(h, alpha[:, :, :, 1], CTCOutLogProbs, T, gamma_n, gamma_b, numBeam,
                                           numClasses - 1, blank, eosIx)
        alpha[:, :, :, 0] = Lambda * CTCHypoLogProbs + (1 - Lambda) * alpha[:, :, :, 1]
        hPaddingShape = list(h.shape)
        hPaddingShape[-2] = 1
        h = torch.cat((torch.zeros(hPaddingShape), h), dim=-2)  # (batch * beam * V * hypoLength)

        activeBatch = (l < T).nonzero().squeeze(-1).tolist()
        for b in activeBatch:
            for i in range(numBeam):
                Omegahat[b].append((h[b, i, -1], alpha[b, i, -1, 0]))  # c=eos

        alpha = torch.cat((torch.full((batch, numBeam, 1, 2), -np.inf), alpha), dim=-2)  # (batch * beam * V * hypoLength)
        alpha[:, :, -1, 0] = -np.inf
        predictionRes = alpha[:, :, :, 0].reshape(batch, -1).topk(beamWidth, -1).indices  # (batch * beamWidth)
        for b in range(batch):
            for pos, c in enumerate(predictionRes[b]):
                beam = c // numClasses
                c = c % numClasses
                Omega[b][l].append((h[b, beam, c], alpha[b, beam, c, 0], alpha[b, beam, c, 1]))
                gamma_n[b, :, pos, 0] = gamma_n[b, :, beam, c]
                gamma_b[b, :, pos, 0] = gamma_b[b, :, beam, c]
        gamma_n[:, :, :, 1:] = -np.inf
        gamma_b[:, :, :, 1:] = -np.inf
        predictionLenBatch += 1

    # predictionBatch = model.decoder.embed_tokens(predictionBatch.transpose(0, 1))  # (B, T')--embed-->(T', B, EED)
    predictionBatch = [sorted(Omegahat[b], key=lambda x: x[1], reverse=True)[0][0] for b in range(batch)]  #[([hypo], score)]
    predictionLenBatch = [len(prediction) - 1 for prediction in predictionBatch]
    return torch.cat([prediction[1:] for prediction in predictionBatch]).int(), torch.tensor(predictionLenBatch).int()


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
    evalCER = 0
    evalWER = 0
    evalPER = 0
    evalCCount = 0
    evalWCount = 0
    evalPCount = 0
    
    target_dictionary = task.target_dictionary
    spm = task.datasets[cfg.dataset.gen_subset].label_processors[0]
    symbols_to_ignore_spm = {target_dictionary.eos(), target_dictionary.pad()}

    Lambda = cfg.generation.Lambda
    codeDirectory = cfg.common_eval.results_path
    if os.path.exists(codeDirectory + "pred_%s.txt" % cfg.generation.decodeType):
        os.remove(codeDirectory + "pred_%s.txt" % cfg.generation.decodeType)
    if os.path.exists(codeDirectory + "trgt.txt"):
        os.remove(codeDirectory + "trgt.txt")

    model.eval()
    for batch, inputBatch in enumerate(evalProgress):
        inputBatch = utils.move_to_cuda(inputBatch) if use_cuda else inputBatch
        targetLenBatch = inputBatch['target_lengths']
        concatTargetoutBatch = inputBatch['target']
        with torch.no_grad():
            if cfg.generation.decodeType == "HYBRID":
                predictionBatch, predictionLenBatch = \
                    inference_hybrid(model, inputBatch, Lambda, cfg.generation.beamWidth,
                                     target_dictionary.eos(), 0, device="cuda" if use_cuda else "cpu")
            # elif inferenceParams["decodeType"] == "ATTN":
            #     predictionBatch, predictionLenBatch = \
            #         model.attentionAutoregression(inputBatch, False, device, inferenceParams["eosIx"])
            # elif inferenceParams["decodeType"] == "TFATTN":
            #     inputLenBatch, o  ;'9utputBatch = model(inputBatch, targetinBatch, targetLenBatch.long(), False)
            #     predictionBatch, predictionLenBatch = teacher_forcing_attention_decode(outputBatch[1], inferenceParams["eosIx"])
            elif cfg.generation.decodeType == "CTC":
                inputLenBatch, outputBatch = model(**inputBatch['net_input'])
                predictionBatch, predictionLenBatch = ctc_greedy_decode(outputBatch[0], inputLenBatch,
                                                                        target_dictionary.eos())
            else:
                raise RuntimeError(f"can not recognize decodeType:{cfg.generation.decodeType}")
            predictionStr = decode_fn(spm, predictionBatch.int().cpu(), symbols_to_ignore_spm)
            targetStr = decode_fn(spm, concatTargetoutBatch.int().cpu(), symbols_to_ignore_spm)

            with open("pred_%s.txt" % cfg.generation.decodeType, "a") as f:
                f.write(predictionStr)

            with open("trgt.txt", "a") as f:
                f.write(targetStr)

            c_edits, c_count = compute_error_ch(predictionBatch, concatTargetoutBatch, predictionLenBatch,
                                                targetLenBatch)
            evalCER += c_edits
            evalCCount += c_count
            w_edits, w_count = compute_error_word(predictionBatch, concatTargetoutBatch, predictionLenBatch,
                                                  targetLenBatch,
                                                  target_dictionary.pad())
            evalWER += w_edits
            evalWCount += w_count
            evalProgress.log({
                "CER": evalCER / evalCCount,
                "WER": evalWER / evalWCount
            }, step=batch+1)
            logger.info(
                "batch%d || Test CER: %.3f || Test WER: %.3f" % (batch + 1, evalCER / evalCCount, evalWER / evalWCount))

    evalCER /= evalCCount if evalCCount > 0 else 1
    evalWER /= evalWCount if evalWCount > 0 else 1
    evalPER /= evalPCount if evalPCount > 0 else 1
    return evalCER, evalWER, evalPER

def load_and_ensemble_args(args):
    conf = OmegaConf.load(args.yaml_path)
    conf.common.user_dir = args.user_dir
    conf.common.log_name = args.log_name
    conf.common_eval.results_path = os.path.join(args.output, args.log_name)
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
    log_name = cfg.common.log_name
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S',
                        filename=f'{log_name}.log', filemode='w')
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
        dataset=task.dataset(cfg.dataset.gen_subset),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), model.max_positions()
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
        # wandb_project=cfg.common.log_name,
    )

    logger.info("\nTesting the trained model .... \n")

    testCER, testWER, testPER = inference(model, progress, logger, task, cfg, True)

    logger.info("Test CER: %.3f || Test WER: %.3f" % (testCER, testWER))
    progress.print({
        "CER": testCER,
        "WER": testWER,
    })

    logger.info("\nTesting Done.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-dir', help='Config directory', default="/home/yfliu/av_hubert/avhubert/conf")
    parser.add_argument('--config-name', help='Name of config file with .postfix', default="hybrid_decode.yaml")
    parser.add_argument('--tsv-dir', help="to override task.label_dir and task.data", default="/home/yfliu/datasets/lrs3/30h_data")
    parser.add_argument('--ckpt-path', help='finetuned checkpoint path', default="/home/yfliu/output/finetune_hybrid/checkpoints/checkpoint_best.pt")
    parser.add_argument('--log-name', help='eval log file and wandb project name', default="decode_hybrid")
    parser.add_argument('--output', help='output dir without project name', default="/home/yfliu/output")
    parser.add_argument('--user-dir', help='command-line pwd result')
    parser.add_argument('--modalities', help='shuold be one of "AO", "VO" or "AV".', default="VO")
    args = parser.parse_args()
    args.yaml_path = os.path.join(args.config_dir, args.config_name)
    args.output = os.path.join(args.output, f'finetune_{args.log_name}')
    # load args from yaml file using OmegaConf
    args = load_and_ensemble_args(args)
    main(args)
