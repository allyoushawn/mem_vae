import os
import sys
import pickle
import argparse
import tempfile
import subprocess

import torch
import models
import data_utils
import train_helper

import numpy as np

from beam_search import beam_search, get_gen_fn
from config import BOS_IDX, EOS_IDX, MAX_GEN_LEN
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--save_file', '-s', type=str)
parser.add_argument('--vocab_file', '-v', type=str)
parser.add_argument('--input_file', '-i', type=str)
parser.add_argument('--ref_file', '-r', type=str)
parser.add_argument('--output_file', '-o', type=str)
parser.add_argument('--beam_size', '-bs', type=int, default=10)
args = parser.parse_args()


save_dict = torch.load(
    args.save_file,
    map_location=lambda storage,
    loc: storage)

config = save_dict['config']
checkpoint = save_dict['state_dict']
config.debug = True

with open(args.vocab_file, "rb") as fp:
    W, vocab = pickle.load(fp)
inv_vocab = {i: w for w, i in vocab.items()}

if config.decoder_type == "lstm":
    config.decoder_type = "lstm_z2y"
    config.ncode = None
    config.nclass = None

with train_helper.experiment(config, config.save_prefix) as e:
    e.log.info("vocab loaded from: {}".format(args.vocab_file))
    model = models.vgvae(
        vocab_size=len(vocab),
        embed_dim=e.config.edim if W is None else W.shape[1],
        embed_init=W,
        experiment=e)
    model.load(checkpointed_state_dict=checkpoint)
    e.log.info(model)

    semantics_input = []
    syntax_input = []
    e.log.info("loading from: {}".format(args.input_file))
    with open(args.input_file) as fp:
        for line in fp:
            seman_in, syn_in = line.strip().split("\t")
            semantics_input.append(
                [vocab.get(w.lower(), 0) for w in
                 seman_in.strip().split(" ")])
            syntax_input.append([vocab.get(w.lower(), 0) for w in
                                 syn_in.strip().split(" ")])
    e.log.info("#evaluation data: {}, {}".format(
        len(semantics_input),
        len(syntax_input)))

    op_f = open(args.output_file, 'w')
    e.log.info('generation saving to {}'.format(args.output_file))
    e.log.info('beam size: {}'.format(args.beam_size))
    for s1, _, m1, s2, _, m2, _, _, _, _, _ in \
        tqdm(data_utils.minibatcher(
                data1=np.array(semantics_input),
                tag1=np.array(semantics_input),
                data2=np.array(syntax_input),
                tag2=np.array(syntax_input),
                tag_bucket=None,
                batch_size=100,
                p_replace=0.,
                shuffle=False,
                p_scramble=0.)):
            with torch.no_grad():
                batch_gen = model.greedy_decode(
                    s1, m1, s2, m2, MAX_GEN_LEN)
            for gen in batch_gen:
                curr_gen = []
                for i in gen:
                    if i == EOS_IDX:
                        break
                    curr_gen.append(inv_vocab[int(i)])

                op_f.write(" ".join(curr_gen))
                op_f.write("\n")
    op_f.close()
