import os
import torch
import model_utils
import encoders
import decoders

import numpy as np
import torch.nn.functional as F
import torch.nn as nn

from von_mises_fisher import VonMisesFisher
from decorators import auto_init_args, auto_init_pytorch

from mem_nn import MemoryNetwork

MAX_LEN = 32


class base(nn.Module):
    def __init__(self, vocab_size, embed_dim, embed_init, experiment):
        super(base, self).__init__()
        self.expe = experiment
        self.eps = self.expe.config.eps
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def pos_loss(self, mask, vecs, func):
        batch_size, seq_len = mask.size()
        # batch size x seq len x MAX LEN
        logits = func(vecs)
        if MAX_LEN - seq_len:
            padded = torch.zeros(batch_size, MAX_LEN - seq_len).to(mask.device)
            new_mask = 1 - torch.cat([mask, padded], -1)
        else:
            new_mask = 1 - mask
        new_mask = new_mask.unsqueeze(1).expand_as(logits)
        logits.masked_fill_(new_mask.bool(), -float('inf'))
        loss = F.softmax(logits, -1)[:, np.arange(int(seq_len)),
                                     np.arange(int(seq_len))]
        loss = -(loss + self.eps).log() * mask

        loss = loss.sum(-1) / mask.sum(1)
        return loss.mean()

    def sample_gaussian(self, mean, logvar):
        sample = mean + torch.exp(0.5 * logvar) * \
            logvar.new_empty(logvar.size()).normal_()
        return sample

    def to_tensor(self, inputs):
        if torch.is_tensor(inputs):
            return inputs.clone().detach().to(self.device)
        else:
            return torch.tensor(inputs, device=self.device)

    def to_tensors(self, *inputs):
        return [self.to_tensor(inputs_) if inputs_ is not None and inputs_.size
                else None for inputs_ in inputs]

    def optimize(self, loss):
        self.opt.zero_grad()
        loss.backward()
        if self.expe.config.gclip is not None:
            torch.nn.utils.clip_grad_norm(
                self.parameters(), self.expe.config.gclip)
        self.opt.step()

    def init_optimizer(self, opt_type, learning_rate, weight_decay):
        if opt_type.lower() == "adam":
            optimizer = torch.optim.Adam
        elif opt_type.lower() == "rmsprop":
            optimizer = torch.optim.RMSprop
        elif opt_type.lower() == "sgd":
            optimizer = torch.optim.SGD
        else:
            raise NotImplementedError("invalid optimizer: {}".format(opt_type))

        opt = optimizer(
            params=filter(
                lambda p: p.requires_grad, self.parameters()
            ),
            weight_decay=weight_decay,
            lr=learning_rate)

        return opt

    def save(self, dev_bleu, dev_stats, test_bleu, test_stats,
             epoch, iteration=None, name="best"):
        save_path = os.path.join(self.expe.experiment_dir, name + ".ckpt")
        checkpoint = {
            "dev_bleu": dev_bleu,
            "dev_stats": dev_stats,
            "test_bleu": test_bleu,
            "test_stats": test_stats,
            "epoch": epoch,
            "iteration": iteration,
            "state_dict": self.state_dict(),
            "opt_state_dict": self.opt.state_dict(),
            "config": self.expe.config
        }
        torch.save(checkpoint, save_path)
        self.expe.log.info("model saved to {}".format(save_path))

    def load(self, checkpointed_state_dict=None, name="best"):
        if checkpointed_state_dict is None:
            save_path = os.path.join(self.expe.experiment_dir, name + ".ckpt")
            checkpoint = torch.load(save_path,
                                    map_location=lambda storage,
                                    loc: storage)
            self.load_state_dict(checkpoint['state_dict'])
            self.opt.load_state_dict(checkpoint.get("opt_state_dict"))
            self.expe.log.info("model loaded from {}".format(save_path))
            self.to(self.device)
            for state in self.opt.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            self.expe.log.info("transferred model to {}".format(self.device))
            return checkpoint.get('epoch', 0), \
                checkpoint.get('iteration', 0), \
                checkpoint.get('dev_bleu', 0), \
                checkpoint.get('test_bleu', 0)
        else:
            self.load_state_dict(checkpointed_state_dict)
            self.expe.log.info("model loaded from checkpoint.")
            self.to(self.device)
            self.expe.log.info("transferred model to {}".format(self.device))


class vgvae(base):
    @auto_init_pytorch
    @auto_init_args
    def __init__(self, vocab_size, embed_dim, embed_init, experiment):
        super(vgvae, self).__init__(
            vocab_size, embed_dim, embed_init, experiment)
        self.yencode = getattr(encoders, self.expe.config.yencoder_type)(
            embed_dim=embed_dim,
            embed_init=embed_init,
            hidden_size=self.expe.config.ensize,
            vocab_size=vocab_size,
            dropout=self.expe.config.dp,
            log=experiment.log)

        self.zencode = getattr(encoders, self.expe.config.zencoder_type)(
            embed_dim=embed_dim,
            embed_init=embed_init,
            hidden_size=self.expe.config.ensize,
            vocab_size=vocab_size,
            dropout=self.expe.config.dp,
            mlp_hidden_size=self.expe.config.mhsize,
            mlp_layer=self.expe.config.mlplayer,
            ncode=self.expe.config.ncode,
            nclass=self.expe.config.nclass,
            log=experiment.log)

        if "lstm" in self.expe.config.yencoder_type.lower():
            y_out_size = 2 * self.expe.config.ensize
        elif self.expe.config.yencoder_type.lower() == "word_avg":
            y_out_size = embed_dim

        if "lstm" in self.expe.config.zencoder_type.lower():
            z_out_size = 2 * self.expe.config.ensize
        elif self.expe.config.zencoder_type.lower() == "word_avg":
            z_out_size = embed_dim

        self.y_code2vec = model_utils.get_mlp(
            input_size=y_out_size,
            hidden_size=self.expe.config.mhsize,
            output_size=self.expe.config.ysize,
            n_layer=self.expe.config.ymlplayer,
            dropout=self.expe.config.dp)


        self.z_code2vec = model_utils.get_mlp(
            input_size=z_out_size,
            hidden_size=self.expe.config.mhsize,
            output_size=self.expe.config.zsize,
            n_layer=self.expe.config.zmlplayer,
            dropout=self.expe.config.dp)

        self.z_merge = model_utils.get_mlp(
            input_size=2 * self.expe.config.zsize,
            hidden_size=self.expe.config.mhsize,
            output_size=self.expe.config.zsize,
            n_layer=2,
            dropout=self.expe.config.dp)

        self.y_merge = model_utils.get_mlp(
            input_size=2 * self.expe.config.ysize,
            hidden_size=self.expe.config.mhsize,
            output_size=self.expe.config.ysize,
            n_layer=2,
            dropout=self.expe.config.dp)

        self.z_phi = model_utils.get_mlp(
            input_size=2 * self.expe.config.zsize,
            hidden_size=self.expe.config.mhsize,
            output_size=self.expe.config.zsize,
            n_layer=2,
            dropout=self.expe.config.dp)

        self.y_phi = model_utils.get_mlp(
            input_size=2 * self.expe.config.ysize,
            hidden_size=self.expe.config.mhsize,
            output_size=self.expe.config.ysize,
            n_layer=2,
            dropout=self.expe.config.dp)

        self.z_p_theta = model_utils.get_mlp(
            input_size=self.expe.config.zsize,
            hidden_size=self.expe.config.mhsize,
            output_size=self.expe.config.zsize,
            n_layer=2,
            dropout=self.expe.config.dp)

        self.y_p_theta = model_utils.get_mlp(
            input_size=self.expe.config.ysize,
            hidden_size=self.expe.config.mhsize,
            output_size=self.expe.config.ysize,
            n_layer=2,
            dropout=self.expe.config.dp)


        self.decode = getattr(decoders, self.expe.config.decoder_type)(
            embed_init=embed_init,
            embed_dim=embed_dim,
            ysize=self.expe.config.ysize,
            zsize=self.expe.config.zsize,
            mlp_hidden_size=self.expe.config.mhsize,
            mlp_layer=self.expe.config.mlplayer,
            hidden_size=self.expe.config.desize,
            dropout=self.expe.config.dp,
            vocab_size=vocab_size,
            log=experiment.log)

        if "lc" in self.expe.config.zencoder_type.lower():
            enc_embed_dim = embed_dim // self.expe.config.ncode *\
                self.expe.config.ncode
        else:
            enc_embed_dim = embed_dim

        self.enc_pos_decode = model_utils.get_mlp(
            input_size=self.expe.config.zsize + enc_embed_dim,
            hidden_size=self.expe.config.mhsize,
            n_layer=self.expe.config.mlplayer,
            output_size=MAX_LEN,
            dropout=self.expe.config.dp)

        self.dec_pos_decode = model_utils.get_mlp(
            input_size=self.expe.config.zsize + embed_dim,
            hidden_size=self.expe.config.mhsize,
            n_layer=self.expe.config.mlplayer,
            output_size=MAX_LEN,
            dropout=self.expe.config.dp)
        self.z_mem_nn = MemoryNetwork(self.expe.config.num_key_z, self.expe.config.zsize, self.device)
        self.y_mem_nn = MemoryNetwork(self.expe.config.num_key_y, self.expe.config.ysize, self.device)

    def sent2param(self, sent, sent_repl, mask):
        yembed, ycode = self.yencode(sent, mask)
        zembed, zcode = self.zencode(sent_repl, mask)

        yvecs = self.y_code2vec(ycode)
        zvecs = self.z_code2vec(zcode)

        return zembed, yvecs, zvecs

    def collect_embedding(self, sent1, sent_repl1, mask1, sent2, sent_repl2,
                mask2, tgt1, tgt_mask1, tgt2, tgt_mask2):
        self.eval()
        sent1, sent_repl1, mask1, sent2, sent_repl2, mask2, tgt1, \
            tgt_mask1, tgt2, tgt_mask2 = \
            self.to_tensors(sent1, sent_repl1, mask1, sent2, sent_repl2,
                            mask2, tgt1, tgt_mask1, tgt2, tgt_mask2)

        s1_zembed, s1_yvecs, s1_zvecs = \
            self.sent2param(sent1, sent_repl1, mask1)
        s2_zembed, s2_yvecs, s2_zvecs = \
            self.sent2param(sent2, sent_repl2, mask2)
        return s1_zvecs.cpu().detach().numpy(), s1_yvecs.cpu().detach().numpy(), s2_zvecs.cpu().detach().numpy(), s2_yvecs.cpu().detach().numpy()


    def gumbel_sample_with_p(self, p):
        # p is the probability of getting 1
        # Convert it to log prob and do some operations for gumbel sample
        log_p = torch.log(p + 1e-10).unsqueeze(axis=2)
        log_one_minus_p = (torch.log(1 - p + 1e-10)).unsqueeze(axis=2)

        # concat
        logits = torch.cat((log_one_minus_p, log_p), axis=2)

        sample = torch.nn.functional.gumbel_softmax(logits, hard=True)
        sample = sample * torch.cat((torch.zeros_like(log_p), torch.ones_like(log_p)), axis=2)
        return sample.sum(axis=2)

    def forward(self, sent1, sent_repl1, mask1, sent2, sent_repl2,
                mask2, tgt1, tgt_mask1, tgt2, tgt_mask2, vtemp, gtemp):
        self.train()
        sent1, sent_repl1, mask1, sent2, sent_repl2, mask2, tgt1, \
            tgt_mask1, tgt2, tgt_mask2 = \
            self.to_tensors(sent1, sent_repl1, mask1, sent2, sent_repl2,
                            mask2, tgt1, tgt_mask1, tgt2, tgt_mask2)

        s1_zembed, s1_yvecs, s1_zvecs = \
            self.sent2param(sent1, sent_repl1, mask1)
        s2_zembed, s2_yvecs, s2_zvecs = \
            self.sent2param(sent2, sent_repl2, mask2)




        # Using memory network
        z1_alpha_theta = self.z_mem_nn.get_alpha(self.z_p_theta(s1_zvecs))

        z1_phi_inp = torch.cat((s1_zvecs, s2_zvecs), axis=1)
        z1_alpha = self.z_mem_nn.get_alpha(self.z_phi(z1_phi_inp))
        z1_alpha_sample = self.gumbel_sample_with_p(z1_alpha)
        sent1_syntax = self.z_mem_nn.read(z1_alpha_sample)


        z2_alpha_theta = self.z_mem_nn.get_alpha(self.z_p_theta(s2_zvecs))

        z2_phi_inp = torch.cat((s2_zvecs, s1_zvecs), axis=1)
        z2_alpha = self.z_mem_nn.get_alpha(self.z_phi(z2_phi_inp))
        z2_alpha_sample = self.gumbel_sample_with_p(z2_alpha)
        sent2_syntax = self.z_mem_nn.read(z2_alpha_sample)


        y1_alpha_theta = self.y_mem_nn.get_alpha(self.y_p_theta(s1_yvecs))

        y1_phi_inp = torch.cat((s1_yvecs, s2_yvecs), axis=1)
        y1_alpha = self.y_mem_nn.get_alpha(self.y_phi(y1_phi_inp))
        y1_alpha_sample = self.gumbel_sample_with_p(y1_alpha)
        sent1_semantic = self.y_mem_nn.read(y1_alpha_sample)


        y2_alpha_theta = self.y_mem_nn.get_alpha(self.y_p_theta(s2_yvecs))
        y2_phi_inp = torch.cat((s2_yvecs, s1_yvecs), axis=1)
        y2_alpha = self.y_mem_nn.get_alpha(self.y_phi(y2_phi_inp))
        y2_alpha_sample = self.gumbel_sample_with_p(y2_alpha)
        sent2_semantic = self.y_mem_nn.read(y2_alpha_sample)

        # Merge
        sent1_syntax = self.z_merge(torch.cat((sent1_syntax, s1_zvecs), axis=1))
        sent2_syntax = self.z_merge(torch.cat((sent2_syntax, s2_zvecs), axis=1))
        sent1_semantic = self.y_merge(torch.cat((sent1_semantic, s1_yvecs), axis=1))
        sent2_semantic = self.y_merge(torch.cat((sent2_semantic, s2_yvecs), axis=1))


        logloss1, s1_decs = self.decode(
            sent1_semantic, sent1_syntax, tgt2, tgt_mask2)
        logloss2, s2_decs = self.decode(
            sent2_semantic, sent2_syntax, tgt1, tgt_mask1)
        ploss = torch.zeros_like(logloss1)


        z_kl = self.z_mem_nn.kl_div(z1_alpha, z1_alpha_theta) + self.z_mem_nn.kl_div(z2_alpha, z2_alpha_theta)
        y_kl = self.y_mem_nn.kl_div(y1_alpha, y1_alpha_theta) + self.y_mem_nn.kl_div(y2_alpha, y2_alpha_theta)

        # Reconstruction
        # Syntax for reconstruction
        sent1_syntax_rec = self.z_merge(torch.cat((torch.zeros_like(sent1_syntax).to(self.device), s1_zvecs), axis=1))
        sent2_syntax_rec = self.z_merge(torch.cat((torch.zeros_like(sent2_syntax).to(self.device), s2_zvecs), axis=1))
        sent1_semantic_rec = self.y_merge(torch.cat((torch.zeros_like(sent1_semantic).to(self.device), s1_yvecs), axis=1))
        sent2_semantic_rec = self.y_merge(torch.cat((torch.zeros_like(sent2_semantic).to(self.device), s2_yvecs), axis=1))
        logloss3, s3_decs = self.decode(
            sent1_semantic_rec, sent1_syntax_rec, tgt1, tgt_mask1)
        logloss4, s4_decs = self.decode(
            sent2_semantic_rec, sent2_syntax_rec, tgt2, tgt_mask2)
        rec_logloss = logloss3 + logloss4


        para_logloss = logloss1 + logloss2
        #rec_logloss = torch.zeros_like(para_logloss).to(self.device)

        loss = self.expe.config.lratio * rec_logloss + \
            self.expe.config.plratio * para_logloss + \
            vtemp * y_kl + gtemp * z_kl + \
            self.expe.config.pratio * ploss

        return loss, y_kl, z_kl, rec_logloss, para_logloss, ploss

    def greedy_decode(self, semantics, semantics_mask,
                      syntax, syntax_mask, max_len):
        self.eval()
        syntax, syntax_mask, semantics, semantics_mask = \
            self.to_tensors(syntax, syntax_mask, semantics, semantics_mask)

        s1 = semantics
        s2 = syntax
        s1_mask = semantics_mask
        s2_mask = syntax_mask

        yembed, ycode = self.yencode(s1, s1_mask)
        zembed, zcode = self.zencode(s1, s1_mask)

        yvecs = self.y_code2vec(ycode)
        zvecs = self.z_code2vec(zcode)

        # Using memory network
        z_alpha = self.z_mem_nn.get_alpha(self.z_p_theta(zvecs))
        z_alpha_sample = self.gumbel_sample_with_p(z_alpha)
        z_tilde = self.z_mem_nn.read(z_alpha_sample)

        y_alpha = self.y_mem_nn.get_alpha(self.y_p_theta(yvecs))
        y_alpha_sample = self.gumbel_sample_with_p(y_alpha)
        y_tilde = self.y_mem_nn.read(y_alpha_sample)

        z_tilde = self.z_merge(torch.cat((z_tilde, zvecs), axis=1))
        y_tilde = self.y_merge(torch.cat((y_tilde, yvecs), axis=1))


        return self.decode.greedy_decode(y_tilde, z_tilde, max_len)
