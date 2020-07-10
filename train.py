import train_helper
import data_utils
import config

import models
from tensorboardX import SummaryWriter

import numpy as np
from sklearn.cluster import KMeans
import torch
import time
import os
import pickle

best_dev_bleu = test_bleu = 0


def run(e, save_prefix):
    global best_dev_bleu, test_bleu

    e.log.info("*" * 25 + " DATA PREPARATION " + "*" * 25)
    dp = data_utils.data_processor(
        train_path=e.config.train_path,
        experiment=e)
    data, W = dp.process()

    e.log.info("*" * 25 + " DATA PREPARATION " + "*" * 25)
    e.log.info("*" * 25 + " MODEL INITIALIZATION " + "*" * 25)

    model = models.vgvae(
        vocab_size=len(data.vocab),
        embed_dim=e.config.edim if W is None else W.shape[1],
        embed_init=W,
        experiment=e)

    start_epoch = true_it = 0
    best_dev_stats = test_stats = None
    if e.config.resume:
        start_epoch, _, best_dev_bleu, test_bleu = \
            model.load(name="latest")
        e.log.info(
            "resumed from previous checkpoint: start epoch: {}, "
            "iteration: {}, best dev bleu: {:.3f}, test bleu: {:.3f}, "
            .format(start_epoch, true_it, best_dev_bleu, test_bleu))

    e.log.info(model)
    e.log.info("*" * 25 + " MODEL INITIALIZATION " + "*" * 25)

    if e.config.summarize:
        writer = SummaryWriter(e.experiment_dir)

    train_batch = data_utils.minibatcher(
        data1=data.train_data[0],
        tag1=data.train_tag[0],
        data2=data.train_data[1],
        tag2=data.train_tag[1],
        tag_bucket=data.tag_bucket,
        vocab_size=len(data.vocab),
        batch_size=e.config.batch_size,
        shuffle=True,
        p_replace=e.config.wr,
        p_scramble=e.config.ps)

    kmeans_batch = data_utils.minibatcher(
        data1=data.train_data[0],
        tag1=data.train_tag[0],
        data2=data.train_data[1],
        tag2=data.train_tag[1],
        tag_bucket=data.tag_bucket,
        vocab_size=len(data.vocab),
        batch_size=e.config.batch_size,
        shuffle=False,
        p_replace=e.config.wr,
        p_scramble=e.config.ps)

    dev_eval = train_helper.evaluator(
        e.config.dev_inp_path, e.config.dev_ref_path,
        model, data.vocab, data.inv_vocab, e)
    test_eval = train_helper.evaluator(
        e.config.test_inp_path, e.config.test_ref_path,
        model, data.vocab, data.inv_vocab, e)

    e.log.info("Training start ...")
    train_stats = train_helper.tracker(["loss", "vmf_kl", "gauss_kl",
                                        "rec_logloss", "para_logloss",
                                        "wploss"])

    n_clusters_z = e.config.num_key_z
    n_clusters_y = e.config.num_key_y
    kmeans_z = KMeans(n_clusters=n_clusters_z, n_jobs=-1) # multiprocessing, use all processors
    kmeans_y = KMeans(n_clusters=n_clusters_y, n_jobs=-1) # multiprocessing, use all processors
    for epoch in range(start_epoch, e.config.n_epoch):
        if (epoch + 1) % 2 == 0:
            # K-means training
            e.log.info("running kmeans... ")
            st = time.time()
            embeds = []
            for it, (s1, sr1, m1, s2, sr2, m2, t1, tm1, t2, tm2, _) in \
                    enumerate(kmeans_batch):

                z1_embeds, y1_embeds, z2_embeds, y2_embeds = model.collect_embedding(s1, sr1, m1, s2, sr2, m2, t1, tm1,
                      t2, tm2)
                embeds.append((z1_embeds, y1_embeds, z2_embeds, y2_embeds))
            z1_embeds = np.concatenate([x[0] for x in embeds], axis=0)
            y1_embeds = np.concatenate([x[1] for x in embeds], axis=0)
            z2_embeds = np.concatenate([x[2] for x in embeds], axis=0)
            y2_embeds = np.concatenate([x[3] for x in embeds], axis=0)

            #    K-means for z memory network
            kmeans_z.fit(z1_embeds)
            cluster_assign = kmeans_z.labels_


            model.z_mem_nn.keys.data = torch.tensor(kmeans_z.cluster_centers_).squeeze().to(model.z_mem_nn.device).data
            values = []
            for i in range(n_clusters_z):
                v = z2_embeds[cluster_assign == i].mean(axis=0, keepdims=True)
                values.append(v)
            model.z_mem_nn.values.data = torch.tensor(values).squeeze().to(model.z_mem_nn.device).data

            #    K-means for y memory network
            kmeans_y.fit(y1_embeds)
            cluster_assign = kmeans_y.labels_

            model.y_mem_nn.keys.data = torch.tensor(kmeans_y.cluster_centers_).squeeze().to(model.y_mem_nn.device).data
            values = []
            for i in range(n_clusters_y):
                v = y2_embeds[cluster_assign == i].mean(axis=0, keepdims=True)
                values.append(v)
            model.y_mem_nn.values.data = torch.tensor(values).squeeze().to(model.y_mem_nn.device).data

            e.log.info("kmeans elapsed time: {:.2f}(s) ".format(time.time()-st))


        for it, (s1, sr1, m1, s2, sr2, m2, t1, tm1, t2, tm2, _) in \
                enumerate(train_batch):
            true_it = it + 1 + epoch * len(train_batch)

            loss, kl, kl2, rec_logloss, para_logloss, wploss = \
                model(s1, sr1, m1, s2, sr2, m2, t1, tm1,
                      t2, tm2, e.config.vmkl, e.config.gmkl)
            model.optimize(loss)
            train_stats.update(
                {"loss": loss, "vmf_kl": kl, "gauss_kl": kl2,
                 "para_logloss": para_logloss, "rec_logloss": rec_logloss,
                 "wploss": wploss},
                len(s1))

            if (true_it + 1) % e.config.print_every == 0 or \
                    (true_it + 1) % len(train_batch) == 0:
                summarization = train_stats.summarize(
                    "epoch: {}, it: {} (max: {}), kl_temp(v|g): {:.2E}|{:.2E}"
                    .format(epoch, it, len(train_batch),
                            e.config.vmkl, e.config.gmkl))
                e.log.info(summarization)
                if e.config.summarize:
                    for name, value in train_stats.stats.items():
                        writer.add_scalar(
                            "train/" + name, value, true_it)
                train_stats.reset()

            if (true_it + 1) % e.config.eval_every == 0 or \
                    (true_it + 1) % len(train_batch) == 0:
                e.log.info("*" * 25 + " DEV SET EVALUATION " + "*" * 25)

                dev_stats, dev_bleu = dev_eval.evaluate("gen_dev")

                e.log.info("*" * 25 + " DEV SET EVALUATION " + "*" * 25)

                if e.config.summarize:
                    for name, value in dev_stats.items():
                        writer.add_scalar(
                            "dev/" + name, value, true_it)

                if best_dev_bleu < dev_bleu:
                    best_dev_bleu = dev_bleu
                    best_dev_stats = dev_stats

                    e.log.info("*" * 25 + " TEST SET EVALUATION " + "*" * 25)

                    test_stats, test_bleu = test_eval.evaluate("gen_test")

                    e.log.info("*" * 25 + " TEST SET EVALUATION " + "*" * 25)

                    model.save(
                        dev_bleu=best_dev_bleu,
                        dev_stats=best_dev_stats,
                        test_bleu=test_bleu,
                        test_stats=test_stats,
                        iteration=true_it,
                        epoch=epoch)
                    with open(os.path.join(save_prefix, 'kmeans_z.pkl'), 'wb') as handle:
                        pickle.dump(kmeans_z, handle)
                    with open(os.path.join(save_prefix, 'kmeans_y.pkl'), 'wb') as handle:
                        pickle.dump(kmeans_y, handle)

                    if e.config.summarize:
                        for name, value in test_stats.items():
                            writer.add_scalar(
                                "test/" + name, value, true_it)

                e.log.info("best dev bleu: {:.4f}, test bleu: {:.4f}"
                           .format(best_dev_bleu, test_bleu))

        model.save(
            dev_bleu=best_dev_bleu,
            dev_stats=best_dev_stats,
            test_bleu=test_bleu,
            test_stats=test_stats,
            iteration=true_it,
            epoch=epoch + 1,
            name="latest")
        with open(os.path.join(save_prefix, 'kmeans_z.pkl'), 'wb') as handle:
            pickle.dump(kmeans_z, handle)
        with open(os.path.join(save_prefix, 'kmeans_y.pkl'), 'wb') as handle:
            pickle.dump(kmeans_y, handle)

        time_per_epoch = (e.elapsed_time / (epoch - start_epoch + 1))
        time_in_need = time_per_epoch * (e.config.n_epoch - epoch - 1)
        e.log.info("elapsed time: {:.2f}(h), "
                   "time per epoch: {:.2f}(h), "
                   "time needed to finish: {:.2f}(h)"
                   .format(e.elapsed_time, time_per_epoch, time_in_need))

        if time_per_epoch + e.elapsed_time > 3.7 and e.config.auto_disconnect:
            exit(1)

    test_gen_stats, test_res = test_eval.evaluate("gen_test")


if __name__ == '__main__':

    args = config.get_base_parser().parse_args()

    def exit_handler(*args):
        print(args)
        print("best dev bleu: {:.4f}, test bleu: {:.4f}"
              .format(best_dev_bleu, test_bleu))
        exit()

    train_helper.register_exit_handler(exit_handler)

    with train_helper.experiment(args, args.save_prefix) as e:

        e.log.info("*" * 25 + " ARGS " + "*" * 25)
        e.log.info(args)
        e.log.info("*" * 25 + " ARGS " + "*" * 25)

        run(e, args.save_prefix)
