#!/bin/bash

# Reset the environment
exp_dir=exp_cvae
data=data
rm -rf $exp_dir
rm -rf vocab

ln -s /media/hdd2/cmu_lti_archive/10708/10708_project/utils/eval/evaluation evaluation

# Training
glove=/media/hdd/GloVe/glove.6B.100d.txt
python train.py \
    --debug 0 \
    --auto_disconnect 1 \
    --save_prefix $exp_dir \
    --decoder_type lstm_z2y \
    --yencoder_type word_avg \
    --zencoder_type bilstm \
    --n_epoch 20 \
    --train_path $data/train.txt \
    --train_tag_path $data/train.tag \
    --tag_vocab_file $data/word2tag.pkl \
    --embed_file $glove \
    --embed_type glove \
    --dev_inp_path $data/dev_input.txt \
    --dev_ref_path $data/dev_ref.txt \
    --test_inp_path $data/test_input.txt \
    --test_ref_path $data/test_ref.txt \
    --pre_train_emb 1 \
    --vocab_file vocab \
    --vocab_size 50000 \
    --batch_size 30 \
    --dropout 0.0 \
    --l2 0.0 \
    --learning_rate 1e-3 \
    --word_replace 0.0 \
    --max_vmf_kl_temp 1e-1 \
    --max_gauss_kl_temp 1e-1 \
    --zmlp_n_layer 2 \
    --ymlp_n_layer 2 \
    --mlp_n_layer 3 \
    --para_logloss_ratio 1.0 \
    --ploss_ratio 1.0 \
    --mlp_hidden_size 100 \
    --ysize 50 \
    --zsize 50 \
    --num_key_z 50 \
    --num_key_y 500 \
    --embed_dim 100 \
    --encoder_size 100 \
    --decoder_size 100 \
    --p_scramble 0.0 \
    --print_every 500 \
    --eval_every 5000 \
    --summarize 1
