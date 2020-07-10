exp_dir=exp_cvae
model_dir=exp_cvae/desize100edim100ensize100gmkl0.1num_key_z50vmkl0.1ymlplayer2ysize50zmlplayer2zsize50
python generate.py -s $model_dir/best.ckpt -v vocab/pre_vocab_50000 -i data/test_input.txt -o $exp_dir/test_out.txt -bs 10
