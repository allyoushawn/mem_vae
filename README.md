# Variational Paraphrasing with Memory Network

The repository is the implementation of a paraphrasing model with memory network. The details is in mem\_vae.pdf

It is based on the PyTorch implementation of [Controllable Paraphrase Generation with a Syntactic Exemplar](https://github.com/mingdachen/syntactic-template-generation)

To set up and run the code, please refer to the original README.md, which is README\_syntactic.md


Also, set upt "glove" in run.sh

``bash run.sh`` should allow you to run the code.

## Requirements

- Python 3.5
- PyTorch >= 1.0
- NLTK
- [tqdm](https://github.com/tqdm/tqdm)
- [py-rouge](https://github.com/Diego999/py-rouge)
- [zss](https://github.com/timtadh/zhang-shasha)

## Resource
Please download the content of the following links and put it in the evaluation folder.
- [evaluation (including multi-bleu, METEOR and a copy of Stanford CoreNLP)](https://drive.google.com/drive/folders/1FJjvMldeZrJnQd-iVXJ3KGFBLEvsndNY?usp=sharing)
