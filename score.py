###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse
import torch
import data
import os

from utils import batchify, get_batch, repackage_hidden
from bleu import cal_bleu

parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./penn',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f)


if args.cuda:
    model.cuda()
else:
    model.cpu()

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

eval_batch_size = 1

val_data = batchify(corpus.valid, eval_batch_size, args)

ntokens = len(corpus.dictionary)


def score(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    target_path = 'target.txt'
    if os.path.exists(target_path):
        os.remove(target_path)
                 
    if os.path.exists(args.outf):
        os.remove(args.outf)
    print("generating...") 
    with open(args.outf, 'w') as outf, open(target_path, 'w') as tarf: 
        step = 0
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i, args)
            targets = targets.view(-1)
            
            output, hidden = model(data, hidden)
            output = output.squeeze()
            output = output.argmax(dim=-1)
            for indx in output:
                step += 1
                word_idx = indx.cpu()
                word = corpus.dictionary.idx2word[word_idx]
                outf.write(word + ('\n' if step % 20 == 19 else ' '))
            
            for indx in targets:
                step += 1
                word_idx = indx.cpu()
                word = corpus.dictionary.idx2word[word_idx]
                tarf.write(word + ('\n' if step % 20 == 19 else ' '))
    
            hidden = repackage_hidden(hidden)
    print("generated!")    
    # get BLEU score
    bleu1, bleu2, bleu3, bleu4 = cal_bleu(target_path, args.outf)
    print('BLEU_1(Cumulative 1-gram): %f' % bleu1)
    print('BLEU_2(Cumulative 2-gram): %f' % bleu2)
    print('BLEU_3(Cumulative 3-gram): %f' % bleu3)
    print('BLEU_4(Cumulative 4-gram): %f' % bleu4)
    
    

if __name__ == '__main__':
    score(val_data, eval_batch_size)
