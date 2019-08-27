import os
import argparse
import logging
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import torch
from torch.optim.lr_scheduler import StepLR
import torchtext

import seq2seq
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss import Perplexity
from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3


parser = argparse.ArgumentParser()
parser.add_argument('--train_path', action='store', dest='train_path',help='Path to train data')
parser.add_argument('--dev_path', action='store', dest='dev_path',help='Path to dev data')
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment',help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--resume', action='store_true', dest='resume',default=False,help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log_level', dest='log_level',default='info',help='Logging level.')

opt = parser.parse_args()

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

if opt.load_checkpoint is not None:
    logging.info("loading checkpoint from {}".format(os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)))
    checkpoint_path = os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model            # 连模型都有
    input_vocab = checkpoint.input_vocab  # 这个checkpoint里好像啥都有啊我擦
    output_vocab = checkpoint.output_vocab
else:
    # Prepare dataset准备数据
    src = SourceField()
    tgt = TargetField()
    max_len = 4000

    def len_filter(example):
        return len(example.src) <= max_len and len(example.tgt) <= max_len

    train = torchtext.data.TabularDataset(path=opt.train_path, format='tsv', fields=[('src', src), ('tgt', tgt)],
                                          filter_pred=len_filter)
    dev = torchtext.data.TabularDataset(path=opt.dev_path, format='tsv', fields=[('src', src), ('tgt', tgt)],
                                        filter_pred=len_filter)

    src.build_vocab(train, max_size=50000)
    tgt.build_vocab(train, max_size=50000)
    input_vocab = src.vocab
    output_vocab = tgt.vocab

    # Prepare loss
    weight = torch.ones(len(tgt.vocab))
    pad = tgt.vocab.stoi[tgt.pad_token]
    loss = Perplexity(weight, pad)
    if torch.cuda.is_available():
        loss.cuda()

    seq2seq = None
    optimizer = None
    if not opt.resume:
        # 初始化模型
        hidden_size = 128
        bidirectional = True
        # 创建编码层
        encoder = EncoderRNN(len(src.vocab), max_len, hidden_size, bidirectional=bidirectional, variable_lengths=True)
        # 创建解码层
        decoder = DecoderRNN(len(tgt.vocab), max_len, hidden_size * 2 if bidirectional else hidden_size,
                             dropout_p=0.2, use_attention=True, bidirectional=bidirectional,
                             eos_id=tgt.eos_id, sos_id=tgt.sos_id)
        # 创建序列模型
        seq2seq = Seq2seq(encoder, decoder)

        if torch.cuda.is_available():
            seq2seq.cuda()

        for param in seq2seq.parameters():
            param.data.uniform_(-0.08, 0.08)

    # train
    t = SupervisedTrainer(loss=loss, batch_size=32, checkpoint_every=50, print_every=10, expt_dir=opt.expt_dir)
    # 模型名字  where to save the model?
    seq2seq = t.train(seq2seq, train, num_epochs=6, dev_data=dev, optimizer=optimizer,
                      teacher_forcing_ratio=0.5, resume=opt.resume)

predictor = Predictor(seq2seq, input_vocab, output_vocab)

while True:
    seq_str = raw_input("Type in a source sequence:")
    seq = seq_str.strip().split()
    print(predictor.predict(seq))
