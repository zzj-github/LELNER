
from model.soft_matcher import *
from config import Reader, Config, ContextEmb
from config.utils import load_bert_vec
import argparse
import random

def parse_arguments(parser):
    parser.add_argument('--eval', type=str, default="False")
    parser.add_argument('--device', type=str, default="cuda:0", choices=['cpu', 'cuda:0'],
                        help="GPU/CPU devices")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--digit2zero', action="store_true", default=True,
                        help="convert the number to 0, make it true is better")
    parser.add_argument('--dataset', type=str, default="CONLL")
    parser.add_argument('--embedding_file', type=str, default="dataset/glove.6B.100d.txt",
                        help="we will using random embeddings if file do not exist")
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default="sgd")
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--l2', type=float, default=1e-8)
    parser.add_argument('--lr_decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--train_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--dev_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--test_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--model_folder', type=str, default="english_model", help="The name to save the model files")
    parser.add_argument('--hidden_dim', type=int, default=300, help="hidden size of the LSTM")
    parser.add_argument('--use_crf_layer', type=int, default=1, help="1 is for using crf layer, 0 for not using CRF layer", choices=[0,1])
    parser.add_argument('--dropout', type=float, default=0.5, help="dropout for embedding")
    parser.add_argument('--use_char_rnn', type=int, default=1, choices=[0, 1], help="use character-level lstm, 0 or 1")
    parser.add_argument('--context_emb', type=str, default="none")
    parser.add_argument('--ds_setting', nargs='+', help="+ hard / soft matching")
    parser.add_argument('--percentage', type=int, default=100, help="how much percentage of training dataset to use")

    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args


parser = argparse.ArgumentParser()
opt = parse_arguments(parser)
conf = Config(opt)
reader = Reader(conf.digit2zero)
dataset, max_length, label_length = reader.read_trigger_txt(conf.trigger_file, -1)
reader.merge_labels(dataset)

trains = reader.read_txt(conf.train_file, conf.dev_num)
devs = reader.read_txt(conf.dev_file, conf.dev_num)
tests = reader.read_txt(conf.test_file, conf.test_num)
print(len(dataset))
if conf.context_emb == ContextEmb.bert:
    print('Loading the BERT vectors for all datasets.')
    conf.context_emb_size = load_bert_vec(conf.trigger_file + "." + conf.context_emb.name + ".vec", dataset)


# setting for data
conf.use_iobes(trains)
conf.use_iobes(dataset)
conf.use_iobes(devs)
conf.use_iobes(tests)

conf.build_label_idx(dataset)
conf.build_word_idx(dataset, devs, tests)
conf.build_emb_table()
conf.map_insts_ids(trains)
conf.map_insts_ids(dataset)
conf.map_insts_ids(devs)
conf.map_insts_ids(tests)

encoder = SoftMatcher(conf, label_length)
trainer = SoftMatcherTrainerc(encoder, conf, devs, tests)

if opt.eval == "True":
    trainer.eval_result(dataset,  devs, tests)
else:

    random.shuffle(dataset)
    trainer.train_model(conf.num_epochs, dataset,  devs, tests, True)


