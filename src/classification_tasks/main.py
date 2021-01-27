import os
import numpy as np
from numpy import linalg as LA
import argparse
from tqdm import tqdm
from collections import defaultdict
import random
import time
from time import sleep
from tabulate import tabulate

import torch
import torch.nn as nn
from classification_models import EmbAttModel, BiLSTMAttModel, BiLSTMModel
import pickle

import log
import util

# gender_tokens = ["he", "she", "her", "his", "him"]

def read_dataset(data_file, block_words=None, block_file=None, attn_file=None, clip_vocab=False, vocab_size=20000, to_anon=False):

    data_lines = open(data_file).readlines()
    global w2i
    global w2c

    if clip_vocab:
        for line in data_lines:
            tag, words = line.strip().lower().split("\t")

            for word in words.split():
                w2c[word] += 1.0
            
        # take only top VOCAB_SIZE words
        word_freq_list = sorted(w2c.items(), key=lambda x: x[1], reverse=True)[:vocab_size - len(w2i)]

        for idx, (word, freq) in enumerate(word_freq_list):
            temp = w2i[word] # assign the next available idx
    
        w2i = defaultdict(lambda: UNK, w2i)

    if block_file is not None:
        block_lines = open(block_file).readlines()
        if len(data_lines) != len(block_lines):
            raise ValueError("num lines in data file does not match w/ block file")

    if attn_file is not None:
        attn_lines = open(attn_file).readlines()
        if len(data_lines) != len(attn_lines):
            raise ValueError("num lines in data file does not match w/ attn file")

    for idx, data_line in enumerate(data_lines):
        tag, words = data_line.strip().lower().split("\t")
        if to_anon:
            words = util.anonymize(words)

        # populate block ids
        words = words.strip().split()
        block_ids = [0 for _ in words]
        attn_wts = None
        if block_words is not None:
            block_ids = [1 if i in block_words else 0 for i in words]
        elif block_file is not None:
            block_ids = [int(i) for i in block_lines[idx].strip().split()]

        if attn_file is not None:
            attn_wts = [float(i) for i in attn_lines[idx].strip().split()]
            # neglect_top = max(0, min(BLOCK_TOP, len(words) - 1))
            # top_ids = np.argsort(neg_attn_wts)[: neglect_top]
            # for i in top_ids:
            #     block_ids[i] = 1

        # check for the right len
        if len(block_ids) != len(words):
            raise ValueError("num of block words not equal to words")
        # done populating
        yield (idx, [w2i[x] for x in words], block_ids, attn_wts, t2i[tag])

def quantify_attention(ix, p, block_ids):
    sent_keyword_idxs = [idx for idx, val in enumerate(block_ids) if val == 1]
    base_prop = len(sent_keyword_idxs) / len(ix)
    att_prop = sum([p[i] for i in sent_keyword_idxs])
    return base_prop, att_prop

def quantify_norms(ix, word_embeddings, block_ids):
    sent_keyword_idxs = [idx for idx, val in enumerate(block_ids) if val == 1]
    base_ratio = len(sent_keyword_idxs) / len(ix)
    attn_ratio = sum([LA.norm(word_embeddings[i]) for i in sent_keyword_idxs])
    # normalize the attn_ratio
    attn_ratio /= sum([LA.norm(emb) for emb in word_embeddings])
    return base_ratio, attn_ratio

def calc_hammer_loss(ix, attention, block_ids, coef=0.0):
    sent_keyword_idxs = [idx for idx, val in enumerate(block_ids) if val == 1]
    if len(sent_keyword_idxs) == 0:
        return torch.zeros([1]).type(float_type)
    loss = -1 * coef * torch.log(1 - torch.sum(attention[sent_keyword_idxs]))
    return loss

def calc_kld_loss(p, q, coef=0.0):
    if p is None or q is None:
        return torch.tensor([0.0]).type(float_type)
    return -1 * coef * torch.dot(p, torch.log(p/q))

def entropy(p):
    return torch.distributions.Categorical(probs=p).entropy()

def calc_entropy_loss(p, beta):
    return -1 * beta * entropy(p)

def evaluate(dataset, iter, model, name='test', attn_stats=False, num_vis=0, emb_size=128, understand=False, flow=False, c_entropy=0., c_kld=0., c_hammer=0.):
    print ("evaluating on %s set" %(name))
    # Perform testing
    test_correct = 0.0
    test_base_prop = 0.0
    test_attn_prop = 0.0
    test_base_emb_norm = 0.0
    test_attn_emb_norm = 0.0
    test_base_h_norm = 0.0
    test_attn_h_norm = 0.0

    example_data = []

    total_loss = 0.0
    if num_vis > 0 and understand:
        wts, bias = model.get_linear_wts()
        print ("Weights below")
        print (wts.detach().cpu().numpy())
        print ("bias below")
        print (bias.detach().cpu().numpy())
    for idx, words, block_ids, attn_orig , tag in dataset:
        words_t = torch.tensor([words]).type(tensor_type)
        tag_t = torch.tensor([tag]).type(tensor_type)
        if attn_orig is not None:
            attn_orig = torch.tensor(attn_orig).type(float_type)

        block_ids_t = torch.tensor([block_ids]).type(float_type)

        if name == 'test' and flow:
            pred, attn = model(words_t, block_ids_t)
        else:
            pred, attn = model(words_t)
        attention = attn[0]

        if not flow or (name != 'test'):
            assert 0.99 < torch.sum(attention).item() < 1.01

        ce_loss = calc_ce_loss(pred, tag_t)
        entropy_loss = calc_entropy_loss(attention, c_entropy)
        hammer_loss = calc_hammer_loss(words, attention,
                                        block_ids, c_hammer)
        kld_loss = calc_kld_loss(attention, attn_orig, c_kld)

        assert hammer_loss.item() >= 0.0
        assert ce_loss.item() >= 0.0

        loss = ce_loss + entropy_loss + hammer_loss
        total_loss += loss.item()

        word_embeddings = model.get_embeddings(words_t)
        word_embeddings = word_embeddings[0].detach().cpu().numpy()
        assert len(words) == len(word_embeddings)

        final_states = model.get_final_states(words_t)
        final_states = final_states[0].detach().cpu().numpy()
        assert len(words) == len(final_states)

        predict = pred[0].argmax().item()
        if predict == tag:
            test_correct += 1


        if idx < num_vis:

            attn_scores = attn[0].detach().cpu().numpy()
            # util.pretty_importance_scores_vertical([i2w[w] \
            #     for w in words], attn_scores)

            example_data.append([[i2w[w] for w in words], attn_scores, i2t[predict], i2t[tag]])


            if understand:
                headers = ['words', 'attn'] + ['e' + str(i + 1) for i in range(emb_size)]
                tabulated_list = []
                for j in range(len(words)):
                    temp_list =  [i2w[words[j]], attn_scores[j]]
                    for emb in word_embeddings[j]:
                        temp_list.append(emb)
                    tabulated_list.append(temp_list)
                print (tabulate(tabulated_list, headers=headers))


        base_prop, attn_prop = quantify_attention(words, attention.detach().cpu().numpy(), block_ids)
        base_emb_norm, attn_emb_norm = quantify_norms(words, word_embeddings, block_ids)
        base_h_norm, attn_h_norm = quantify_norms(words, final_states, block_ids)

        test_base_prop += base_prop
        test_attn_prop += attn_prop

        test_base_emb_norm += base_emb_norm
        test_attn_emb_norm += attn_emb_norm

        test_base_h_norm += base_h_norm
        test_attn_h_norm += attn_h_norm

    print("iter %r: %s acc = %.2f" % (iter, name, 100.*test_correct/len(dataset)))
    print("iter %r: %s loss = %.8f" % (iter, name, total_loss/len(dataset)))

    '''
    outfile_name = "examples/" + TASK_NAME + "_" + MODEL_TYPE + "_hammer=" + str(C_HAMMER) \
         +"_kld=" + str(C_KLD) + "_seed=" + str(SEED) + "_iter=" +  str(iter) + ".pickle"
        
    pickle.dump(example_data, open(outfile_name, 'wb'))
    '''

    attn_mass = None

    if attn_stats:
        attn_mass = test_attn_prop/len(dataset)

        print("iter %r: in %s set base_ratio = %.8f, attention_ratio = %.14f" % (
            iter,
			name,
            test_base_prop/len(dataset),
            attn_mass))

        print("iter %r: in %s set base_emb_norm = %.4f, attn_emb_norm = %.4f" % (
            iter,
			name,
            test_base_emb_norm/len(dataset),
            test_attn_emb_norm/len(dataset)))

        print("iter %r: in %s set base_h_norm = %.4f, attn_h_norm = %.4f" % (
            iter,
			name,
            test_base_h_norm/len(dataset),
            test_attn_h_norm/len(dataset)))

    return test_correct/len(dataset), total_loss/len(dataset), attn_mass


def dump_attention_maps(dataset, filename, model):

    fw = open(filename, 'w')

    dataset = sorted(dataset, key=lambda x:x[0])
    for _ , words, _ , _, _ in dataset:
        words_t = torch.tensor([words]).type(tensor_type)
        _ , attn = model(words_t)
        attention = attn[0].detach().cpu().numpy()

        for att in attention:
            fw.write(str(att) + " ")
        fw.write("\n")
    fw.close()
    return


def run_classification_experiment(task_name, seed=1, c_entropy=0., c_hammer=0., c_kld=0., num_vis=5, num_epochs=5, block_top=3, block_words=None, emb_size=128, hid_size=64, epsilon=1e-12, to_anon=False, to_dump_attn=False, use_attn_file=False, use_block_file=False, model_type='emb-att', use_loss=False, debug=False, understand=False, flow=False, clip_vocab=False, vocab_size=20000):
    # print useful info
    log.pr_blue("Task: %s" %(task_name))
    log.pr_blue("Model: %s" %(model_type))
    log.pr_blue("Coef (hammer): %0.2f" %(c_hammer))
    log.pr_blue("Coef (random-entropy): %0.2f" %(c_entropy))

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    global w2i
    global w2c
    global t2i
    global UNK

    w2i = defaultdict(lambda: len(w2i))
    w2c = defaultdict(lambda: 0.0) # word to count
    t2i = defaultdict(lambda: len(t2i))
    UNK = w2i["<unk>"]

    """" Reading the data """
    directory = os.path.dirname(os.path.realpath(__file__))
    prefix = directory + "/data/" + task_name + "/"

    if use_block_file:
        log.pr_blue("Using block file")
        train = list(read_dataset(prefix+"train.txt",
                    block_file=prefix + "train.txt.block", clip_vocab=clip_vocab, vocab_size=vocab_size, to_anon=to_anon))
        w2i = defaultdict(lambda: UNK, w2i)
        nwords = len(w2i) if not clip_vocab else vocab_size
        t2i = defaultdict(lambda: UNK, t2i)

        dev = list(read_dataset(prefix+"dev.txt",
                    block_file=prefix + "dev.txt.block", vocab_size=vocab_size, to_anon=to_anon))
        test = list(read_dataset(prefix+"test.txt",
                    block_file=prefix + "test.txt.block", vocab_size=vocab_size, to_anon=to_anon))
    elif use_attn_file:
        log.pr_blue("Using attn file")
        train = list(read_dataset(prefix+"train.txt", block_words=block_words,
                    attn_file=prefix + "train.txt.attn." + model_type, clip_vocab=clip_vocab, vocab_size=vocab_size, to_anon=to_anon))
        w2i = defaultdict(lambda: UNK, w2i)
        nwords = len(w2i) if not clip_vocab else vocab_size
        t2i = defaultdict(lambda: UNK, t2i)

        dev = list(read_dataset(prefix+"dev.txt", block_words=block-words,
                    attn_file=prefix + "dev.txt.attn." + model_type, vocab_size=vocab_size, to_anon=to_anon))
        test = list(read_dataset(prefix+"test.txt", block_words=block_words,
                    attn_file=prefix + "test.txt.attn." + model_type, vocab_size=vocab_size, to_anon=to_anon))
    else:
        if block_words is None:
            log.pr_blue("Vanilla case: no attention manipulation")
        else:
            log.pr_blue("Using block words")

        train = list(read_dataset(prefix+"train.txt", block_words=block_words, clip_vocab=clip_vocab, vocab_size=vocab_size, to_anon=to_anon))
        nwords = len(w2i) if not clip_vocab else vocab_size
        w2i = defaultdict(lambda: UNK, w2i)
        t2i = defaultdict(lambda: UNK, t2i)

        dev = list(read_dataset(prefix+"dev.txt", block_words=block_words, vocab_size=vocab_size, to_anon=to_anon))
        test = list(read_dataset(prefix+"test.txt", block_words=block_words, vocab_size=vocab_size, to_anon=to_anon))


    if debug:
        train = train[:100]
        dev = dev[:100]
        test = test[:100]

    global i2w
    global i2t
    # Create reverse dicts
    i2w = {v: k for k, v in w2i.items()}
    i2w[UNK] = "<unk>"
    i2t = {v: k for k, v in t2i.items()}


    ntags = len(t2i)

    log.pr_cyan("The vocabulary size is %d" %(nwords))

    if model_type == 'emb-att':
        model = EmbAttModel(nwords, emb_size, ntags)
    elif model_type == 'emb-lstm-att':
        model = BiLSTMAttModel(nwords, emb_size, hid_size, ntags)
    elif model_type == 'no-att-only-lstm':
        model = BiLSTMModel(nwords, emb_size, hid_size, ntags)
    else:
        raise ValueError("model type not compatible")

    global calc_ce_loss
    calc_ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    global tensor_type
    global float_type

    tensor_type = torch.LongTensor
    float_type = torch.FloatTensor
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        tensor_type = torch.cuda.LongTensor
        float_type = torch.cuda.FloatTensor
        model.cuda()

    print ("evaluating without any training ...")
    _, _, _ = evaluate(test, 0, model, name='test', attn_stats=True, num_vis=0, emb_size=emb_size, flow=flow, understand=understand, c_hammer=c_hammer, c_kld=c_kld, c_entropy=c_entropy)


    print ("starting to train")


    best_dev_accuracy  = 0.
    best_dev_loss = np.inf
    best_test_accuracy = 0.
    test_am = None
    best_epoch = 0

    for ITER in range(1, num_epochs+1):
        random.shuffle(train)
        train_loss = 0.0
        train_ce_loss = 0.0
        train_entropy_loss = 0.0
        train_hammer_loss = 0.0
        train_kld_loss = 0.0

        start = time.time()
        for num, (idx, words_orig, block_ids, attn_orig, tag) in enumerate(train):

            words = torch.tensor([words_orig]).type(tensor_type)
            tag = torch.tensor([tag]).type(tensor_type)
            if attn_orig is not None:
                attn_orig = torch.tensor(attn_orig).type(float_type)

            # forward pass
            out, attns = model(words)
            attention = attns[0]

            ce_loss = calc_ce_loss(out, tag)
            entropy_loss = calc_entropy_loss(attention, c_entropy)
            hammer_loss = calc_hammer_loss(words_orig, attention,
                                            block_ids, c_hammer)

            kld_loss = calc_kld_loss(attention, attn_orig, c_kld)

            loss = ce_loss + entropy_loss + hammer_loss + kld_loss
            train_loss += loss.item()

            train_ce_loss += ce_loss.item()
            train_entropy_loss += entropy_loss.item()
            train_hammer_loss += hammer_loss.item()
            train_kld_loss += kld_loss.item()

            print ("ID: %4d\t CE: %0.4f\t ENTROPY: %0.4f\t HAMMER: %0.4f\t KLD: %.4f\t TOTAL: %0.4f" %(
                num,
                ce_loss.item(),
                entropy_loss.item(),
                hammer_loss.item(),
                kld_loss.item(),
                loss.item()
            ), end='\r')

            # update the params
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("iter %r: train loss=%.4f, ce_loss=%.4f, entropy_loss=%.4f,"
                    "hammer_loss=%.4f, kld_loss==%.4f, time=%.2fs" % (
                    ITER,
                    train_loss/len(train),
                    train_ce_loss/len(train),
                    train_entropy_loss/len(train),
                    train_hammer_loss/len(train),
                    train_kld_loss/len(train),
                    time.time()-start))

        _, _, _  = evaluate(train, ITER, model, name='train', emb_size=emb_size, flow=flow, understand=understand, c_hammer=c_hammer, c_kld=c_kld, c_entropy=c_entropy)
        dev_acc, dev_loss, dev_am  = evaluate(dev, ITER, model, name='dev', attn_stats=True, emb_size=emb_size, flow=flow, understand=understand, c_hammer=c_hammer, c_kld=c_kld, c_entropy=c_entropy)
        test_acc, test_loss, current_test_am = evaluate(test, ITER, model, name='test', attn_stats=True,
                            num_vis=num_vis, emb_size=emb_size, flow=flow, understand=understand, c_hammer=c_hammer, c_kld=c_kld, c_entropy=c_entropy)

        if ((not use_loss) and dev_acc > best_dev_accuracy) or (use_loss and dev_loss < best_dev_loss):

            if use_loss:
                best_dev_loss = dev_loss
            else:
                best_dev_accuracy = dev_acc
            best_test_accuracy = test_acc
            test_am = current_test_am
            best_epoch = ITER

            if to_dump_attn:
                log.pr_bmagenta("dumping attention maps")
                dump_attention_maps(train, prefix + "train.txt.attn." + model_type, model)
                dump_attention_maps(dev, prefix + "dev.txt.attn." + model_type, model)
                dump_attention_maps(test, prefix + "test.txt.attn." + model_type, model)


        print ("iter %r: best test accuracy = %.4f attained after epoch = %d" %(
            ITER, best_test_accuracy, best_epoch))

    return best_test_accuracy, test_am


if __name__ == '__main__':
    # parsing stuff from the command line
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--emb-size', dest='emb_size', type=int, default=128,
            help = 'number of dimensions for the embedding layer')

    parser.add_argument('--hid-size', dest='hid_size', type=int, default=64,
            help = 'size of the hidden dimension')

    parser.add_argument('--model', dest='model', default='emb-att',
            choices=('emb-att', 'emb-lstm-att', 'no-att-only-lstm'),
            help = 'select the model you want to run')

    parser.add_argument('--task', dest='task', default='pronoun',
            choices=('pronoun', 'sst', 'sst-wiki', 'sst-wiki-unshuff', 'reco', 'reco-rank', 'de-pronoun', 'de-refs', 'de-sst-wiki', 'occupation-classification', 'de-occupation-classification', 'occupation-classification_all'),
            help = 'select the task you want to run on')

    parser.add_argument('--num-epochs', dest='num_epochs', type=int, default=5,
            help = 'number of epochs')

    parser.add_argument('--num-visualize', dest='num_vis', type=int, default=5,
            help = 'number of examples to visualize')

    parser.add_argument('--loss-entropy', dest='loss_entropy', type=float, default=0.,
            help = 'strength for entropy loss on attention weights')

    parser.add_argument('--loss-hammer', dest='loss_hammer', type=float, default=0.,
            help = 'strength for hammer loss on attention weights')

    parser.add_argument('--loss-kld', dest='loss_kld', type=float, default=0.,
            help = 'strength for KL Divergence Loss on attention weights')

    parser.add_argument('--top', dest='top', type=int, default=3,
            help = 'how many of the most attended words to ignore (default is 3)')

    parser.add_argument('--seed', dest='seed', type=int, default=1,
            help = 'set random seed, defualt = 1')

    # flags specifying whether to use the block and attn file or not
    parser.add_argument('--use-attn-file', dest='use_attn_file', action='store_true')

    parser.add_argument('--use-block-file', dest='use_block_file', action='store_true')

    parser.add_argument('--block-words', dest='block_words', nargs='+', default=None,
            help = 'list of words you wish to block (default is None)')

    parser.add_argument('--dump-attn', dest='dump_attn', action='store_true')

    parser.add_argument('--use-loss', dest='use_loss', action='store_true')

    parser.add_argument('--anon', dest='anon', action='store_true')

    parser.add_argument('--debug', dest='debug', action='store_true')

    parser.add_argument('--understand', dest='understand', action='store_true')

    parser.add_argument('--flow', dest='flow', action='store_true')

    parser.add_argument('--clip-vocab', dest='clip_vocab', action='store_true')

    parser.add_argument('--vocab-size', dest='vocab_size', type=int, default=20000,
            help='in case you clip vocab, specify the vocab size')

    params = vars(parser.parse_args())

    run_classification_experiment(
        task_name = params['task'],
        seed = params['seed'],
        c_entropy = params['loss_entropy'],
        c_hammer = params['loss_hammer'],
        c_kld = params['loss_kld'],
        num_vis = params['num_vis'],
        num_epochs = params['num_epochs'],
        emb_size = params['emb_size'],
        hid_size = params['hid_size'],
        epsilon = 1e-12,
        to_anon = params['anon'],
        to_dump_attn = params['dump_attn'],
        block_top = params['top'],
        block_words = params['block_words'],
        use_attn_file = params['use_attn_file'],
        use_block_file = params['use_block_file'],
        model_type = params['model'],
        use_loss = params['use_loss'],
        debug = params['debug'],
        understand = params['understand'],
        flow = params['flow'],
        clip_vocab = params['clip_vocab'],
        vocab_size = params['vocab_size']
    )
