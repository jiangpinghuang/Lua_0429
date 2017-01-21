#!/usr/bin/env python

import os
import sys
import h5py
import argparse
import itertools
import numpy as np
from collections import defaultdict

class Indexer:
    def __init__(self, symbols=["<blank>", "<unk>", "<s>", "</s>"]):
        self.vocab = defaultdict(int)
        self.PAD = symbols[0]
        self.UNK = symbols[1]
        self.BOS = symbols[2]
        self.EOS = symbols[3]
        self.d = {self.PAD: 1, self.UNK: 2, self.BOS: 3, self.EOS: 4}
        
    def add_word(self, ws):
        for w in ws:
            if w not in self.d:
                self.d[w] = len(self.d) + 1
                
    def convert(self, w):
        return self.d[w] if w in self.d else self.d[self.UNK]
    
    def convert_sequence(self, ls):
        return [self.convert(l) for l in ls]
    
    def clean(self, s):
        s = s.replace(self.PAD, "")
        s = s.replace(self.BOS, "")
        s = s.replace(self.EOS, "")
        return s
    
    def write(self, outfile, char=0):
        out = open(outfile, "w")
        items = [(v, k) for k, v in self.d.iteritems()]
        items.sort()
        for v, k in items:
            if char == 1:
                print >> out, k.encode('utf-8'), v
            else:
                print >> out, k, v 
        out.close()
        
    def prune_vocab(self, k):
        vocab_list = [(word, count) for word, count in self.vocab.iteritems()]
        vocab_list.sort(key = lambda x: x[1], reverse = True)
        k = min(k, len(vocab_list))
        self.pruned_vocab = {pair[0]:pair[1] for pair in vocab_list[:k]}
        for word in self.pruned_vocab:
            if word not in self.d:
                self.d[word] = len(self.d) + 1
                
    def load_vocab(self, vocab_file, char=0):
        self.d = {}
        for line in open(vocab_file, 'r'):
            if char == 1:
                v, k = line.decode("utf-8").strip().split()
            else:
                v, k = line.strip().split()
            self.d[v] = int(k)
            
def pad(ls, length, symbol):
    if len(ls) >= length:
        return ls[:length]
    return ls + [symbol] * (length - len(ls))
        
def save_feature(name, indexer, outputfile):
    if len(indexer) > 0:
        print("Number of additional features on {} side: {}".format(name, len(indexer)))
    for i in range(len(indexer)):
        indexer[i].write(outputfile + "." + name + "_feature_" + str(i + 1) + ".dict", )
        print(" * {} feature {} of size: {}".format(name, i + 1, len(indexer[i].d)))
            
def load_feature(name, indexer, outputfile):
    for i in range(len(indexer)):
        indexer[i].load_vocab(outputfile + "." + name + "_feature_" + str(i + 1) + ".dict", )
        print(" * {} feature {} of size: {}".format(name, i + 1, len(indexer[i].d)))
            
def get_data(args):
    src_indexer = Indexer(["<blank>", "<unk>", "<s>", "</s>"])
    src_feature_indexer = []
    tar_indexer = Indexer(["<blank>", "<unk>", "<s>", "</s>"])
    char_indexer = Indexer(["<blank>", "<unk>", "{", "}"])
    char_indexer.add_w([src_indexer.PAD, src_indexer.UNK, src_indexer.BOS, src_indexer.EOS])

    def init_feature_indexer(indexer, count):
        for i in range(count):
            indexer.append(Indexer(["<blank>", "<unk>", "<s>", "</s>"]))
                
    def load_sentence(sent, indexer):
        sent_seq = sent.strip().split()
        sent_word = ''
        sent_feature = []
            
        for entry in sent_seq:
            fields = entry.split('-|-')
            word = fields[0]
            sent_word += (' ' if sent_word else '') + word
                
            if len(fields) > 1:
                count = len(fields) - 1
                if len(sent_feature) == 0:
                    sent_feature = [ [] for i in range(count) ]
                if len(indexer) == 0:
                    init_feature_indexers(indexer, count)
                for i in range(1, len(fields)):
                    sent_feature[i - 1].append(fields[i])
        return sent_word, sent_feature
        
    def add_feature_vocab(feature, indexer):
        if len(indexer) > 0:
            index = 0
            for value in feature:
                indexer[index].add_word(value)
                index += 1
        
    def make_vocab(srcfile, tarfile, seq_len, max_word_len=0, char=0, train=1):
        num_sent = 0
        for _, (src, tar) in enumerate(itertools.izip(open(srcfile, 'r'), open(tarfile, 'r'))):
            src, src_feature = load_sentence(src, src_feature_indexer)
            if char == 1:
                src = src_indexer.clean(src.decode("utf-8").strip())
                tar = tar_indexer.clean(tar.decode("utf-8").strip())
            else:
                src = src_indexer.clean(src.strip())
                tar = tar_indexer.clean(tar.strip())
            tar = tar.strip().split()
            src = src.strip().split()
            if len(tar) > seq_len or len(src) > seq_len or len(tar) < 1 or len(src) < 1:
                continue
            num_sent += 1
            if train == 1:
                for w in tar:
                    if char == 1:
                        w = char_indexer.clean(w)
                        if len(w) == 0:
                            continue
                        max_word_len = max(len(w) + 2, max_word_len)
                        for c in list(w):
                            char_indexer.vocab[c] += 1
                    tar_indexer.vocab[w] += 1
                    
                add_feature_vocab(src_feature, src_feature_indexer)
                    
                for w in src:
                    if char == 1:
                        w = char_indexer.clean(w)
                        if len(w) == 0:
                            continue
                        max_word_len = max(len(w) + 2, max_word_len)
                        for c in list(w):
                            char_indexer.vocab[c] += 1
                    src_indexer.vocab[w] += 1
        return max_word_len, num_sent
        
    def convert(srcfile, tarfile, alifile, batchsize, seq_len, outfile, num_sent, max_word_len,
                max_sent_len=0, char=0, unkfilter=0, shuffle=0):
            
        def init_feature_tensor(indexer):
            return [np.zeros((num_sent, new_seq_len), dtype=int) for i in range(len(indexer))]
            
        def load_feature(feature, indexer, seq_len):
            if len(feature) == 0:
                return None
                
            features = []
            for i in range(len(feature)):
                features.append([[indexer[i].BOS]] + feature[i] + [[indexer[i].EOS]]) 
                    
            for i in range(len(features)):
                features[i] = pad(features[i], seq_len, [indexer[i].PAD])
                for j in range(len(features[i])):
                    features[i][j] = indexer[i].convert_sequence(features[i][j])[0]
                features[i] = np.array(features[i], dtype=int)
            return features
            
        new_seq_len = seq_len + 2
            
        alignfile_hdl = None
        alignments = None
        if not alignfile == '':
            alignfile_hdl = open(alignfile, 'r')
            alignments = np.zeros((num_sents, newseqlength, newseqlength), dtype=np.uint8)
                    
        targets = np.zeros((num_sents, newseqlength), dtype=int)
        target_output = np.zeros((num_sents, newseqlength), dtype=int)
        sources = np.zeros((num_sents, newseqlength), dtype=int)
        sources_features = init_features_tensor(src_feature_indexers)
        source_lengths = np.zeros((num_sents,), dtype=int)
        target_lengths = np.zeros((num_sents,), dtype=int)
        if chars == 1:
            sources_char = np.zeros((num_sents, newseqlength, max_word_l), dtype=int)
            targets_char = np.zeros((num_sents, newseqlength, max_word_l), dtype=int)
        dropped = 0
        sent_id = 0
        for _, (src_orig, targ_orig) in enumerate(itertools.izip(open(srcfile, 'r'), open(tarfile, 'r'))):
            src_orig, src_orig_features = load_sentence(src_orig, src_feature_indexers)
            if chars == 1:
                src_orig = src_indexer.clean(src_orig.decode("utf-8").strip())
                targ_orig = target_indexer.clean(targ_orig.decode("utf-8").strip())
            else:
                src_orig = src_indexer.clean(src_orig.strip())
                targ_orig = target_indexer.clean(targ_orig.strip())
            targ = [target_indexer.BOS] + targ_orig.strip().split() + [target_indexer.EOS]
            src = [src_indexer.BOS] + targ_orig.strip().split() + [src_indexer.EOS]
            max_sent_l = max(len(targ), len(src), max_sent_l)
                
            align = []
            if alignfile_hdl:
                align = alignfile_hdl.readline().strip().split(" ")
                    
            if len(targ) > newseqlength or len(src) > newseqlength or len(targ) < 3 or len(src) < 3:
                dropped += 1
                continue
            targ = pad(targ, newseqlength + 1, target_indexer.PAD)
            targ_char = []
            for word in targ:
                if chars == 1:
                    word = char_indexer.clean(word)
                    
                word = word if word in target_indexer.d else target_indexer.UNK
                if chars == 1:
                    char = [char_indexer.BOS] + list[word] + [char_indexer.EOS]
                    if len(char) > max_word_l:
                        char = char[:max_word_l]
                        char[-1] = char_indexer.EOS
                    char_idx = char_indexer.convert_sequence(pad(char, max_word_l, char_indexer.PAD))
                    targ_char.append(char_idx)
            targ = target_indexer.convert_sequence(targ)
            targ = np.array(targ, dtype=int)
                
            src = pad(src, newseqlength, src_indexer.PAD)
            src_char = []
            for word in src:
                if chars == 1:
                    word = char_indexer.clean(word)
                    char = [char_indexer.BOS] + list(word) + [char_indexer.EOS]
                    if len(char) > max_word_l:
                        char = char[:max_word_l]
                        char[-1] = char_indexer.EOS
                    char_idx = char_indexer.convert_sequence(pad(char, max_word_l, char_indexer.PAD))
                    src_char.append(char_idx)
            src = src_indexer.convert_sequence(src)
            src = np.array(src, dtype=int)
                
            if unkfilter > 0:
                targ_unks = float((targ[:-1] == 2).sum())
                src_unks = float((src == 2).sum())
                if unkfilter < 1:
                    targ_unks = targ_unks / (len(targ[:-1]) - 2)
                    src_unks = src_unks / (len(src) - 2)
                if targ_unks > unkfilter or src_unks > unkfilter:
                    dropped += 1
                    continue
                    
            targets[sent_id] = np.array(targ[:-1], dtype=int)
            target_lengths[sent_id] = (targets[sent_id] != 1).sum()
            if chars == 1:
                targets_char[sent_id] = np.array(targ_char[:-1], dtype=int)
            target_output[sent_id] = np.array(targ[1:], dtype=int)
            sources[sent_id] = np.array(src, dtype=int)
            source_lengths[sent_id] = (sources[sent_id] != 1).sum()
            if chars == 1:
                sources_char[sent_id] = np.array(src_char, dtype=int)
                    
            sources_features = load_features(src_orig_features, src_feature_indexers, newseqlength)
            for i in range(len(src_feature_indexers)):
                sources_features[i][sent_id] = np.array(sources_features[i], dtype=int)
                    
            if alignfile_hdl:
                for pair in align:
                    aFrom, aTo = pair.split('-')
                    alignments[sent_id][int(aFrom) + 1][int(aTo) + 1] = 1
                      
            sent_id += 1
            if sent_id % 100000 == 0:
                print("{}/{} sentences processed".format(sent_id, num_sents))
                    
        print(sent_id, num_sents)
        if shuffle == 1:
            rand_idx = np.random.permutation(sent_id)
            targets = targets[rand_idx]
            target_output = target_output[rand_idx]
            source_lengths = source_lengths[rand_idx]
            target_lengths = target_lengths[rand_idx]
            for i in range(len(sources_features)):
                sources_features[i] = sources_features[i][rand_idx]
            if chars == 1:
                sources_char = sources_char[rand_idx]
                targets_char = targets_char[rand_idx]
            
        source_lengths = source_lengths[:sent_id]
        source_sort = np.argsort(source_lengths)
            
        sources = sources[source_sort]
        targets = targets[source_sort]
        target_output = target_output[source_sort]
        if alignments is not None:
            alignments = alignments[source_sort]
        target_l = target_lengths[source_sort]
        source_l = source_lengths[source_sort] 
            
        for i in range(len(src_feature_indexers)):
            sources_features[i] = sources_features[i][source_sort]
                
        curr_l = 0
        l_location = []
            
        for j, i in enumerate(source_sort):
            if source_l[i] > curr_l:
                curr_l = source_lengths[i]
                l_location.append(j + 1)
        l_location.append(len(sources) + 1)
            
        curr_idx = 1
        batch_idx = [1]
        nonzeros = []
        batch_l = []
        batch_w = []
        target_l_max = []
        for i in range(len(l_location) - 1):
            while curr_idx < l_location[i + 1]:
                curr_idx = min(curr_idx + batchsize, l_location[i + 1])
                batch_idx.append(curr_idx)
        for i in range(len(batch_idx) - 1):
            batch_l.append(batch_idx[i + 1] - batch_idx[i])
            batch_w.append(source_l[batch_idx[i] - 1])
            nonzeros.append((target_output[batch_idx[i] - 1:batch_idx[i + 1] - 1] != 1).sum().sum())
            target_l_max.append(max(target_l[batch_idx[i] - 1:batch_idx[i + 1] - 1]))
                
        f = h5py.File(outfile, "w")
            
        f["source"] = sources
        f["target"] = targets
        f["target_output"] = target_output
        if alignments is not None:
            print "build alignment structure"
            alignment_cc_val = []
            alignment_cc_colidx = []
            alignment_cc_sentidx = []
            S = {}
            for k in range(sent_idx - 1):
                alignment_cc_sentidx.append(len(alignment_cc_colidx))
                for i in xrange(0, source_l[k]):
                    a = ''
                    maxnalign = 0
                    for j in xrange(0, newseqlength):
                        a = a + chr(ord('0') + int(alignments[k][i][j]))
                    if not a in S:
                        alignment_cc_colidx.append(len(alignment_cc_val))
                        S[a] = len(alignment_cc_val)
                        for j in xrange(0, newseqlength):
                            alignment_cc_val.append(alignments[k][i][j])
                    else:
                        alignment_cc_colidx.append(S[a])
                
            assert(len(alignment_cc_colidx) < 4294967296)
            f["alignment_cc_sentidx"] = np.array(alignment_cc_sentidx, dtype=np.uint32)
            f["alignment_cc_colidx"] = np.array(alignment_cc_colidx, dtype=np.uint32)
            f["alignment_cc_val"] = np.array(alignment_cc_val, dtype=np.uint8)
                
        f["target_l"] = np.array(target_l_max, dtype=int)
        f["target_l_all"] = target_l
        f["batch_l"] = np.array(batch_l, dtype=int)
        f["batch_w"] = np.array(batch_w, dtype=int)
        f["batch_idx"] = np.array(batch_w, dtype=int)
        f["target_nonzeros"] = np.array(nonzeros, dtype=int)
        f["source_size"] = np.array([len(src_indexer.d)])
        f["target_size"] = np.array([len(target_indexer.d)])
        f["num_source_features"] = np.array([len(src_feature_indexers)])
        for i in range(len(src_feature_indexers)):
            f["source_feature_" + str(i + 1)] = sources_features[i]
            f["source_feature_" + str(i + 1) + "_size"] = np.array([len(src_feature_indexers[i].d)])
        if chars == 1:
            del sources, targets, target_output
            sources_char = sources_char[source_sort]
            f["target_char"] = targets_char
            f["char_size"] = np.array([len(char_indexer.d)])
        print("Saved {} sentences (dropped {} due to length/unk filter)".format(len(f["source"]), dropped))
        f.close()
        return max_sent_l
        
    print("First pass through data to get vocab...")
    max_word_l, num_sents_train = make_vocab(args.srcfile, args.targetfile, args.seqlength, 0, args.chars)
    print("Number of sentences in training: {}".format(num_sents_train))
    max_word_l, num_sents_valid = make_vocab(args.srcvalfile, args.targetvalfile, args.seqlength, max_word_l, args.chars, 0)
    print("Number of sentences in valid: {}".format(num_sents_valid))
    if args.chars == 1:
        print("Max word length (before cutting): {}".format(max_word_l))
            
    src_indexer.prune_vocab(args.srcvocabsize)
    target_indexer.prune_vocab(args.targetvocabsize)
    if args.srcvocabfile != '':
        print('Loading pre-specified source vocab from ' + args.srcvocabfile)
        src_indexer.load_vocab(args.srcvocabfile, args.chars)
    if args.targetvocabfile != '':
        print('Loading pre-specified target vocab from ' + args.targetvocabfile)
        target_indexer.load_vocab(args.targetvocabfile, args.chars)
    if args.charvocabfile != '':
        print('Loading pre-specified char vocab from ' + args.charvocabfile)
        char_indexer.load_vocab(args.charvocabfile, args.chars)
            
    src_indexer.write(args.outputfile + ".src.dict", args.chars)
    target_indexer.write(args.outputfile + ".targ.dict", args.chars)
    if args.chars == 1:
        if args.charvocabfile == '':
            char_indexer.prune_vocab(500)
        char_indexer.write(args.outputfile + ".char.dict", args.chars)
        print("Character vocab size: {}".format(len(char_indexer.vocab)))
            
    if args.reusefeaturefile != '':
        load_features('source', src_feature_indexers, args.reusefeaturefile)
            
    save_features('source', src_feature_indexers, args.outputfile)
        
    print("Source vocab size: Original = {}, Pruned = {}".format(len(src_indexer.vocab), len(src_indexer.d)))
    print("Target vocab size: Original = {}, Pruned = {}".format(len(target_indexer.vocab), len(target_indexer.d)))
        
    max_sent_l = 0
    max_sent_l = convert(args.srcvalfile, args.targetvalfile, args.alignvalfile, args.batchsize, args.seqlength,
                        args.outputfile + "-val.hdf5", num_sents_valid, max_word_l, max_sent_l, args.chars,
                        args.unkfilter, args.shuffle)
    max_sent_l = convert(args.srcfile, args.targetfile, args.alignfile, args.batchsize, args.seqlength,
                        args.outputfile + "-train.hdf5", num_sents_train, max_word_l, max_sent_l, args.chars,
                        args.unkfilter, args.shuffle)
        
    print("Max sent length (before dropping): {}".format(max_sent_l))
        
def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--srcvocabsize', help="Size of source vocabulary, constructed by taking the top X most frequent words. "
                        "Rest are replaced with special UNK tokens.", type=int, default=50000)
    parser.add_argument('--targetvocabsize', help="Size of target vocabulary, constructed by taking the top X most frequent words. "
                        "Rest are replaced with special UNK tokens.", type=int, default=50000)
    parser.add_argument('--srcfile', help="Path to source training data, where each line represents a single source/target sequence.",
                        required=True)
    parser.add_argument('--targetfile', help="Path to target training data, where each line represents a single source/target sequence",
                        required=True)
    parser.add_argument('--srcvalfile', help="Path to source validation data.", required=True)
    parser.add_argument('--targetvalfile', help="Path to target validation data.", required=True)
    parser.add_argument('--batchsize', help="Size of each minibatch.", type=int, default=64)
    parser.add_argument('--seqlength', help="Maximum sequence length. Sequences longer than this are dropped.", type=int, default=50)
    parser.add_argument('--outputfile', help="Prefix of the output file names.", type=str, required=True)
    parser.add_argument('--maxwordlength', help="For the character models, words are (if longer than maxwordlength) or zero-padded " 
                        "(if shorter) to maxwordlength", type=int, default=35)
    parser.add_argument('--chars', help="If 1, construct the character-level dataset as well. This might take up a lot of space depending "
                        "on your data size, so you may want to break up the training data into different words.", type=int, default=0)
    parser.add_argument('--srcvocabfile', help="If working with a present vocab, then including this will ignore srcvocabsize and use the "
                        "vocab provided here.", type=str, default='')
    parser.add_argument('--targetvocabfile', help="If working with a present vocab, then including this will ignore targetvocabsize and "
                        "use the vocab provided here.", type=str, default='')
    parser.add_argument('--charvocabfile', help="If working with a present vocab, then including this use the char vocab provided here.",
                        type=str, default='')
    parser.add_argument('--unkfilter', help="Ignore sentences with too many UNK tokens. Can be an absolute count limit (if > 1) or a "
                        "proportional limit (0 < unkfilter < 1).", type=float, default=0)
    parser.add_argument('--reusefeaturefile', help="Use existing features vocabs", type=str, default='')
    parser.add_argument('--shuffle', help="If = 1, shuffle sentences before sorting (based on source length).", type=int, default=0)
    parser.add_argument('--alignfile', help="Path to source-to-target alignment of training data, where each line represents a set of alignments "
                        "per train instance.", type=str, required=False, default='')
    parser.add_argument('--alignvalfile', help="Path to source-to-target alignment of validation data", type=str, required=False, default='')
    args = parser.parse_args(arguments)
    get_data(args)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
