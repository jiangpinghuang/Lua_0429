require('nn')
require('nngraph')
require('hdf5')

require 'util.data'
require 'util.model'
require 'util.util'

cmd = torch.CmdLine()

cmd:text("")
cmd:text("**data options**")
cmd:text("")
cmd:option('-data_file', 'data/demo-train.hdf5', [[path to the training *hdf5 file.]])
cmd:option('-valid_file', 'data/demo-valid.hdf5', [[path to the validation *hdf5 file.]])
cmd:option('-save_file', 'stsa', [[saved file name as savefile_epochX_PPL.t7.]])
cmd:option('-num_shard', 0, [[if the training data has been broken up into shards.]])
cmd:option('-train_from', '', [[if training from a checkpoint then this is the model.]])

cmd:text("")
cmd:text("**model options**")
cmd:text("")
cmd:option('-num_layer', 2, [[number of layers in the lstm encoder or decoder.]])
cmd:option('-rnn_size', 500, [[size of lstm hidden states.]])
cmd:option('-word_vec_size', 500, [[word embedding sizes.]])
cmd:option('-attn', 1, [[if = 1, use attention. if = 0, use the last hidden state of the decoder.]])
cmd:option('-brnn', 0, [[if = 1, use a bidirectional rnn.]])
cmd:option('-use_char_enc', 0, [[if = 1, use character on the encoder side.]])
cmd:option('-use_char_dec', 0, [[if = 1, use character on the decoder side.]])
cmd:option('-reverse_src', 0, [[if = 1, reverse the source sequence.]])

