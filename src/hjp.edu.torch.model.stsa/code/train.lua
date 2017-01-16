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
cmd:option('-init_dec', 1, [[initialize the hidden state of the decoder.]])
cmd:option('-input_feed', 1, [[if = 1, feed the context vector at each time step.]])
cmd:option('-multi_attn', 0, [[if > 1, then use a another attention layer.]])
cmd:option('-res_net', 0, [[use residual connections between lstm stacks.]])
cmd:option('-guided_align', 0, [[if = 1, use external alignments to guide the attention weights.]])
cmd:option('-guided_align_weight', 0.5, [[default weights for external alignments.]])
cmd:option('-guided_align_decay', 1, [[decay rate per epoch for alignment weight.]])

cmd:text("")
cmd:text("**character model**")
cmd:text("")
cmd:option('-char_vec_size', 25, [[size of the character embeddings.]])
cmd:option('-kernel_width', 6, [[size of the convolutional filter.]])
cmd:option('-num_highway_layer', 2, [[number of highway layer.]])

cmd:text("")
cmd:text("**optimization options**")
cmd:text("")
cmd:option('-epochs', 13, [[number of training epochs.]])
cmd:option('-start_epoch', 1, [[which epoch to start from checkpoint.]])
cmd:option('-init_param', 0.1, [[initialized parameters over uniform distribution.]])
cmd:option('-optim', 'sgd', [[optimization method option: sgd, adagrad, adadelta, adam.]])
cmd:option('-learn_rate', 1, [[starting learning rate.]])
cmd:option('-layer_lr', '', [[comma-separated learning rate.]])
cmd:option('-max_grad_norm', 5, [[have the norm equal to max_grad_norm.]])
cmd:option('-dropout', 0.3, [[dropout probability.]])
cmd:option('-lr_decay', 0.5, [[decay learning rate.]])
cmd:option('-start_decay', 9, [[start decay after this epoch.]])
cmd:option('-curriculum', 0, [[order the mini-batch based on source sequence length.]])
cmd:option('-feature_emb_dim_exp', 0.7, [[embedding dimension will be set to n^exponent.]])
cmd:option('-pre_word_vec_enc', '', [[load pre-trained word embedding on the encoder side.]])
cmd:option('-pre_word_vec_dec', '', [[load pre-trained word embedding on the decoder side.]])
cmd:option('-fix_word_vec_enc', 0, [[if = 1, fix word embedding on the encoder side.]])
cmd:option('-fix_word_vec_dec', 0, [[if = 1, fix word embedding on the decoder side.]])
cmd:option('-max_batch', '', [[infer the max batch size from validation data.]])

cmd:text("")
cmd:text("**other options**")
cmd:text("")
cmd:option('-start_symbol', 0, [[use special start-of-sentence and end-of-sentence tokens.]])
cmd:option('-gpuid', -1, [[which gpu to use. use cpu = -1.]])
cmd:option('-gpuids', -1, [[if >= 0, model will use two gpus.]])
cmd:option('-cudnn', 0, [[whether to use cudnn or not for convolutions.]])
cmd:option('-save_every', 1, [[save every this may epoch.]])
cmd:option('-print_every', 50, [[print states after this many batches.]])
cmd:option('-seed', 3456, [[seed for random initialization.]])
cmd:option('-prealloc', 1, [[use memory preallocation.]])

function zero_table(t)
  for i = 1, #t do
    if opt.gpuid >= 0 and opt.gpuids >= 0 then
      if i == 1 then 
        cutorch.setDevice(opt.gpuid)
      else
        cutorch.setDevice(opt.gpuids)
      end
    end
    t[i]:zero()
  end
end

function append_table(dst, src)
  for i = 1, #src do
    table.insert(dst, src[i])
  end
end

function train(train_data, valid_data)
  local timer = torch.Timer()
  local num_param = 0
  local num_prunedparam = 0
  local start_decay = 0
  params, grad_params = {}, {}
  opt.train_perf = {}
  opt.valid_perf = {}
  
  for i = 1, #layers do
    if opt.gpuids >= 0 then
      if i == 1 then
        cutorch.setDevice(opt.gpuid)
      else
        cutorch.setDevice(opt.gpuids)
      end
    end
    local p, gp = layers[i]:getParameters()
    if opt.train_from:len() == 0 then
      p:uniform(-opt.init_param, opt.init_param)
    end
    num_param = num_param + p:size(1)
    param[i] = p
    grad_param[i] = gp
    layers[i]:apply(function(m) if m.nPruned then num_prunedparam = num_prunedparam + m:nPruned() end end)
  end
  
  if opt.pre_word_vec_enc:len() > 0 then
    local f = hdf5.open(opt.pre_word_vec_enc)
    local pre_word_vec = f:read('word_vec'):all()
    for i = 1, pre_word_vec:size(1) do
      word_vec_layer[1].weight[i]:copy(pre_word_vec[i])
    end
  end  
  if opt.pre_word_vec_dec:len() > 0 then
    local f = hdf5.open(opt.pre_word_vec_dec)
    local pre_word_vec = f:read('word_vec'):all()
    for i = 1, pre_word_vec:size(1) do
      word_vec_layer[2].weight[i]:copy(pre_word_vec[i])
    end
  end
  if opt.brnn == 1 then
    num_param = num_param - word_vec_layers[1].weight:nElement()
    word_vec_layer[3].weight:copy(word_vec_layer[1].weight)
    if opt.use_char_enc == 1 then
      for i = 1, charcnn_offset do
        num_param = num_param - charcnn_layer[i]:nElement()
        charcnn_layer[i + charcnn_offset]:copy(charcnn_layer[i])
      end
    end
  end
  
  print("number of parameters: " .. num_param .. " (active: " .. num_param - num_prunedparam .. ")")
  
  if opt.gpuid >= 0 and opt.gpuids >= 0 then
    cutorch.setDevice(opt.gpuid)
    word_vec_layer[1].weight[1]:zero()
    cutorch.setDevice(opt.gpuids)
    word_vec_layer[2].weight[1]:zero()
  else
    word_vec_layer[1].weight[1]:zero()
    word_vec_layer[2].weight[1]:zero()
    if opt.brnn == 1 then
      word_vec_layer[3].weight[1]:zero()
    end
  end
  
  encoder_grad_proto = torch.zeros(opt.max_batch_l, opt.max_sent_l, opt.rnn_size) 
  encoder_bwd_grad_proto = torch.zeros(opt.max_batch_l, opt.max_sent_l, opt.rnn_size)
  context_proto = torch.zeros(opt.max_batch_l, opt.max_sent_l, opt.rnn_size)
  if opt.gpuids >= 0 then
    encoder_grad_protos = torch.zeros(opt.max_batch_l, opt.max_sent_l, opt.rnn_size)
    context_protos = torch.zeros(opt.max_batch_l, opt.max_sent_l, opt.rnn_size)
    encoder_bwd_grad_protos = torch.zeros(opt.max_batch_l, opt.max_sent_l, opt.rnn_size)
  end
  
  encoder_clone = clone_many_times(encoder, opt.max_sent_l_src)
  decoder_clone = clone_many_times(decoder, opt.max_sent_l_targ)
  if opt.brnn == 1 then
    encoder_bwd_clone = clone_many_times(encoder_bwd, opt.max_sent_l_src)
  end
  for i = 1, opt.max_sent_l_src do
    if encoder_clone[i].apply then
      encoder_clone[i]:apply(function(m) m:setReuse() end)
      if opt.prealloc == 1 then encoder_clone[i]:apply(function(m) m:setPrealloc() end) end
    end
    if opt.brnn == 1 then
      encoder_bwd_clone[i]:apply(function(m) m:setReuse() end)
      if opt.prealloc == 1 then encoder_bwd_clone[i]:apply(function(m) m:setPrealloc() end) end
    end
  end
  for i = 1, opt.max_sent_l_targ do
    if decoder_clone[i].apply then
      decoder_clone[i]:apply(function(m) m:setReuse() end)
      if opt.prealloc == 1 then decoder_clone[i]:apply(function(m) m:setPrealloc() end) end
    end
  end
  
  local attn_init = torch.zeros(opt.max_batch_l, opt.max_sent_l)
  local hidden_init = torch.zeros(opt.max_batch_l, opt.rnn_size)
  if opt.gupid >= 0 then
    attn_init = attn_init:cuda()
    hidden_init = hidden_init:cuda()
    cutorch.setDevice(opt.gpuid)
    if opt.gpuids >= 0 then
      encoder_grad_protos = encoder_grad_protos:cuda()
      encoder_bwd_grad_protos = encoder_bwd_grad_protos:cuda()
      context_proto = context_proto:cuda()
      cutorch.setDevice(opt.gpuids)
      encoder_grad_proto = encoder_grad_proto:cuda()
      encoder_bwd_grad_proto = encoder_bwd_grad_proto:cuda()
      context_protos = context_protos:cuda()
      cutorch.setDevice(opt.gpuid)
    else
      context_proto = context_proto:cuda()
      encoder_grad_proto = encoder_grad_proto:cuda()
      if opt.brnn == 1 then
        encoder_bwd_grad_proto = encodr_bwd_grad_proto:cuda()
      end
    end
  end
  
  fwd_enc_init = {}
  bwd_end_init = {}
  fwd_dec_init = {}
  bwd_dec_init = {}
  
  for L = 1, opt.num_layer do
    table.insert(fwd_enc_init, hidden_init:clone())
    table.insert(fwd_enc_init, hidden_init:clone())
    table.insert(bwd_enc_init, hidden_init:clone())
    table.insert(bwd_enc_init, hidden_init:clone())    
  end
  if opt.gpuids >= 0 then
    cutorch.setDevice(opt.gpuids)
  end
  if opt.input_feed == 1 then
    table.insert(fwd_dec_init, hidden_init:clone())
  end
  table.insert(bwd_dec_init, hidden_init:clone())
  for L = 1, opt.num_layer do
    table.insert(fwd_dec_init, hidden_init:clone())
    table.insert(fwd_dec_init, hidden_init:clone())
    table.insert(bwd_dec_init, hidden_init:clone())
    table.insert(bwd_dec_init, hidden_init:clone())  
  end
  
  dec_offset = 3
  if opt.init_feed == 1 then
    dec_offset = dec_offset + 1
  end
  
  function reset_state(state, batch_l, t)
    if t == nil then
      local u = {}
      for i = 1, #state do
        state[i]:zero()
        table.insert(u, state[i][{{1, batch_l}}])
      end
      return u
    else
      local u = {[t] = {}}
      for i = 1, #state do
        state[i]:zero()
        table.insert(u[t], state[i][{{1, batch_l}}])
      end
      return u
    end
  end
  
  function clean_layer(layer)
    if opt.gpuid >= 0 then
      layer.output = torch.CudaTensor()
      layer.gradInput = torch.CudaTensor()
    else
      layer.output = torch.DoubleTensor()
      layer.gradInput = torch.DoubleTensor()
    end
    if layer.modules then
      for i, mod in ipairs(layer.modules) do
        clean_layer(mod)
      end
    elseif torch.type(self) == "nn.gModule" then
      layer:apply(clean_layer)
    end
  end
  
  function lr_decay(epoch)
    print(opt.valid_perf)
    if epoch >= opt.start_decay then
      start_decay = 1
    end
    
    if opt.valid_perf[#opt.valid_perf] ~= nil and opt.valid_perf[#opt.valid_perf - 1] ~= nil then
      local curr_ppl = opt.valid_perf[#opt.valid_perf]
      local prev_ppl = opt.valid_perf[#opt.valid_perf - 1]
      if curr_ppl > prev_ppl then
        start_decay = 1
      end
    end
    if start_decay == 1 then
      opt.learn_rate = opt.learn_rate * opt.lr_decay
    end
  end
  
  function train_batch(data, epoch)
    opt.num_src_feature = data.num_src_feature
    
    local train_nonzero = 0
    local train_loss = 0
    local train_loss_cll = 0
    local batch_order = torch.randperm(data.length)
    local start_time = timer:time().real
    local num_word_source = 0
    local num_word_target = 0
    
    for i = 1, data:size() do
      zero_table(grad_param, 'zero')
      local d
      if opoch <= opt.curriculum then
        d = data[i]
      else
        d = data[batch_order[i]]
      end
      local target, target_out, nonzero, source = d[1], d[2], d[3], d[4]
      local batch_l, target_l, source_l = d[5], d[6], d[7]
      local source_feature = d[9]
      local alignment = d[10]
      local norm_alignment
      if opt.guided_alignment == 1 then
        replicator = nn.Replicate(alignment:size(2), 2)
        if opt.gpuid >= 0 then
          cutorch.setDevice(opt.gpuid)
          if opt.gpuids >= 0 then
            cutorch.setDevice(opt.gpuids)
          end
          replicator = replicator:cuda()
        end
        norm_alignment = torch.cdiv(alignment, replicator:forward(torch.sum(alignment, 2):squeeze(2)))
        norm_alignment[norm_alignment:ne(norm_alignment)] = 0
      end
      
      local encoder_grad = encoder_grad_proto[{{1, batch_l}, {1, source_l}}]
      local encoder_bwd_grad
      if opt.brnn == 1 then
        encoder_bwd_grad = encoder_bwd_grad_proto[{{1, batch_l}, {1, source_l}}]
      end
      if opt.gpuid >= 0 then
        cutorch.setDevice(opt.gpuid)
      end
      local rnn_state_enc = reset_state(fwd_enc_init, batch_l, 0)
      local context = context_proto[{{1, batch_l}, {1, source_l}}]
      for t = 1, source_l do
        encoder_clone[t]:training()
        local encoder_input = {source[t]}
        if data.num_src_feature > 0 then
          append_table(encoder_input, source_feature[t])
        end
        append_table(encoder_input, rnn_state_enc[t-1])
        local out = encoder_clone[t]:forward(encoder_input)
        rnn_state_enc[t] = out
        context[{{}, t}]:copy(out[#out])
      end
      
      local rnn_state_enc_bwd
      if opt.brnn == 1 then
        rnn_state_enc_bwd = reset_state(fwd_enc_init, batch_l, source_l+1)
        for t = source_l, 1, -1 do
          encoder_bwd_clone[t]:training()
          local encoder_input = {source[t]}
          if data.num_src_feature > 0 then
            append_table(encoder_input, source_feature[t])
          end
          append_table(encoder_input, rnn_state_enc_bwd[t+1])
          local out = encoder_bwd_clone[t]:forward(encoder_input)
          rnn_state_enc_bwd[t] = out
          context[{{}, t}]:add(out[#out])
        end
      end
    
      if opt.gpuid >= 0 and opt.gpuids >= 0 then
        cutorch.setDevice(opt.gpuids)
        local contexts = context_protos[{{1, batch_l}, {1, source_l}}]
        contexts:copy(context)
        context = contexts
      end
      local rnn_state_dec = reset_state(fwd_dec_init, batch_l, 0)
      if opt.dec_init == 1 then
        for L = 1, opt.num_layer do
          rnn_state_dec[0][L*2-1+opt.input_feed]:copy(rnn_state_enc[source_l][L*2-1])
          rnn_state_dec[0][L*2+opt.input_feed]:copy(rnn_state_enc[source_l][L*2])
        end
        if opt.brnn == 1 then
          for L = 1, opt.num_layer do
            rnn_state_dec[0][L*2-1+opt.input_feed]:add(rnn_state_enc_bwd[1][L*2-1])
            rnn_state_dec[0][L*2+opt.input_feed]:add(rnn_state_bwd[1][L*2])
          end
        end
      end
    
      local pred = {}
      local attn_output = {}
      local decoder_input
      for t = 1, target_l do
        decoder_clone[t]:training()
        local decoder_input
        if opt.attn == 1 then
          decoder_input = {target[t], context, table.unpack(rnn_state_dec[t-1])}
        else
          decoder_input = {target[t], context[{{}, source_l}], table.unpack(rnn_state_dec[t-1])}
        end
        local out = decoder_clone[t]:forward(decoder_input)
        local out_pred_idx = #out
        if opt.gpuid_align == 1 then
          our_pred_idx = #out - 1
          table.insert(attn_output, out[#out])
        end
        local next_state = {}
        table.insert(pred, out[out_pred_idx])  
        if opt.input_feed == 1 then
          table.insert(next_state, out[out_pred_idx])
        end
        for j = 1, out_pred_idx - 1 do
          table.insert(next_state, out[j])
        end
        rnn_state_dec[t] = next_state
      end
    
      encoder_grad:zero()
      if opt.brnn == 1 then
        encoder_bwd_grad:zero()
      end
    
      local drnn_state_dec = reset_state(bwd_dec_init, batch_l)
      if opt.gpuid_align == 1 then
        attn_init:zero()
        table.insert(drnn_state_dec, attn_init[{{1, batch_l}, {1, source_l}}])
      end
      local loss = 0
      local loss_cll = 0
      for t = target_l, 1, -1 do
        local pred = generator:forward(pred[t])
        local input = pred
        local output = target_out[t]
        if opt.guided_align == 1 then
          input = {input, attn_output[t]}
          output = {output, norm_align[{{}, {}, t}]}
        end
      
        loss = loss + criterion:forward(input, output) / batch_l
      
        local drnn_state_attn
        local dl_dpred
        if opt.guided_align == 1 then
          local dl_dpred_attn = criterion:backward(input, output)
          dl_dpred = dl_dpred_attn[1]
          drnn_state_attn = dl_dpred_attn[2]
          drnn_state_attn:div(batch_l)
          loss_cll = loss_cll + cll_criterion:forward(input[1], output[1]) / batch_l
        else
          dl_dpred = criterion:backward(input, output)
        end
      
        dl_dpred:div(batch_l)
        local dl_dtar = generator:backward(pred[t], dl_dpred)
      
        local rnn_state_dec_pred_idex = #drnn_state_dec
        if opt.guided_align == 1 then
          rnn_state_dec_pred_idx = #drnn_state_dec - 1
          drnn_state_dec[#drnn_state_dec]:add(drnn_state_attn)
        end
        drnn_state_dec[rnn_state_dec_pred_idx]:add(dl_dtar)
      
        local decoder_input
        if opt.attn == 1 then
          decoder_input = {target[t], context, table.unpack(rnn_state_dec[t-1])}
        else
          decoder_input = {target[t], context[{{}, source_l}], table.unpack(rnn_state_dec[t-1])}
        end
        local dlst = decoder_clone[t]:backward(decoder_input, drnn_state_dec)
        if opt.attn == 1 then
          encoder_grad:add(dlst[2])
          if opt.brnn == 1 then
            encoder_bwd_grad:add(dlst[2])
          end
        else
          encoder_grad[{{}, source_l}]:add(dlst[2])
          if opt.brnn == 1 then
            encoder_bwd_grad[{{}, 1}]:add(dlst[2])
          end
        end
      
        drnn_state_dec[rnn_state_dec_pred_idx]:zero()
        if opt.guided_align == 1 then
          drnn_state_dec[#drnn_state_dec]:zero()
        end
        if opt.input_feed == 1 then
          drnn_state_dec[rnn_state_dec_pred_idx]:add(dlst[3])
        end
        for j = dec_offset, #dlst do
          drnn_state_dec[rnn_state_dec_pred_idx]:add(dlst[3])
        end
        for j = dec_offset, #dlst do
          drnn_state_dec[j-dec_offset+1]:copy(dlst[j])
        end
      end
      word_vec_layer[2].gradWeight[1]:zero()
      if opt.fix_word_vec_dec == 1 then
        word_vec_layer[2].gradWeight:zero()
      end
    
      local grad_norm = 0
      grad_norm = grad_norm + grad_param[2]:norm()^2 + grad_param[3]:norm()^2
    
      if opt.gpuid >= 0 and opt.gpuids >= 0 then
        cutorch.setDevice(opt.gpuid)
        local encoder_grads = encoder_grad_protos[{{1, batch_l}, {1, source_l}}]
        encoder_grads:zero()
        encoder_grads:copy(encoder_grads)
        encoder_grad = encoder_grads
      end
    
      local drnn_state_enc = reset_state(bwd_enc_init, batch_l)
      if opt.dec_init == 1 then
        for L = 1, opt.num_layer do
          drnn_state_enc[L*2-1]:copy(drnn_state_dec[L*2-1])
          drnn_state_enc[L*2]:copy(drnn_state_dec[L*2])
        end
      end
    
      for t = source_l, 1, -1 do
        local encoder_input = {source[t]}
        if data.num_src_feature > 0 then
          append_table(encoder_input, src_feature[t])
        end
        append_table(encoder_input, rnn_state_enc[t-1])
        if opt.attn == 1 then
          drnn_state_enc[#drnn_state_enc]:add(encoder_grad[{{}, t}])
        else
          if t == source_l then
            drnn_state_enc[#drnn_state_enc]:add(encoder_grad[{{}, t}])
          end
        end
        local dlst = encoder_clone[t]:backward(encoder_input, drnn_state_enc)
        for j = 1, #drnn_state_enc do
          drnn_state_enc[j]:copy(dlst[j+1+data.num_src_feature])
        end
      end
    
      if opt.brnn == 1 then
        local drnn_state_enc = reset_state(bwd_enc_init, batch_l)
        if opt.dec_init == 1 then
          for L = 1, opt.num_layer do
            drnn_state_enc[L*2-1]:copy(drnn_state_dec[L*2-1])
            drnn_state_enc[L*2]:copy(drnn_state_dec[L*2])
          end
        end
        for t = 1, source_l do
          local encoder_input = {source[t]}
          if data.num_src_feature > 0 then
            append_table(encoder_input, src_feature[t])
          end
          append_table(encoder_input, rnn_state_enc_bwd[t+1])
          if opt.attn == 1 then
            drnn_state_enc[#drnn_state_enc]:add(encoder_bwd_grad[{{}, t}])
          else
            if t == 1 then
              drnn_state_enc[#drnn_state_enc]:add(encoder_bwd_grad[{{}, t}])
            end
          end
          local dlst = encoder_bwd_clone[t]:backward(encoder_input, drnn_state_enc)
          for j = 1, #drnn_state_enc do
            drnn_state_enc[j]:copy(dlst[j+1+data.num_src_feature])
          end
        end
      end
    
      word_vec_layer[1].gradWeight[1]:zero()
      if opt.fix_word_vec_enc == 1 then
        word_vec_layer[1].gradWeight:zero()
      end
    
      if opt.brnn == 1 then
        word_vec_layer[1].gradWeight:add(word_vec_layer[3].gradWeight)
        if opt.use_char_enc == 1 then
          for j = 1, charcnn_offset do
            charcnn_grad_layer[j]:add(charcnn_grad_layer[j+charcnn_offset])
            charcnn_grad_layer[j+charcnn_offset]:zero()
          end
        end
        word_vec_layer[3].gradWeight:zero()
      end
    
      grad_norm = grad_norm + grad_param[1]:norm()^2
      if opt.brnn == 1 then
        grad_norm = grad_norm + grad_param[4]:norm()^2
      end
      grad_norm = grad_norm^0.5
      local param_norm = 0
      local shrinkage = opt.max_grad_norm / grad_norm
      for j = 1, #grad_param do
        if opt.gpuid >= 0 and opt.gpuids >= 0 then
          if j == 1 then
            cutorch.setDevice(opt.gpuid)
          else
            cutorch.setDevice(opt.gpuids)
          end
        end
        if shrinkage < 1 then
          grad_params[j]:mul(shrinkage)
        end
        if opt.optim == 'adagrad' then
          adagrad_step(param[j], grad_param[j], layer_eta[j], optState[j])
        elseif opt.optim == 'adadelta' then
          adadelta_step(param[j], grad_param[j], layer_eta[j], optState[j])
        elseif opt.optim == 'adam' then
          adam_step(param[j], grad_param[j], layer_eta[j], optState[j])
        else
          param[j]:add(grad_param[j]:mul(-opt.learn_rate))
        end
        param_norm = param_norm + param[j]:norm()^2
      end
      param_norm = param_norm^0.5
      if opt.brnn == 1 then
        word_vec_layer[3].weight:copy(word_vec_layer[1].weight)
        if opt.use_char_enc == 1 then
          for j = 1, charcnn_offset do
            charcnn_layer[j+charcnn_offset]:copy(charcnn_layer[j])
          end
        end
      end
    
      num_word_tar = num_word_tar + batch_layer * tar_layer
      num_word_src = num_word_src + batch_layer * src_layer
      train_nonzero = train_nonzero + nonzero
      train_loss = train_loss + loss * batch_layer
      if opt.guide_align == 1 then
        train_loss_cll = train_loss_cll + loss_cll * batch_layer
      end
      local time_taken = timer:time().real - start_time
      if i % opt.print_every == 0 then
        local stat = string.format('epoch: %d, batch: %d/%d, batch size: %d, lr: %.4f, ', 
                     epoch, i, data:size(), batch_layer, opt.learn_rate)
        if opt.guide_align == 1 then
          stat = stat .. string.format('ppl: %.2f, ppl_cll: %.2f, |param|: %.2f, |gparam|: %.2f, ',
                 math.exp(train_loss/train_nonzero), math.exp(train_loss_cll/train_nonzero), param_norm, grad_norm)
        else
          stat = stat .. string.format('ppl: %.2f, |param|: %.2f, |gparam|: %.2f, ', 
                 math.exp(train_loss/train_nonzero), param_norm, grad_norm)
        end
        stat = stat .. string.format('training: %d/%d/%d total/source/target tokens/sec',
               (num_word_tar + num_word_src) / time_taken, num_word_src / time_taken, num_word_tar / time_taken)
        print(stat)
      end
      if i % 200 == 0 then
        collectgarbage()
      end
    end
    if opt.guide_align == 1 then
      return train_loss, train_nonzero, train_loss_cll
    else
      return train_loss, train_nonzero
    end
  end   
  
  local total_loss, total_nonzero, batch_loss, batch_nonzero, total_loss_cll, batch_loss_cll
  for epoch = opt.start_epoch, opt.epoch do
    generator:training()
    if opt.num_shard > 0 then
      total_loss = 0
      total_nonzero = 0
      total_loss_cll = 0
      local shard_order = torch.randperm(opt.num_shard)
      for s = 1, opt.num_shard do
        local fn = train_data .. '.' .. shard_order[s] .. '.hdf5'
        print('load shard #' .. shard_order[s])
        local shard_data = data.new(opt, fn)
        if opt.guide_align == 1 then
          batch_loss, batch_nonzero, batch_loss_cll = train_batch(shard_data, epoch)
          total_loss_cll = total_loss_cll + batch_loss_cll
        else
          batch_loss, batch_nonzero = train_batch(shard_data, epoch)
        end
        total_loss = total_loss + batch_loss
        total_nonzero = total_nonzero + batch_nonzero
      end
    else
      if opt.guide_align == 1 then
        total_loss, total_nonzero, total_loss_cll = train_batch(train_data, epoch)
      else
        total_loss, total_nonzero = train_batch(train_data, epoch)
      end
    end
    local train_score = math.exp(total_loss / total_nonzero)
    print('train', train_score)
    opt.train_perf[#opt.train_perf + 1] = train_score
    local score = eval(valid_data)
    opt.valid_perf[#opt.valid_perf + 1] = score
    if opt.optim == 'sgd' then
      lr_decay(epoch)
    end
    if opt.guide_align == 1 then
      opt.guide_align_weight = opt.guide_align_weight * opt.guide_align_decay
      criterion.weights[1] = 1 - opt.guide_align_weight
      criterion.weights[2] = opt.guide_align_weight
    end
    local savefile = string.format('%s_epoch%.2f_%.2f.t7', opt.savefile, epoch, score)
    if epoch % opt.save_every == 0 then
      print('saving checkpoint on ' ..  savefile)
      clean_layer(generator)
      if opt.brnn == 0 then
        torch.save(savefile, {{encoder, decoder, generator}, opt})
      else
        torch.save(savefile, {{encoder, decoder, generator, encoder_bwd}, opt})
      end
    end
  end
  local savefile = string.format('%s_final.t7', opt.savefile) 
  clean_layer(generator)
  print('saving final model to ' .. savefile)
  if opt.brnn == 0 then
    torch.save(savefile, {{encoder:double(), decoder:double(), generator:double()}, opt})
  else
    torch.save(savefile, {{encoder:double(), decoder:double(), generator:double(), encoder_bwd:double()}, opt})
  end
end

function eval(data)
  encoder_clone[1]:evaluate()
  decoder_clone[1]:evaluate()
  generator:evaluate()
  if opt.brnn == 1 then
    encoder_bwd_clone[1]:evaluate()
  end
  
  local nll = 0
  local nll_cll = 0
  local total = 0
  for i = 1, data:size() do
    local d = data[i]
    local target, target_out, nonzero, source = d[1], d[2], d[3], d[4]
    local batch_layer, target_layer, source_layer = d[5], d[6], d[7]
    local source_feature = d[9]
    local alignment = d[10]
    local norm_align
    if opt.guide_align == 1 then
      replicator = nn.Replicate(alignment:size(2), 2)
      if opt.gpuid >= 0 then
        cutorch.setDevice(opt.gpuid)
        if opt.gpuids >= 0 then
          cutorch.setDevice(opt.gpuids)
        end
        replicator = replicator:cuda()
      end
      norm_align = torch.cdiv(alignment, replicator:forward(torch.sum(alignment, 2):squeeze(2)))
      norm_align[norm_align:ne(norm_align)] = 0
    end
    if opt.gpuid >= 0 and opt.gpuids >= 0 then
      cutorch.setDevice(opt.gpuid)
    end
    local rnn_state_enc = reset_state(fwd_enc_init, batch_layer)
    local context = context_proto[{{1, batch_layer}, {1, source_layer}}]
    for t = 1, source_layer do
      local encoder_input = {source[t]}
      if data.num_src_feature > 0 then
        append_table(encoder_input, source_feature[t])
      end
      append_table(encoder_input, rnn_state_enc)
      local out = encoer_clone[1]:forward(encoder_input)
      rnn_state_enc = out
      context[{{}, t}]:copy(out[#out])
    end
    
    if opt.gpuid >= 0 and opt.gpuids >= 0 then
      cutorch.setDevice(opt.gpuids)
      local contexts = context_protos[{{1, batch_layer}, {1, source_layer}}]
      contexts:copy(context)
      context = contexts
    end
    
    local rnn_state_dec = reset_state(fwd_dec_init, batch_layer)
    if opt.dec_init == 1 then
      for L = 1, opt.num_layer do
        rnn_state_dec[L*2-1+opt.input_feed]:copy(rnn_state_enc[L*2-1])
        rnn_state_dec[L*2+opt.input_feed]:copy(rnn_state_enc[L*2])
      end
    end
    
    if opt.brnn == 1 then
      local rnn_state_enc = reset_state(fwd_enc_init, batch_layer)
      for t = source_layer, 1, -1 do
        local encoder_input = {source[t]}
        if data.num_src_feature > 0 then
          append_table(encoder_input, source_feature[t])
        end
        append_table(encoder_input, rnn_state_enc)
        local out = encoder_bwd_clone[1]:forward(encoder_input)
        rnn_state_enc = out
        context[{{}, t}]:add(out[#out])
      end
      if opt.dec_init == 1 then
        for L = 1, opt.num_layer do
          rnn_state_dec[L*2-1+opt.input_feed]:add(rnn_state_enc[L*2-1])
          rnn_state_dec[L*2+opt.input_feed]:add(rnn_state_enc[L*2])
        end
      end
    end
    
    local loss = 0
    local loss_cll = 0
    local attn_output = {}
    for t = 1, tar_layer do
      local decoder_input
      if opt.attn == 1 then
        decoder_input = {target[t], context, table.unpack(rnn_state_dec)}
      else
        decoder_input = {target[t], context[{{}, source_layer}], table.unpack(rnn_state_dec)}
      end
      local out = decoder.clone[1]:forward(decoder_input)
      
      local out_pred_idx = #out
      if opt.guide_align == 1 then
        out_pred_idx = #out - 1
        table.insert(attn_output, out[#out])
      end
      
      rnn_state_dec = {}
      if opt.input_feed == 1 then
        table.insert(rnn_state_dec, out[out_pred_idx])
      end
      for j = 1, out_pred_idx - 1 do
        table.insert(rnn_state_dec, out[j])
      end
      local pred = generator:forward(out[out_pred_idx])
      
      local input_pred
      local output = target_out[t]
      if opt.guide_alignment == 1 then
        input = {input, attn_output[t]}
        output = {output, norm_align[{{}, {}, t}]}
      end
      
      loss = loss + criterion:forward(input, output)
      
      if opt.guide_align == 1 then
        loss_cll = loss_cll + cll_criterion:forward(input[1], output[1])
      end
    end
    nll = nll + loss
    if opt.guide_align == 1 then
      nll_cll = nll_cll + loss_cll
    end
    total = total_nonzero
  end
  local valid = math.exp(nll / total)
  print("valid", valid)
  if opt.guide_align == 1 then
    local valid_cll = math.exp(nll_cll / total)
    print("valid_cll", valid_cll)
  end
  collectgarbage()
  return valid
end   

function get_layer(layer)
  if layer.name ~= nil then
    if layer.name == 'word_vec_dec' then
      table.insert(word_vec_layer, layer)
    elseif layer.name == 'word_vec_enc' then
      table.insert(word_vec_layer, layer)
    elseif layer.name == 'charcnn_enc' or layer.name == 'mlp_enc' then
      local p, gp = layer:parameters()
      for i = 1, #p do
        table.insert(charcnn_layer, p[i])
        table.insert(charcnn_grad_layer, gp[i])
      end
    end
  end
end

function main()
  opt = cmd:parse(arg)  
  torch.manualSeed(opt.seed)  
  if opt.gpuid >= 0  then
    print('using cuda on gpu ' .. opt.gpuid .. '...')
    if opt.gpuids >= 0 then
      print('using cuda on second gpu ' .. opt.gpuids .. '...')
    end
    require('cunn')
    require('cutorch')
    if opt.cudnn == 1 then
      print('loading cudnn...')
      require('cudnn')
    end
    cutorch.setDevice(opt.gpuid)
    cutorch.manualSeed(opt.seed)
  end
  
  print('loading data...')
  if opt.num_shard == 0 then
    train_data = data.new(opt, opt.data_file)
  else
    train_data = opt.data_file
  end
  
  valid_data = data.new(opt, opt.valid_file)
  print('done!')
  print(string.format('source vocab size: %d, target vocab size: %d', valid_data.source_size, valid_data.target_size))
  opt.max_sent_l_src = valid_data.source:size(2)
  opt.max_sent_l_tar = valid_data.target:size(2)
  opt.max_sent_l = math.max(opt.max_sent_l_src, opt.max_sent_l_tar)
  if opt.max_batch_l == '' then
    opt.max_batch_l = valid_data.batch_l:max()
  end
  
  if opt.use_char_enc == 1 or opt.use_char_dec == 1 then
    opt.max_word_l = valid_data.char_length
  end
  print(string.format('source max sent len: %d, target max sent len: %d', valid_data.source:size(2), valid_data.target:size(2)))
  print(string.format('number of additional features on source side: %d', valid_data.num_src_feature))
  
  preallocateMemory(opt.prealloc)
  
  if opt.train_from:len() == 0 then
    encoder = make_lstm(valid_data, opt, 'enc', opt.use_char_enc)
    decoder = make_lstm(valid_data, opt, 'dec', opt.use_char_dec)
    generator, criterion = make_generator(valid_data, opt)
    if opt.brnn == 1 then
      encoder_bwd = make_lstm(valid_data, opt, 'enc', opt.use_char_enc)
    end
  else
    assert(path.exists(opt.train_from), 'checkpoint path invalid')
    print('loading ' .. opt.train_from .. '...')
    local checkpoint = torch.load(opt.train_from)
    local model, model_opt = checkpoint[1], chenkpoint[2]
    opt.num_layer = model_opt.num_layer
    opt.rnn_size = model_opt.rnn_size
    opt.input_feed = model_opt.input_feed or 1
    opt.attn = model_opt.attn or 1
    opt.brnn = model_opt.brnn or 0
    encoder = model[1]
    decoder = model[2]
    generator = model[3]
    if model_opt.brnn == 1 then 
      encoder_bwd = model[4]
    end
    _, criterion = make_generator(valid_data, opt)
  end
  
  if opt.guide_align == 1 then
    cll_criterion = criterion
    criterion = nn.ParallelCriterion()
    criterion:add(cll_criterion, (1 - opt.guide_align_weight))
    criterion:add(nn.MSECriterion(), opt.guide_align_weight)
  end
  
  layer = {encoder, decoder, generator}
  if opt.brnn == 1 then
    table.insert(layer, encoder_bwd)
  end
  
  if opt.optim ~= 'sgd' then
    layer_eta = {}
    optState = {}
    
    if opt.layer_lr:len() > 0  then
      local stringx = require('pl.stringx')
      local str_lr = stringx.split(opt.layer_lr, ',')
      if #str_lr ~= #layer then error('1 learning rate per layer expected') end
      for i = 1, #str_lr do
        local lr = tonumber(stringx.strip(str_lr[i]))
        if not lr then
          error(string.format('malformed learning rate: %s', str_lr[i]))
        else
          layer_eta[i] = lr
        end
      end
    end
    
    for i = 1, #layer do
      layer_eta[i] = layer_eta[i] or opt.learn_rate
      optState[i] = {}
    end
  end
  
  if opt.gpuid >= 0 then
    for i = 1, #layer do 
      if opt.gpuids >= 0 then
        if i == 1 or i == 4 then
          cutorch.setDevice(opt.gpuid)
        else
          cutorch.setDevice(opt.gpuids)
        end
      end
      layer[i]:cuda()
    end
    if opt.gpuids >= 0 then
      cutorch.setDevice(opt.gpuids)
    end
    criterion:cuda()
  end
  
  word_vec_layer = {}
  if opt.use_char_enc == 1 then
    char_cnn_layer = {}
    char_cnn_grad_layer = {}
  end
  encoder:apply(get_layer)
  decoder:apply(get_layer)
  if opt.brnn == 1 then 
    if opt.use_char_enc == 1 then
      char_cnn_offset = #char_cnn_layer
    end
    encoder_bwd:apply(get_layer)
  end
  train(train_data, valid_data)              
end

main()
