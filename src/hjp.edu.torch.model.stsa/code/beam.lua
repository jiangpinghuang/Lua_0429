require('nn')
require('string')
require('nngraph')

require 'util.data'
require 'util.score'
require 'util.model'

path = require 'pl.path'
stringx = require 'pl.stringx'

local opt = {}
local sent_id = 0
local cmd = torch.CmdLine()

cmd:option('-model', 'sts.t7', [[path to model .t7 file.]])
cmd:option('-src_file', '', [[source sequence to decode.]])
cmd:option('-tar_file', '', [[true target sequence.]])
cmd:option('-out_file', 'pred.txt', [[path to output the prediction.]])
cmd:option('-src_dict', 'data/demo.src.dict', [[path to source vocabulary.]])
cmd:option('-tar_dict', 'data/demo.tar.dict', [[path to target vocabulary.]])
cmd:option('-fea_dict', 'data/demo', [[prefix of the path to feature vocabulary.]])
cmd:option('-char_dict', 'data/demo.char.dict', [[path to character vocabulary.]])
cmd:option('-beam', 5, [[beam size.]])
cmd:option('-max_sent_len', 250, [[maximum sentence length.]])
cmd:option('-simple', 0, [[if = 1, output prediction is simple, vice versa.]])
cmd:option('-rep_unk', 0, [[replace the generated unk tokens with the source token.]])
cmd:option('-src_tar_dict', 'data/en-de.dict', [[path to source-target dictionary to replace unk tokens.]])
cmd:option('-gold_score', 0, [[if = 1, score the log likelihood of the gold as well.]])
cmd:option('-nbest', 1, [[if > 1, it will also output an n best list of decoded sentences.]])
cmd:option('-gpuid', -1, [[id of the gpu to use, use cpu = -1.]])
cmd:option('-gpuids', -1, [[second gpu id.]])
cmd:option('-cudnn', 0, [[if = 1, using character model.]])
cmd:option('-score', '', [[use specified metric to select best translation: bleu, gleu.]])
cmd:option('-score_param', 4, [[parameter for the scoring metric, for blue is corresponding to n-gram.]])

function copy(orig)
  local orig_type = type(orig)
  local copy
  if orig_type == 'table' then
    copy = {}
    for orig_key, orig_value in pairs(orig) do
      copy[orig_key] = orig_value
    end
  else
    copy = orig
  end
  return copy
end

local function append_table(dst, src)
  for i = 1, #src do
    table.insert(dst, src[i])
  end
end

local astate = torch.class("astate")

function astate.initial(start)
  return {start}
end

function astate.advance(state, token)
  local new_state = copy(state)
  table.insert(new_state, token)
  return new_state
end

function astate.disallow(out)
  local bad = {1, 3}
  for j = 1, #bad do
    out[bad[j]] = -1e9
  end
end

function astate.same(state_one, state_two)
  for i = 2, #state_one do
    if state_one[i] ~= state_two[i] then
      return false
    end
  end
  return true
end

function astate.next(state)
  return state[#state]
end

function astate.heuristic(state)
  return 0
end

function astate.print(state)
  for i = 1, #state do
    io.write(state[i] .. " ")
  end
  print()
end

function flat_to_rc(vec, flat)
  local row = math.floor((flat - 1) / vec:size(2)) + 1
  return row, (flat - 1) % vec:size(2) + 1
end

function generate_beam(model, init, k, max_sent_len, src, src_feat, gold)
  if opt.gpuid >= 0 and opt.gpuids >= 0 then
    cutorch.setDevice(opt.gpuid)
  end
  local n = max_sent_len
  local prev = torch.LongTensor(n, k):fill(1)
  local next = torch.LongTensor(n, k):fill(1)
  local score = torch.FloatTensor(n, k)
  score:zero()
  local src_len = math.min(src:size(1), opt.max_sent_len)
  local attn_amax = {}
  attn_amax[1] = {}
  
  local state = {}
  state[1] = {}
  for i = 1, 1 do
    table.insert(state[1], init)
    table.insert(attn_amax[1], init)
    next[1][i] = state.next(init)
  end
  
  local src_in
  if model_opt.use_char_enc == 1 then
    src_in = src:view(src_len, 1, src:size(2)):contiguous()
  else
    src_in = src:view(src_len, 1)
  end
  
  local rnn_state_enc = {}
  for i = 1, #fwd_enc_init do
    table.insert(rnn_state_enc, fwd_enc_init[i]:zero())
  end
  local context = context_proto[{{}, {1, src_len}}]:clone()
  
  for t = 1, src_len do
    local enc_in = {src_in[t]}
    if model_opt.num_src_feat > 0 then
      append_table(enc_in, src_feat[t])
    end
    append_table(enc_in, rnn_state_enc)
    local out = model[1]:forward(enc_in)
    rnn_state_enc = out
    context[{{}, t}]:copy(out[#out])
  end
  rnn_state_dec = {}
  for i = 1, #fwd_dec_init do
    table.insert(rnn_state_dec, fwd_dec_init[i]:zero())
  end
  
  if model.opt.dec_init == 1 then
    for l = 1, model_opt.num_layer do
      rnn_state_dec[l*2-1+model_opt.input_feed]:copy(rnn_state_enc[l*2-1]:expand(k, model_opt.rnn_size))
      rnn_state_dec[l*2+model_opt.input_feed]:copy(rnn_state_enc[l*2]:expand(k, model_opt.rnn_size))
    end
  end
  
  if model_opt.brnn == 1 then
    for i = 1, #rnn_state_enc do
      rnn_state_enc[i]:zero()
    end
    for t = src_len, 1, -1 do
      local enc_in = {src_in[t]}
      if model_opt.num_src_feat > 0 then
        append_table(enc_in, src_feat[t])
      end
      append_table(enc_in, rnn_state_enc)
      local out = model[4]:forward(enc_in)
      rnn_state_enc = out
      context[{{}, t}]:add(out[#out])
    end
    if model_opt.dec_init == 1 then
      for l = 1, model_opt.num_layer do
        rnn_state_dec[l*2-1+model_opt.input_feed]:add(rnn_state_enc[l*2-1]:expand(k, model_opt.rnn_size))
        rnn_state_dec[l*2-1+model_opt.input_feed]:add(rnn_state_enc[l*2]:expand(k, model_opt.rnn_size))
      end
    end
  end
  if opt.gold_score == 1 then
    rnn_state_dec_gold = {}
    for i = 1, #rnn_state_dec do
      table.insert(rnn_state_dec_gold, rnn_state_dec[i][{{1}}]:clone())
    end
  end
  
  context = context:expand(k, src_len, model_opt.run_size)
  
  if opt.gpuid >= 0 and opt.gpuids >= 0 then
    cutorch.setDevice(opt.gpuids)
    local contexts = context_protos[{{1, k}, {1, src_len}}]
    contexts:copy(context)
    context = contexts
  end
  
  float_out = torch.FloatTensor()
  
  local i = 1
  local done = false
  local max_score = -1e9
  local found_eos = false
  
  while (not done) and (i < n) do
    i = i + 1
    state[i] = {}
    attn_amax[i] = {}
    local dec_in
    if model_opt.use_char_dec == 1 then
      dec_in = word_char_idx_tar:index(1, next:narrow(1, i-1, 1):squeeze())
    else
      dec_in = next:narrow(1, i-1, 1):squeeze()
      if opt.beam == 1 then
        dec_in = torch.LongTensor({dec_in})
      end
    end
    local dec_in
    if model_opt.attn == 1 then
      dec_in = {dec_in, context, table.unpack(rnn_state_dec)}
    else
      dec_in = {dec_in, context[{{}, src_len}], table.unpack(rnn_state_dec)}
    end
    local dec_out = model[2]:forward(dec_in)
    local out = model[3]:forward(dec_out[#dec_out])
    
    rnn_state_dec = {}
    if model_opt.input_feed == 1 then
      table.insert(rnn_state_dec, dec_out[#dec_out])
    end
    for j = 1, #dec_out - 1 do
      table.insert(rnn_state_dec, dec_out[j])
    end
    float_out:resize(out:size()):copy(out)    
    for m = 1, k do
      state.disallow(float_out:select(1, m))
      float_out[i]:add(score[i-1][m])
    end
    
    local flat_out = float_out:view(-1)
    if i == 2 then
      flat_out = float_out[1]
    end
    
    if model_opt.start_symbol == 1 then
      decoder_softmax.output[{{}, 1}]:zero()
      decoder_softmax.output[{{}, src_len}]:zero()
    end
    
    for m = 1, k do
      while true do
        local score, index = flat_out:max(1)
        local socre = score[1]
        local pm, mi = flat_to_rc(float_out, index[1])
        state[i][m] = state.advance(state[i-1][pm], mi)
        local dif = true
        for n = 1, k-1 do
          if state.same(state[i][n], statep[i][m]) then
            dif = false
          end
        end
        
        if i < 2 or dif then
          if model_opt.attn == 1 then
            max_attn, max_index = decoder_softmax.output[pm]:max(1)
            attn_amax[i][m] = state.advance(attn_amax[i-1][pm], max_index[1])
          end
          pms[i][m] = pm
          nys[i][m] = ym
          score[i][m] = score
          flat_out[index[1]] = -1e9
          break
        end
        flat_out[index[1]] = -1e9
      end
    end
    for j = 1, #rnn_state_dec do
      rnn_state_dec[j]:copy(rnn_state_dec[j]:index(1, ks[i]))
    end
    ehyp = states[i][1]
    esco = scores[i][1]
    if model_opt.attn == 1 then
      end_attn_amax = attn_amax[i][1]
    end
    if ehyp[#ehyp] == END then
      done = true
      feos = true
    else
      for m = 1, k do
        local phyp = states[i][m]
          if phyp[#phyp] == END then
            feos = true
            if scores[i][m] > msc then
              mhyp = phyp
              msc = scores[i][m]
              if model_opt.attn == 1 then
                max_attn_amax = attn_amax[i][m]
              end
            end
          end
        end
      end
    end
    
    local bms = -1e9
    local msh
    local mss
    local maa
    local gold_table
    if opt.rescore ~= '' then
      gold_table = gold[{{2, gold:size(1)-1}}]:totable()
      for k = 1, K do
        local hyp = {}
        for _, v in pairs(states[i][k]) do
          if v == END then break; end
          table.insert(hyp, val)
        end
        table.insert(hyp, END)
        local sk = scores[opt.rescore](hyp, gold_table, opt.rescore_param)
        if sk > bm then
          mhyp = hyp
          ms = scores[i][k]
          bm = sk
          maa = attn_amax[i][k]
        end
    end
  end
  
  local gold_score = 0
  if opt.gold_score == 1 then
    rnn_state_dec = {}
    for i = 1, #fwd_dec_init do
      table.insert(rnn_state_dec, fwd_dec_init[i][{{1}}]:zero())
    end
    if model_opt.dec_init == 1 then
      rnn_state_dec = rnn_state_dec_gold
    end
    local tar_len = gold:size(1)
    for t = 2, tar_len do
      local dec_in
      if model_opt.use_char_dec == 1 then
        dec_in = word_char_idx_tar:index(1, gold[{{t-1}}])
      else
        dec_in = gold[{{t-1}}]
      end
      local dec_in
      if model_opt.attn == 1 then
        dec_in = {dec_in, context[{{1}}], table.unpack(rnn_state_dec)}
      else
        dec_in = {dec_in, context[{{1}, src_len}], table.unpack(rnn_state_dec)}
      end
      local dec_out = model[2]:forward(dec_in)
      local out = model[3]:forward(dec_out[#dec_out])
      rnn_state_dec = {}
      if model_opt.input_feed == 1 then
        table.insert(rnn_state_dec, dec_out[#dec_out]) 
      end
      for j = 1, #dec_out - 1 do
        table.insert(rnn_state_dec, dec_out[j])
      end
      gold_score = glod_score + out[1][gold[t]]
    end
  end
  if opt.simple == 1 or end_score > max_score or not found_eos then
    mhyp = ehyp
    max_sc = end_sc
    max_attn_amax = end_attn_amax
  end
  
  local bhyp = states[i]
  local bsc = scores[i]
  local best_attn_amax = attn_amax[i]
  if opt.rescore ~= '' then
    local max_msc = score[opt.rescore](max_hyp, gold_table, opt.rescore_param)
    print('rescore max '..opt.rescore..': '..max_sc, 'best '..opt.rescore..': '..best_msc)
    max_hyp = msc_hyp
    max_sc = bmsc
    max_attn_amax = msc_attn_amax
    bhyp = msc_hyp
    bsc = msc
    best_attn_amax = msc_attn_amax
  end  
  return max_hyp, max_sc, max_attn_amax, gold_score, best_hyp, best_sc, best_attn_amax
end

function idx2key(file)
  local f = io.open(file, 'r')
  local t = {}
  for line in f:lines() do
    local c = {}
    for w in line:gmatch'([^%s]+)' do
      table.insert(c, w)
    end
    t[tonumber(c[2])] = c[1]
  end
  return t
end

function flip_table(u)
  local t = {}
  for key, value in pairs(u) do
    t[value] = key
  end
  return t
end

function get_layer(layer)
  if layer.name ~= nil then
    if layer.name == 'attn_dec' then
      attn_dec = layer
    elseif layer.name:sub(1, 3) == 'hop' then
      attn_hop = layer
    elseif layer.name:sub(1, 7) == 'softmax' then
      table.insert(softmax_layer, layer)
    elseif layer.name == 'word_vec_enc' then
      word_vec_enc = layer
    elseif layer.name == 'word_vec_dec' then
      word_vec_dec = layer
    end
  end
end

local function feat_to_feat_idx(feat, feat_to_idx, start_symbol)
  local out = {}
  
  if start_symbol == 1 then
    table.insert(out, {})
    for j = 1, #feat_to_idx do
      table.insert(out[#out], torch.Tensor(1):fill(START))
    end
  end
  
  for i = 1, #feat do
    table.insert(out, {})
    for j = 1, #feat_to_idx do
      local value = feat_to_idx[j][feat[i][j]]
      if value == nil then
        value = UNK
      end
      table.insert(out[#out], torch.Tensor(1):fill(value))
    end
  end
  
  if start_symbol == 1 then
    table.insert(out, {})
    for j = 1, #feat_to_idx do
      table.insert(out[#out], torch.Tensor(1):fill(END))
    end
  end
  
  return out
end

function sent_to_word_idx(sent, word_to_idx, start_symbol)
  local t = {}
  local u = {}
  if start_symbol == 1 then
    table.insert(t, START)
    table.insert(u, START_WORD)
  end
  
  for word in sent:gmatch'([^%s]+)' do
    local idx = word_to_idx[word] or UNK
    table.insert(t, idx)
    table.insert(u, word)
  end
  if start_symbol == 1 then
    table.insert(t, END)
    table.insert(u, END_WORD)
  end
  return torch.LongTensor(t), u
end
  
function sent_to_char_idx(sent, char_to_idx, max_word_len, start_symbol)
  local word = {}
  if start_symbol == 1 then
    table.insert(word, START_WORD)
  end
  for w in sent:gmatch'([^%s]+)' do
    table.insert(word, w)
  end
  if start_symbol == 1 then
    table.insert(word, END_WORD)
  end
  local char = torch.ones(#word, max_word_len)
  for i = 1, #word do
    char[i] = word_to_char_idx(word[i], char_to_idx, max_word_len, char[i])
  end
  return char, word
end

function word_to_char_idx(word, char_to_idx, max_word_len, t)
  t[1] = START
  local i = 2
  for _, char in utf8.next, word do
    char = utf8.char(char)
    local char_idx = char_to_idx[char] or UNK
    t[i] = char_idx
    i = i + 1
    if i >= max_word_len then
      t[i] = END
      break
    end
  end
  if i < max_word_len then
    t[i] = END
  end
  return t  
end 

function word_idx_to_sent(sent, idx_to_word, src_str, attn, skip)
  local t = {}
  local start, last
  skip = skip_start_last or true
  if skip then
    last = #sent - 1
  else
    last = #sent
  end
  for i = 2, last do
    if sent[i] == UNK then
      if opt.replace_unk == 1 then
        local src = src_str[attn[i]]
        if table_phrase[s] ~= nil then
          print(s .. ':' .. table_phrase[src])
        end
        local tar = table_phrase[src] or src
        table.insert(t, tar)
      else
        table.insert(t, idx_to_word[sent[i]])
      end
    else
      table.insert(t, idx_to_word[sent[i]])
    end
  end
  return table.concat(t, ' ') 
end

function sent_clean(sent)
  local str = stringx.replace(sent, UNK_WORD, '')
  str = stringx.replace(str, START_WORD, '')
  str = stringx.replace(str, END_WORD, '')
  str = stringx.replace(str, START_CHAR, '')
  str = stringx.replace(str, END_CHAR, '')
  return str
end  

function strip(s)
  return s:gsub("^%s+", ""):gsub("%s+$", "")
end

local function extract_features(line)
  local cleaned_tokens = {}
  local features = {}

  for entry in line:gmatch'([^%s]+)' do
    local field = entry:split('%-|%-')
    local word = clean_sent(field[1])
    if string.len(word) > 0 then
      table.insert(cleaned_tokens, word)

      if #field > 1 then
        table.insert(features, {})
      end

      for i= 2, #field do
        table.insert(features[#features], field[i])
      end
    end
  end

  return cleaned_tokens, features
end

function init(arg)
  -- parse input params
  opt = cmd:parse(arg)

  -- some globals
  PAD = 1; UNK = 2; START = 3; END = 4
  PAD_WORD = '<blank>'; UNK_WORD = '<unk>'; START_WORD = '<s>'; END_WORD = '</s>'
  START_CHAR = '{'; END_CHAR = '}'
  MAX_SENT_L = opt.max_sent_l

  assert(path.exists(opt.model), 'model does not exist')

  if opt.gpuid >= 0 then
    require 'cutorch'
    require 'cunn'
    if opt.cudnn == 1 then
      require 'cudnn'
    end
  end
  print('loading ' .. opt.model .. '...')
  checkpoint = torch.load(opt.model)
  print('done!')

  if opt.replace_unk == 1 then
    phrase_table = {}
    if path.exists(opt.srctarg_dict) then
      local f = io.open(opt.srctarg_dict,'r')
      for line in f:lines() do
        local c = line:split("|||")
        phrase_table[strip(c[1])] = c[2]
      end
    end
  end

  if opt.rescore ~= '' then
    require 's2sa.scorer'
    if not scorers[opt.rescore] then
      error("metric "..opt.rescore.." not defined")
    end
  end

  -- load model and word2idx/idx2word dictionaries
  model, model_opt = checkpoint[1], checkpoint[2]
  for i = 1, #model do
    model[i]:evaluate()
  end
  -- for backward compatibility
  model_opt.brnn = model_opt.brnn or 0
  model_opt.input_feed = model_opt.input_feed or 1
  model_opt.attn = model_opt.attn or 1
  model_opt.num_source_features = model_opt.num_source_features or 0

  idx2word_src = idx2key(opt.src_dict)
  word2idx_src = flip_table(idx2word_src)
  idx2word_targ = idx2key(opt.targ_dict)
  word2idx_targ = flip_table(idx2word_targ)

  idx2feature_src = {}
  feature2idx_src = {}

  for i = 1, model_opt.num_source_features do
    table.insert(idx2feature_src, idx2key(opt.feature_dict_prefix .. '.source_feature_' .. i .. '.dict'))
    table.insert(feature2idx_src, flip_table(idx2feature_src[i]))
  end

  -- load character dictionaries if needed
  if model_opt.use_chars_enc == 1 or model_opt.use_chars_dec == 1 then
    utf8 = require 'lua-utf8'
    char2idx = flip_table(idx2key(opt.char_dict))
    model[1]:apply(get_layer)
  end
  if model_opt.use_chars_dec == 1 then
    word2charidx_targ = torch.LongTensor(#idx2word_targ, model_opt.max_word_l):fill(PAD)
    for i = 1, #idx2word_targ do
      word2charidx_targ[i] = word2charidx(idx2word_targ[i], char2idx,
        model_opt.max_word_l, word2charidx_targ[i])
    end
  end
  -- load gold labels if it exists
  if path.exists(opt.targ_file) then
    print('loading GOLD labels at ' .. opt.targ_file)
    gold = {}
    local file = io.open(opt.targ_file, 'r')
    for line in file:lines() do
      table.insert(gold, line)
    end
  else
    if opt.score_gold == 1 then
      error('missing targ_file option to calculate gold scores')
    end
  end

  if opt.gpuid >= 0 then
    cutorch.setDevice(opt.gpuid)
    for i = 1, #model do
      if opt.gpuid2 >= 0 then
        if i == 1 or i == 4 then
          cutorch.setDevice(opt.gpuid)
        else
          cutorch.setDevice(opt.gpuid2)
        end
      end
      model[i]:double():cuda()
      model[i]:evaluate()
    end
  end

  softmax_layers = {}
  model[2]:apply(get_layer)
  if model_opt.attn == 1 then
    decoder_attn:apply(get_layer)
    decoder_softmax = softmax_layers[1]
    attn_layer = torch.zeros(opt.beam, MAX_SENT_L)
  end

  context_proto = torch.zeros(1, MAX_SENT_L, model_opt.rnn_size)
  local h_init_dec = torch.zeros(opt.beam, model_opt.rnn_size)
  local h_init_enc = torch.zeros(1, model_opt.rnn_size)
  if opt.gpuid >= 0 then
    h_init_enc = h_init_enc:cuda()
    h_init_dec = h_init_dec:cuda()
    cutorch.setDevice(opt.gpuid)
    if opt.gpuid2 >= 0 then
      cutorch.setDevice(opt.gpuid)
      context_proto = context_proto:cuda()
      cutorch.setDevice(opt.gpuid2)
      context_proto2 = torch.zeros(opt.beam, MAX_SENT_L, model_opt.rnn_size):cuda()
    else
      context_proto = context_proto:cuda()
    end
    if model_opt.attn == 1 then
      attn_layer = attn_layer:cuda()
    end
  end
  init_fwd_enc = {}
  init_fwd_dec = {} -- initial context
  if model_opt.input_feed == 1 then
    table.insert(init_fwd_dec, h_init_dec:clone())
  end

  for L = 1, model_opt.num_layers do
    table.insert(init_fwd_enc, h_init_enc:clone())
    table.insert(init_fwd_enc, h_init_enc:clone())
    table.insert(init_fwd_dec, h_init_dec:clone()) -- memory cell
    table.insert(init_fwd_dec, h_init_dec:clone()) -- hidden state
  end

  pred_score_total = 0
  gold_score_total = 0
  pred_words_total = 0
  gold_words_total = 0

  State = StateAll
  sent_id = 0
end

function search(line)
  sent_id = sent_id + 1
  local cleaned_tokens, source_features_str = extract_features(line)
  local cleaned_line = table.concat(cleaned_tokens, ' ')
  print('SENT ' .. sent_id .. ': ' ..line)
  local source, source_str
  local source_features = features2featureidx(source_features_str, feature2idx_src, model_opt.start_symbol)
  if model_opt.use_chars_enc == 0 then
    source, source_str = sent2wordidx(cleaned_line, word2idx_src, model_opt.start_symbol)
  else
    source, source_str = sent2charidx(cleaned_line, char2idx, model_opt.max_word_l, model_opt.start_symbol)
  end
  if gold then
    target, target_str = sent2wordidx(gold[sent_id], word2idx_targ, 1)
  end
  state = State.initial(START)
  pred, pred_score, attn, gold_score, all_sents, all_scores, all_attn = generate_beam(model,
    state, opt.beam, MAX_SENT_L, source, source_features, target)
  pred_score_total = pred_score_total + pred_score
  pred_words_total = pred_words_total + #pred - 1
  pred_sent = wordidx2sent(pred, idx2word_targ, source_str, attn, true)

  print('PRED ' .. sent_id .. ': ' .. pred_sent)
  if gold ~= nil then
    print('GOLD ' .. sent_id .. ': ' .. gold[sent_id])
    if opt.score_gold == 1 then
      print(string.format("PRED SCORE: %.4f, GOLD SCORE: %.4f", pred_score, gold_score))
      gold_score_total = gold_score_total + gold_score
      gold_words_total = gold_words_total + target:size(1) - 1
    end
  end

  nbests = {}

  if opt.n_best > 1 then
    for n = 1, opt.n_best do
      pred_sent_n = wordidx2sent(all_sents[n], idx2word_targ, source_str, all_attn[n], false)
      local out_n = string.format("%d ||| %s ||| %.4f", n, pred_sent_n, all_scores[n])
      print(out_n)
      nbests[n] = out_n
    end
  end

  print('')

  return pred_sent, nbests
end

function getOptions()
  return opt
end

return {
  init = init,
  search = search,
  getOptions = getOptions
}
