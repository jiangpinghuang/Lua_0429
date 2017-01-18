local beam = require 'util.beam'

function main()
  beam:init(arg)
  local opt = beam:getOption()
  
  assert(path.exists(opt.src_file), 'src_file does not exist.')
  
  local file = io.open(opt.src_file, 'r')
  local out = io.open(opt.output_file, 'w')
  for line in file:lines() do
    result, nbest = beam.search(line)
    out:write(result .. '\n')
    
    for n = 1, #nbest do
      out:write(nbest[n] .. '\n')
    end
  end
  
  print(string.format("pred avg score: %.4f, pred ppl: %.4f", pred_score_total / pred_word_total,
        math.exp(-pred_score_total / pred_word_total)))
  if opt.gold_score == 1 then
    print(string.format("gold avg score: %.4f, gold ppl: %.4f", gold_score_total / gold_word_total,
          math.exp(-gold_score_total / gold_word_total)))
  end
  out:close()
end

main()
