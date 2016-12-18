--[[
  --Create a convolutional neural network for sentence classification.
]]--

V = { ["*padding*"] = 1, ["I"] = 2, ["am"] = 3, ["a"] = 4, ["he"] = 5, 
      ["it"] = 6, ["dog"] = 7, ["is"] = 8, ["she"] = 9 }
nV = 10

function make_data(sent, n, start_pad)
  out = {}
  for i = 1, start_pad do
    v = V["*padding*"]
    table.insert(out, v)
  end
  for i = 1, n - start_pad do
    if i <= #sent then
      v = V[sent[i]]
    else
      v = V["*padding*"]
    end
    table.insert(out, v)
  end
  return out
end

in_data = {}
out_data = {}

table.insert(in_data, make_data({"I", "am", "a", "dog"}, 10, 3))
table.insert(out_data, 1)
table.insert(in_data, make_data({"he", "is", "a", "dog"}, 10, 3))
table.insert(out_data, 2)
table.insert(in_data, make_data({"she", "is", "a", "dog"}, 10, 3))
table.insert(out_data, 2)
table.insert(in_data, make_data({"it", "is", "a", "dog"}, 10, 3))
table.insert(out_data, 2)

X = torch.DoubleTensor(in_data)
y = torch.DoubleTensor(out_data)
nY = 2

print(in_data)
print(out_data)
print(X)
print(y)