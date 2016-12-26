--[[
  --Create a convolutional neural network for sentence classification.
]]--

require('nn')
require('cudnn')
require('cutorch')
require('optim')

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

d = 10

model = nn.Sequential()
matrixV = nn.LookupTable(nV, d)
model:add(matrixV)

nd = 10
h = 3

conv = nn.Sequential()
conv:add(nn.TemporalConvolution(d, nd, h))
conv:add(nn.ReLU())
conv:add(nn.Max(2))

model:add(conv)

logistic = nn.Sequential()

logistic:add(nn.Linear(nd, nY))
logistic:add(nn.LogSoftMax())

model:add(logistic)

output = model:forward(X)

print(output)

criterion = nn.ClassNLLCriterion()

cudnn_model = nn.Sequential()
matrixV = nn.LookupTable(nV, d)
model:add(matrixV)

nd = 10
h = 3
S = 10

conv = nn.Sequential()
conv:add(nn.Reshape(1, S, d, false))
conv:add(cudnn.SpatialConvolution(1, nd, d, h))
conv:add(nn.Reshape(nd, S-h+1, false))
conv:add(cudnn.ReLU())
conv:add(nn.Max(3))

cudnn_model:add(conv)

logistic = nn.Sequential()

logistic:add(nn.Linear(nd, nY))
logistic:add(cudnn.LogSoftMax())

cudnn_model:add(logistic)

criterion = nn.ClassNLLCriterion()

cudnn_model:cuda()
criterion:cuda()

model:reset()
model:training()

params, grads = model:getParameters()

config = {rho = 0.95, eps = 1e-6}
state = {}

for epoch = 1, 20 do
  func = function(x)
    if x ~= params then
      params:copy(x)
    end
    grads:zero()
    
    out = model:forward(X)
    err = criterion:forward(out, y)
    
    dout = criterion:backward(out, y)
    model:backward(X, dout)
    
    return err, grads
  end
  
  optim.adadelta(func, params, config, state)
  print("Epoch:", epoch, err)
end

