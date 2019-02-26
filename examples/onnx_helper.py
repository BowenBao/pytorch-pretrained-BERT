import onnx
import torch
import os
import numpy as np

# data dir
data_dir = 'test_data_set_0'

def SaveTensorProto(file_path, name, data):
    tp = onnx.TensorProto()
    tp.name = name
    print(name)
    for d in data.shape:
        tp.dims.append(d)

    if data.dtype == np.int64:
        tp.data_type = onnx.TensorProto.INT64
    else:
        tp.data_type = onnx.TensorProto.FLOAT
    print('TensorProto dims:')
    print(tp.dims)
    tp.raw_data = data.tobytes()
    with open(file_path, 'wb') as f:
        f.write(tp.SerializeToString())
        
def SaveData(test_data_dir, prefix, data_list):
    if isinstance(data_list, torch.autograd.Variable) or isinstance(data_list, torch.Tensor):
        data_list = [data_list]
    for i, d in enumerate(data_list):
        d = d.data.cpu().numpy()
        SaveTensorProto(os.path.join(test_data_dir, '{0}_{1}.pb'.format(prefix, i)), prefix + str(i+1), d)

def Save(dir, name, model, inputs, outputs, input_names = ['input1'], output_names = ['output1']):
    model.train(False)
    if not os.path.exists(dir):
        os.makedirs(dir)
    dir = os.path.join(dir, 'test_' + name)
    if not os.path.exists(dir):
        os.makedirs(dir)

    print('inputs:')
    print(inputs)
    print('outputs:')
    print(outputs)

    def f(t):
        return [f(i) for i in t] if isinstance(t, (list, tuple)) else t
    def g(t, res):
        for i in t:
            res.append(i) if not isinstance(i, (list, tuple)) else g(i, res)
        return res

    torch.onnx.export(model, tuple(inputs), os.path.join(dir, 'model.onnx'), verbose=True, input_names=input_names, output_names=output_names)

    test_data_dir = os.path.join(dir, data_dir)
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)

    inputs = f(inputs)
    inputs = g(inputs, [])
    outputs = f(outputs)
    outputs = g(outputs, [])

    print('inputs:')
    [print(i.dtype) for i in inputs]
    print('outputs: ')
    [print(i.dtype) for i in outputs]


    # print(outputs[0][0][0])
    # for i in range(1000):
    #     outputs[0][0][i] = 123
    # print(outputs[0][0][0])


    SaveData(test_data_dir, 'input', inputs)
    SaveData(test_data_dir, 'output', outputs)