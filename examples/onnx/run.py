from onnx_t5 import SimplifiedGenerator, fix_pretrained_model_weight
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
import numpy as np
import os
from onnx import numpy_helper


data_dir_name = 'test_data_set_0'


def flatten(inputs):
    return [[flatten(i) for i in inputs] if isinstance(inputs, (list, tuple)) else inputs]


def update_flatten_list(inputs, res_list):
    for i in inputs:
        res_list.append(i) if not isinstance(i, (list, tuple)) else update_flatten_list(i, res_list)
    return res_list


def to_numpy(x):
    if type(x) is not np.ndarray:
        x = x.detach().cpu().numpy() if x.requires_grad else x.cpu().numpy()
    return x


def save_tensor_proto(file_path, name, data):
    tp = numpy_helper.from_array(data)
    tp.name = name

    with open(file_path, 'wb') as f:
        f.write(tp.SerializeToString())


def save_data(test_data_dir, prefix, names, data_list):
    if isinstance(data_list, torch.autograd.Variable) or isinstance(data_list, torch.Tensor):
        data_list = [data_list]
    for i, d in enumerate(data_list):
        if isinstance(d, int):
            d = np.array(d)
        else:
            d = d.data.cpu().numpy()
        save_tensor_proto(os.path.join(test_data_dir, '{0}_{1}.pb'.format(prefix, i)), names[i], d)


def save_model(name, model, inputs, outputs, input_names=None, output_names=None, **kwargs):
    model.eval()
    dir = './'
    if not os.path.exists(dir):
        os.makedirs(dir)
    dir = os.path.join(dir, 'test_' + name)
    if not os.path.exists(dir):
        os.makedirs(dir)

    inputs_flatten = flatten(inputs)
    inputs_flatten = update_flatten_list(inputs_flatten, [])
    outputs_flatten = flatten(outputs)
    outputs_flatten = update_flatten_list(outputs_flatten, [])

    if input_names is None:
        input_names = []
        for i, _ in enumerate(inputs_flatten):
            input_names.append('input' + str(i+1))
    else:
        np.testing.assert_equal(len(input_names), len(inputs_flatten),
                                "Number of input names provided is not equal to the number of inputs.")

    if output_names is None:
        output_names = []
        for i, _ in enumerate(outputs_flatten):
            output_names.append('output' + str(i+1))
    else:
        np.testing.assert_equal(len(output_names), len(outputs_flatten),
                                "Number of output names provided is not equal to the number of output.")

    model_dir = os.path.join(dir, 'model.onnx')
    torch.onnx.export(
        model,
        (enc['input_ids'], enc['attention_mask'], 2),
        'model.onnx',
        opset_version=15,
        verbose=True,
        input_names=['input_ids', 'attention_mask', 'num_beams'],
        output_names=['tokens'],
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'sequence_in'},
            'attention_mask': {0: 'batch', 1: 'sequence_in'},
            'tokens': {0: 'batch', 1: 'sequence_out'}
        },
        example_outputs=tokens)

    test_data_dir = os.path.join(dir, data_dir_name)
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)

    # print(inputs_flatten)
    print(outputs_flatten)
    save_data(test_data_dir, "input", input_names, inputs_flatten)
    save_data(test_data_dir, "output", output_names, outputs_flatten)

    return model_dir, test_data_dir


tokenizer = AutoTokenizer.from_pretrained("t5-small")
translate_str = "translate English to French: This is fantastic! One in the name, number one in the game. "
enc = tokenizer(translate_str, return_tensors="pt")

# baseline
enc = tokenizer(translate_str, return_tensors="pt")
# model = T5ForConditionalGeneration.from_pretrained('t5-small')
# fix_pretrained_model_weight(model)
# outputs = model.generate(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'], num_beams=2, use_cache=True)
# print("output1 : ", outputs)
# print('Baseline: ', tokenizer.batch_decode(outputs))

# ONNX beam search baseline
model = SimplifiedGenerator(model_name_or_path="t5-small", onnx_path="onnx_models")
# outputs = model.generate(input_ids=enc['input_ids'], num_beams=2)
# print("Simplified generator outputs:", outputs)
# print("Simplified generator: ", tokenizer.batch_decode(outputs))

script_model = torch.jit.script(model)
tokens = script_model(enc['input_ids'], enc['attention_mask'], num_beams=2)
input_names = ['input_ids', 'attention_mask', 'num_beams']
output_names = ['tokens']
print('Scripted Simplified generator: ', tokenizer.batch_decode(tokens))

inputs = tuple((enc['input_ids'], enc['attention_mask'], 2))
outputs = tokens
save_model("t5", script_model, inputs, outputs, input_names=input_names, output_names=output_names)
