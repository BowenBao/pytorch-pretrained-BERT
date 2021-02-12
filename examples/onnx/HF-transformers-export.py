import time
import torch
from onnx import numpy_helper
from transformers import (
    ProphetNetConfig,
    ProphetNetModel,
    ProphetNetTokenizer,
    ProphetNetEncoder,
    ProphetNetDecoder,
    RobertaModel,
    RobertaTokenizer,
    XLMProphetNetConfig,
    XLMProphetNetModel,
    XLMProphetNetTokenizer,
    XLMRobertaModel,
    XLMRobertaTokenizer,
)
import numpy as np
import os
import onnx
import onnxruntime
from python.register_custom_ops_pytorch_exporter import register_custom_op

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
        if isinstance(d, torch.Tensor):
            d = d.cpu().numpy()
        else:
            d = d.last_hidden_state.cpu().numpy()
        save_tensor_proto(os.path.join(test_data_dir, '{0}_{1}.pb'.format(prefix, i)), names[i], d)


def save_model(name, model, inputs, outputs, input_names=None, output_names=None, **kwargs):
    if hasattr(model, 'train'):
        model.train(False)
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

    print([i.shape for i in inputs_flatten])

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
    print('start export*****')
    torch.onnx.export(model, inputs, model_dir, verbose=True, input_names=input_names,
                      output_names=output_names, example_outputs=outputs, custom_opsets={'com.microsoft': 1}, **kwargs)
    print('end export*****')
    test_data_dir = os.path.join(dir, data_dir_name)
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)

    save_data(test_data_dir, "input", input_names, inputs_flatten)
    save_data(test_data_dir, "output", output_names, outputs_flatten)

    return model_dir, test_data_dir


def inference(file, inputs, outputs):
    inputs_flatten = flatten(inputs)
    inputs_flatten = update_flatten_list(inputs_flatten, [])
    outputs_flatten = flatten(outputs)
    outputs_flatten = update_flatten_list(outputs_flatten, [])

    sess = onnxruntime.InferenceSession(file)
    ort_inputs = dict((sess.get_inputs()[i].name, to_numpy(input)) for i, input in enumerate(inputs_flatten))
    res = sess.run(None, ort_inputs)
    import time
    count = 10
    time_start = time.time()
    for i in range(count):
        sess.run(None, ort_inputs)
    time_end = time.time()
    print('avg ort time =' + str((time_end - time_start)/count))

    if outputs is not None:
        print("== Checking model output ==")
        print(" Got {} output tensors.".format(len(outputs_flatten)))
        [np.testing.assert_allclose(to_numpy(output), res[i], rtol=1e-02, atol=1e-04) for i, output in enumerate(outputs_flatten)]
        print("== Done ==")


'''
For ProphetNetModel/XLMProphetNetModel, run PYTHONPATH=/home/david/dev/onnxruntime/tools/ python HF-transformers-export.py
'''
def transformers_test():
    register_custom_op()
    MODELS = [
        ('ProphetNet', ProphetNetModel, ProphetNetTokenizer, 'microsoft/prophetnet-large-uncased'),
        # ('XLMProphetNet', XLMProphetNetModel, XLMProphetNetTokenizer, 'microsoft/xprophetnet-large-wiki100-cased'),
        # ('Roberta', RobertaModel, RobertaTokenizer, 'roberta-base'),
        # ('XLMRoberta', XLMRobertaModel, XLMRobertaTokenizer, 'xlm-roberta-base'),
    ]

    for model_name, model_class, tokenizer_class, pretrained_weights in MODELS:
        # Load pretrained model/tokenizer
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights)
        model.eval()

        text = """The rapid growth of submissions and the increasing popularity of preprints have caused several problems to the current ACL reviewing system. To address these problems, the ACL Committee on Reviewing has been working on two proposals for reforming the reviewing system of ACL-related conferences: short-term and long-term. The following document presents the short-term proposals: https://www.aclweb.org/adminwiki/index.php?title=Short-Term_Reform_Propo... It consists of four complementary actions that can be realistically implemented to improve the ACL review process in the near future (while the committee continues to investigate changes that require a longer lead time). These actions address several of the problems identified in the proposal. The ACL Executive Committee has adopted these proposals. We hope that their implementation will have a quick positive impact on reviewing at ACL conferences."""

        # Inputs are provided through numpy array
        model_inputs = tokenizer.encode_plus(text=text,
                                             text_pair=text,
                                             add_special_tokens=True,
                                             max_length=16,
                                             pad_to_max_length=False,
                                             return_token_type_ids=False,
                                             return_attention_mask=True,
                                             return_overflowing_tokens=False,
                                             return_special_tokens_mask=False,
                                             return_tensors='pt',
                                             )

        if model_name in ['ProphetNet', 'XLMProphetNet']:
            model_inputs['decoder_input_ids'] = tokenizer("The rapid growth of submissions", return_tensors="pt").input_ids

        inputs = tuple(model_inputs.values())

        with torch.no_grad():
            output_1 = model(return_dict=True, **model_inputs)
            print('output:', output_1)

        if model_name == 'XLMProphetNet':
            use_external_data_format = True
        else:
            use_external_data_format = False

        input_names = ['input_ids', 'attention_mask']
        # dynamic_axes = {'input_ids': {0: 'batch_size', 1: 'answers', 2: 'sequence'},
        #                 'attention_mask': {0: 'batch_size', 1: 'answers', 2: 'sequence'},
        #                 'score': {0: 'batch_size', 1: 'answers', 2: 'sequence'},
        #                 }
        dynamic_axes = {'input_ids': {0: 'batch_size', 1: 'sequence'},
                        'attention_mask': {0: 'batch_size', 1: 'sequence'},
                        'score': {0: 'batch_size', 1: 'sequence'},
                        }

        if model_name in ['ProphetNet', 'XLMProphetNet']:
            input_names.append('decoder_input_ids')
            dynamic_axes['decoder_input_ids'] = {0: 'batch_size', 1: 'target_sequence'}

        model_dir, data_dir = save_model(model_name, model.cpu(), inputs, output_1,
                                         opset_version=12,
                                         input_names=input_names,
                                         output_names=['score'],  # the model's output names
                                         dynamic_axes=dynamic_axes,
                                         use_external_data_format=use_external_data_format)

        text = """The rapid growth of submissions and the increasing popularity of preprints have caused several problems to the current ACL reviewing system. To address these problems, the ACL Committee on Reviewing has been working on two proposals for reforming the reviewing system of ACL-related conferences: short-term and long-term. The following document presents the short-term proposals: https://www.aclweb.org/adminwiki/index.php?title=Short-Term_Reform_Propo... It consists of four complementary actions that can be realistically implemented to improve the ACL review process in the near future (while the committee continues to investigate changes that require a longer lead time). These actions address several of the problems identified in the proposal. The ACL Executive Committee has adopted these proposals. We hope that their implementation will have a quick positive impact on reviewing at ACL conferences."""

        # Inputs are provided through numpy array
        model_inputs = tokenizer.encode_plus(text=text,
                                             text_pair=text,
                                             add_special_tokens=True,
                                             max_length=16,
                                             pad_to_max_length=False,
                                             return_token_type_ids=False,
                                             return_attention_mask=True,
                                             return_overflowing_tokens=False,
                                             return_special_tokens_mask=False,
                                             return_tensors='pt',
                                             )

        if model_name in ['ProphetNet', 'XLMProphetNet']:
            model_inputs['decoder_input_ids'] = tokenizer("Studies show that", return_tensors="pt").input_ids

        inputs = tuple(model_inputs.values())

        with torch.no_grad():
            output_1 = model(return_dict=True, **model_inputs)

        count = 10
        time_start = time.time()
        for i in range(count):
            with torch.no_grad():
                output_1 = model(return_dict=True, **model_inputs)
        time_end = time.time()
        print('avg pytorch time =' + str((time_end - time_start) / count))

        inference(model_dir, inputs, output_1[0])


transformers_test()
