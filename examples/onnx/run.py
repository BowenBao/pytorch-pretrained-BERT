from onnx_t5 import SimplifiedGenerator, fix_pretrained_model_weight
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch

tokenizer = AutoTokenizer.from_pretrained("t5-small")
translate_str = "translate English to French: This is fantastic! One in the name, number one in the game. "
enc = tokenizer(translate_str, return_tensors="pt")

# baseline
enc = tokenizer(translate_str, return_tensors="pt")
model = T5ForConditionalGeneration.from_pretrained('t5-small')
fix_pretrained_model_weight(model)
outputs = model.generate(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'], num_beams=2, use_cache=True)
print('Baseline: ', tokenizer.batch_decode(outputs))

# ONNX beam search baseline
model = SimplifiedGenerator(model_name_or_path="t5-small", onnx_path="onnx_models")
outputs = model.forward(enc['input_ids'], enc['attention_mask'], 2)
print("Simplified generator outputs:", outputs)
print("Simplified generator: ", tokenizer.batch_decode(outputs))


script_model = torch.jit.script(model)
tokens = script_model.forward(enc['input_ids'], enc['attention_mask'], num_beams=2)
print('Scripted Simplified generator: ', tokenizer.batch_decode(tokens))

torch.onnx.export(
    script_model,
    (enc['input_ids'], enc['attention_mask'], 2),
    'model.onnx',
    opset_version=13,
    input_names=['input_ids', 'attention_mask', 'num_beams'],
    output_names=['tokens'],
    dynamic_axes={
        'input_ids': {0: 'batch', 1: 'sequence_in'},
        'attention_mask': {0: 'batch', 1: 'sequence_in'},
        'tokens': {0: 'batch', 1: 'sequence_out'}
    },
    example_outputs=tokens)

print('Complete exporting to onnx.')

import numpy as np
import onnxruntime
sess = onnxruntime.InferenceSession('model.onnx')

ort_out = sess.run(None, {
    'input_ids': enc['input_ids'].cpu().numpy(),
    'attention_mask': enc['attention_mask'].cpu().numpy(),
    'num_beams': np.array(2, dtype=np.long),
})

print(ort_out)
print("ONNX Model: ", tokenizer.batch_decode(torch.tensor(ort_out[0])))
