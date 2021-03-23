from onnx_t5 import OnnxT5_Full, fix_pretrained_model_weight, SimplifiedGenerator
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
import torch

tokenizer = AutoTokenizer.from_pretrained("t5-small")

"""
NOTE: tmp skip to speed-up debugging.
"""
#"summarize: studies have shown that owning a dog is good for you "
#"translate English to French: This is fantastic!"
translate_str = "translate English to French: This is fantastic! One in the name, number one in the game. "
# summarize_str = "summarize: The most simple ones are presented here, showcasing usage for tasks such as question answering, sequence classification, named entity recognition and others."
enc = tokenizer(translate_str, return_tensors="pt")
# onnx_model = OnnxT5_Full(model_name_or_path="t5-small", onnx_path="onnx_models")
# tokens = onnx_model.forward(enc['input_ids'], enc['attention_mask'], num_beams=2, use_cache=True)
# print('OnnxT5_full: ', tokenizer.batch_decode(tokens))

# baseline
# enc = tokenizer(translate_str, return_tensors="pt")
# model = T5ForConditionalGeneration.from_pretrained('t5-small')
# fix_pretrained_model_weight(model)
# outputs = model.generate(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'], num_beams=2, use_cache=True)
# print('Baseline: ', tokenizer.batch_decode(outputs))

# ONNX beam search baseline
model = SimplifiedGenerator(model_name_or_path="t5-small", onnx_path="onnx_models")
outputs = model.forward(enc['input_ids'], enc['attention_mask'], 2)  # torch.tensor(2, dtype=torch.long)) # NOTE: num_beams has to be int, checked in code
print("Simplified generator outputs:", outputs)
print("Simplified generator: ", tokenizer.batch_decode(outputs))



"""
"""

script_model = torch.jit.script(model)
tokens = script_model.forward(enc['input_ids'], enc['attention_mask'], num_beams=2)
print('Scripted Simplified generator: ', tokenizer.batch_decode(tokens))

print('Scripted Simplified generator graph: ', script_model.forward.graph)

torch.onnx.export(
    script_model,
    (enc['input_ids'], enc['attention_mask'], 2),
    'model.onnx',
    opset_version=12,
    input_names=['input_ids', 'attention_mask', 'num_beams'],
    output_names=['tokens'],
    dynamic_axes={
        'input_ids': {0: 'batch', 1: 'sequence_in'},
        'attention_mask': {0: 'batch', 1: 'sequence_in'},
        'tokens': {0: 'batch', 1: 'sequence_out'}
    },
    example_outputs=tokens)


### t5

# from transformers import T5Tokenizer, T5ForConditionalGeneration

# tokenizer = T5Tokenizer.from_pretrained('t5-small')
# model = T5ForConditionalGeneration.from_pretrained('t5-small')

# input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
# labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
# outputs = model(input_ids=input_ids, labels=labels)
# loss = outputs.loss
# logits = outputs.logits

# input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
# outputs = model.generate(input_ids, num_beams=2, use_cache=True)

# print(outputs)