from onnx_t5 import OnnxT5
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

tokenizer = AutoTokenizer.from_pretrained("t5-small")

enc = tokenizer("translate English to French: This is fantastic!", return_tensors="pt")
onnx_model = OnnxT5(model_name_or_path="t5-small", onnx_path="onnx_models")
tokens = onnx_model.forward(enc['input_ids'], enc['attention_mask'], num_beams=2, use_cache=True)
# tokens = onnx_model.generate(**enc, num_beams=2, use_cache=True) # same HF's generate method
print(tokenizer.batch_decode(tokens))

script_model = torch.jit.script(onnx_model)
tokens = script_model.forward(enc['input_ids'], enc['attention_mask'], num_beams=2, use_cache=True)
print(tokenizer.batch_decode(tokens))


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