from onnx_t5 import OnnxT5_Full, fix_pretrained_model_weight, SimplifiedGenerator
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
import torch

tokenizer = AutoTokenizer.from_pretrained("t5-small")

#"summarize: studies have shown that owning a dog is good for you "
#"translate English to French: This is fantastic!"
translate_str = "translate English to French: This is fantastic! One in the name, number one in the game. "
translate_str_2 = "translate English to French: The most simple ones are presented here, showcasing usage for tasks such as question answering, sequence classification, named entity recognition and others."
summarize_str = "summarize: The most simple ones are presented here, showcasing usage for tasks such as question answering, sequence classification, named entity recognition and others."
enc = tokenizer(translate_str, return_tensors="pt")

# ONNX beam search baseline
model = SimplifiedGenerator(model_name_or_path="t5-small", onnx_path="onnx_models")
outputs = model.forward(enc['input_ids'], enc['attention_mask'], 2)  # torch.tensor(2, dtype=torch.long)) # NOTE: num_beams has to be int, checked in code
print("Simplified generator outputs:", outputs)
print("Simplified generator: ", tokenizer.batch_decode(outputs))

pt_tokens = []
script_model = torch.jit.script(model)

def run_pt(message):
    enc = tokenizer(message, return_tensors="pt")
    tokens = script_model.forward(enc['input_ids'], enc['attention_mask'], num_beams=2)
    pt_tokens.append(tokens)
    print('Scripted Simplified generator: ', tokenizer.batch_decode(tokens))
    print('Scripted Simplified generator graph: ', script_model.forward.graph)

run_pt(translate_str)
run_pt(translate_str_2)
run_pt(summarize_str)

import numpy as np
import onnxruntime
sess = onnxruntime.InferenceSession('model.onnx')

ort_tokens = []
def run_ort(message):
    enc = tokenizer(message, return_tensors="pt")
    ort_out = sess.run(None, {
        'input_ids': enc['input_ids'].cpu().numpy(),
        'attention_mask': enc['attention_mask'].cpu().numpy(),
        'num_beams': np.array(2, dtype=np.long),
    })

    ort_tokens.append(ort_out)
    print("ONNX Model: ", tokenizer.batch_decode(torch.tensor(ort_out[0])))


run_ort(translate_str)
run_ort(translate_str_2)
run_ort(summarize_str)


print('pt tokens:', pt_tokens)
print('ort tokens:', ort_tokens)