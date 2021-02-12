import inspect
import logging
import os
from pathlib import Path

import torch
from psutil import cpu_count
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
from transformers.generation_utils import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, Seq2SeqLMOutput

# Constants from the performance optimization available in onnxruntime
# It needs to be done before importing onnxruntime
os.environ["OMP_NUM_THREADS"] = str(cpu_count(logical=True))
os.environ["OMP_WAIT_POLICY"] = "ACTIVE"

from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions, get_all_providers

ONNX_CACHE_DIR = Path(os.path.dirname(__file__)).parent.joinpath(".onnx")
logger = logging.getLogger(__name__)


def create_t5_encoder_decoder(model="t5-base"):
    """Generates an encoder and a decoder model with a language model head from a pretrained huggingface model
    Args:
        model (str): Name of a pretrained model, or path to a pretrained / finetuned version of T5
    Returns:
        t5_encoder: pytorch t5 encoder with a wrapper to output only the hidden states
        t5_decoder: pytorch t5 decoder with a language modeling head
    """

    # T5 is an encoder / decoder model with a language modeling head on top.
    # We need to separate those out for efficient language generation
    if isinstance(model, str):
        model = T5ForConditionalGeneration.from_pretrained(model)

    encoder = model.encoder
    decoder = model.decoder
    lm_head = model.lm_head

    t5_encoder = T5Encoder(encoder).eval()
    t5_decoder = T5Decoder(decoder, model.config).eval()
    t5_decoder_with_past = T5Decoder(decoder, model.config).eval()
    t5_lm_head = T5LMHead(lm_head).eval()
    return t5_encoder, t5_decoder, t5_decoder_with_past, t5_lm_head


def generate_onnx_representation(model, encoder_path, decoder_path, lm_path):
    """Exports a given huggingface pretrained model, or a given model and tokenizer, to onnx
    Args:
        pretrained_version (str): Name of a pretrained model, or path to a pretrained / finetuned version of T5
        output_prefix (str): Path to the onnx file
    """

    simplified_encoder, decoder, decoder_with_past, lm_head = create_t5_encoder_decoder(model)

    # Example sequence
    tok = T5Tokenizer.from_pretrained(model)
    enc = tok("42 is the answer", return_tensors="pt")
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    traced_simplified_encoder = torch.jit.trace(simplified_encoder, (input_ids, attention_mask))
    encoder_out = simplified_encoder(input_ids, attention_mask)

    # Exports to ONNX
    _ = torch.onnx._export(
        traced_simplified_encoder,
        (input_ids, attention_mask),
        encoder_path,
        export_params=True,
        opset_version=12,
        input_names=["input_ids", "attention_mask"],
        output_names=["encoder_hidden_states"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "encoder_hidden_states": {0: "batch", 1: "sequence"},
        },
        example_outputs=encoder_out,
    )

    print("Test 0: test encoder")
    encoder_session = InferenceSession(encoder_path)
    ort_encoder_inputs = {"input_ids": input_ids.detach().cpu().numpy(), "attention_mask": attention_mask.detach().cpu().numpy()}
    ort_encoder_outputs = encoder_session.run(None, ort_encoder_inputs)
    import numpy
    for i in range(len(encoder_out)):
        print('i all close: ', i, numpy.allclose(encoder_out[i].detach().cpu().numpy(), ort_encoder_outputs[i], rtol=1e-03, atol=1e-05))


    decoder_input_ids = torch.zeros((input_ids.shape[0], 1), dtype=input_ids.dtype, device=input_ids.device)
    attention_mask = input_ids.new_ones(input_ids.shape)

    has_past = torch.tensor(False)
    decoder_outs = decoder(decoder_input_ids, encoder_out, attention_mask, has_past)
    # flatten decoder outs
    d = [decoder_outs[0]]
    for t in decoder_outs[1]:
        d = d + list(t)
    decoder_outs_flatten = tuple(d)
    # decoder_outs_flatten = decoder_outs

    input_states_names = []
    output_states_names = []
    decoder_dynamic_axes = {
        "input_ids": {0: "batch", 1: "sequence"},
        "encoder_hidden_states": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1:"sequence"},
        "hidden_states": {0: "batch", 1: "sequence"},
    }
    for i, s in enumerate(decoder_outs_flatten[1:]):
        input_states_names.append(f'state_{i}')
        output_states_names.append(f'out_state_{i}')
        decoder_dynamic_axes[input_states_names[-1]] = {0: "batch", 2: f"sequence_in_{i}"}
        decoder_dynamic_axes[output_states_names[-1]] = {0: "batch", 2: f"sequence_out_{i}"}

    print('export decoder')
    traced_decoder = torch.jit.trace(decoder, (decoder_input_ids, encoder_out, attention_mask, has_past))

    # _ = torch.onnx._export(
    #     # decoder,
    #     traced_decoder,
    #     (decoder_input_ids, encoder_out, attention_mask),
    #     decoder_path,
    #     opset_version=12,
    #     input_names=["input_ids", "encoder_hidden_states", "attention_mask"],
    #     output_names=["hidden_states"] + output_states_names,
    #     dynamic_axes = decoder_dynamic_axes,
    #     example_outputs = decoder_outs,
    #     verbose=True,
    # )
    has_past = torch.tensor(True)
    print('decoder out:', len(decoder_outs_flatten), [v.shape for v in decoder_outs_flatten])
    decoder_outs_with_past = decoder_with_past(decoder_input_ids, encoder_out, attention_mask, has_past, *decoder_outs_flatten[1:])
    # NOTE: flatten tuple output.
    d = [decoder_outs_with_past[0]]
    for t in decoder_outs_with_past[1]:
        d = d + list(t)
    decoder_outs_with_past = tuple(d)
    print('decoder out with past: ', len(decoder_outs_with_past), [v.shape for v in decoder_outs_with_past])

    print('trace decoder with past')
    traced_decoder = torch.jit.trace(decoder_with_past, (decoder_input_ids, encoder_out, attention_mask, has_past, *decoder_outs_flatten[1:]))

    print('export decoder with past')
    _ = torch.onnx._export(
        traced_decoder,
        (decoder_input_ids, encoder_out, attention_mask, has_past, *decoder_outs_flatten[1:]),
        decoder_path,
        opset_version=12,
        input_names=["input_ids", "encoder_hidden_states", "attention_mask", "has_past"] + input_states_names,
        output_names=["hidden_states"] + output_states_names,
        dynamic_axes = decoder_dynamic_axes,
        example_outputs = decoder_outs,
        verbose=True,
    )

    # import onnx
    # decoder_with_past_model = onnx.load(decoder_path)

    # import pdb; pdb.set_trace()

    print("Test 1: test decoder with past provided")
    has_past = torch.tensor(True)
    ort_decoder_inputs = {k:v.detach().cpu().numpy() for k, v in zip(["input_ids", "encoder_hidden_states", "attention_mask", "has_past"] + input_states_names, [decoder_input_ids, encoder_out, attention_mask, has_past, *decoder_outs_flatten[1:]])}
    decoder_with_past_session = InferenceSession(decoder_path)
    ort_decoder_outputs = decoder_with_past_session.run(None, ort_decoder_inputs)

    import numpy
    for i in range(len(decoder_outs_with_past)):
        print('i all close: ', i, numpy.allclose(decoder_outs_with_past[i].detach().cpu().numpy(), ort_decoder_outputs[i], rtol=1e-03, atol=1e-05))

    print("Test 2: test decoder with past not provided")
    has_past = torch.tensor(False)
    unused_decoder_state_inputs = [torch.zeros(v.shape) for v in decoder_outs_flatten[1:]]
    ort_decoder_inputs = {k:v.detach().cpu().numpy() for k, v in zip(["input_ids", "encoder_hidden_states", "attention_mask", "has_past"] + input_states_names, [decoder_input_ids, encoder_out, attention_mask, has_past, *unused_decoder_state_inputs])}
    ort_decoder_outputs = decoder_with_past_session.run(None, ort_decoder_inputs)

    import numpy
    for i in range(len(decoder_outs_flatten)):
        print('i all close: ', i, numpy.allclose(decoder_outs_flatten[i].detach().cpu().numpy(), ort_decoder_outputs[i], rtol=1e-03, atol=1e-05))

    # decoder_out, past_key_values = decoder(input_ids, encoder_out)

    _ = torch.onnx._export(
        lm_head,
        decoder_outs_flatten[0],
        lm_path,
        export_params=True,
        opset_version=12,
        input_names=["decoder_output"],
        output_names=["lm_logits"],
        dynamic_axes={
            "decoder_output": {0: "batch", 1: "sequence"},
            "lm_logits": {0: "batch", 1: "sequence"},
        },
    )


def create_model_for_provider(model_path: str, provider: str) -> InferenceSession:

    assert provider in get_all_providers(), f"provider {provider} not found, {get_all_providers()}"

    # Few properties that might have an impact on performances (provided by MS)
    options = SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    # Load the model as a graph and prepare the CPU backend
    session = InferenceSession(model_path, options, providers=[provider])
    session.disable_fallback()

    return session


class T5Encoder(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]


class T5Decoder(torch.nn.Module):
    def __init__(self, decoder, config):
        super().__init__()
        self.decoder = decoder
        self.config = config

    def forward(self, input_ids, encoder_hidden_states, attention_mask, has_past, *past):
        # NOTE: not needed anymore
        # past_arg_key = (
        #     "past_key_value_states"
        #     if "past_key_value_states" in inspect.getfullargspec(self.decoder.forward).args
        #     else "past_key_values"
        # )
        # past_arg = {past_arg_key: past_key_values}
        past_key_values = tuple()
        print('len of past: ', len(past))
        for i in range(len(past) // 4):
            past_key_values = past_key_values + (past[i*4:i*4+4],)
            print('len of tuple:', len(past_key_values[-1]))
        if len(past_key_values) == 0:
            past_key_values = None
        else:
            print('len of past tuple: ', len(past_key_values))

        decoder_output = self.decoder(
            input_ids=input_ids,
            encoder_attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            use_cache=True,
            return_dict=True,
            past_key_values=past_key_values,
            has_past=has_past,
        )
        past_key_values = decoder_output.past_key_values
        sequence_output = decoder_output.last_hidden_state
        sequence_output = sequence_output * (self.config.d_model ** -0.5)

        # # NOTE: flatten tuple output.
        # if past_key_values is not None:
        #     d = [sequence_output]
        #     for t in past_key_values:
        #         d = d + list(t)
        #     return tuple(d)

        return sequence_output, past_key_values


class T5LMHead(torch.nn.Module):
    def __init__(self, lm_head):
        super().__init__()
        self.lm_head = lm_head

    def forward(self, decoder_output):
        return self.lm_head(decoder_output)


class OnnxT5(GenerationMixin, torch.nn.Module):
    def __init__(self, model_name_or_path, onnx_path):
        super().__init__()
        self.device = torch.device('cpu')

        self.model_name_or_path = Path(model_name_or_path)
        self.onnx_path = Path(onnx_path)
        self.mode_base_name = self.model_name_or_path.stem
        self.encoder_path = self.onnx_path.joinpath(f"{self.mode_base_name}_encoder.onnx")
        self.decoder_path = self.onnx_path.joinpath(f"{self.mode_base_name}_decoder.onnx")
        self.lm_head_path = self.onnx_path.joinpath(f"{self.mode_base_name}_lm_head.onnx")

        # if not (self.encoder_path.exists() and self.lm_head_path.exists()):
        self._export_onnx_graph()

        self.encoder_sess = create_model_for_provider(self.encoder_path.as_posix(), "CPUExecutionProvider")
        self.lm_sess = create_model_for_provider(self.lm_head_path.as_posix(), "CPUExecutionProvider")

        self.config = T5Config.from_pretrained(model_name_or_path)
        decoder = T5ForConditionalGeneration.from_pretrained(model_name_or_path).decoder
        self.decoder = T5Decoder(decoder, self.config).eval()

        self._warmup_onnx_graph()

    @torch.no_grad()
    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs=None,
        past_key_values=None,
        **kwargs,
    ):
        if input_ids is not None:
            return self._encoder_forward(input_ids=input_ids, attention_mask=attention_mask)

        decoder_output, past = self.decoder(decoder_input_ids, encoder_outputs, attention_mask, past_key_values)

        inputs = {"decoder_output": decoder_output.cpu().detach().numpy()}
        lm_logits = self.lm_sess.run(None, inputs)[0]
        lm_logits = torch.from_numpy(lm_logits)
        return Seq2SeqLMOutput(logits=lm_logits, past_key_values=past)

    def _encoder_forward(self, input_ids=None, attention_mask=None):
        inputs = {
            "input_ids": input_ids.cpu().detach().numpy(),
            "attention_mask": attention_mask.cpu().detach().numpy(),
        }
        last_hidden_state = self.encoder_sess.run(None, inputs)[0]
        last_hidden_state = torch.from_numpy(last_hidden_state)
        return BaseModelOutputWithPast(last_hidden_state=last_hidden_state)

    def get_encoder(self):
        return self

    def get_output_embeddings(self):
        return self

    def prepare_inputs_for_generation(self, input_ids, attention_mask, use_cache, encoder_outputs, past=None, **kwargs):
        if past is not None:
            input_ids = input_ids[:, -1:]
        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs.last_hidden_state,
            "attention_mask": attention_mask,
            "use_cache": True,
        }

    def parameters(self):
        return iter(torch.tensor([42, 42]))

    def _export_onnx_graph(self):
        self.onnx_path.mkdir(parents=True, exist_ok=True)
        generate_onnx_representation(
            self.model_name_or_path.as_posix(), self.encoder_path.as_posix(), self.decoder_path.as_posix(), self.lm_head_path.as_posix()
        )

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

    def _warmup_onnx_graph(self):
        input_ids = torch.ones(1, 512, dtype=torch.long)
        attention_mask = torch.ones(1, 512, dtype=torch.long)
        for _ in range(10):
            encoder_outputs = self._encoder_forward(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state

        decoder_output, _ = self.decoder(input_ids, encoder_outputs, attention_mask)
        inputs = {"decoder_output": decoder_output.cpu().detach().numpy()}
        for _ in range(10):
            self.lm_sess.run(None, inputs)

    def forward(self, input_ids, attention_mask, num_beams, use_cache):
        return self.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=num_beams, use_cache=use_cache)
