import inspect
import logging
import os
from pathlib import Path

import torch
from torch.nn import functional as F

from psutil import cpu_count
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
from transformers.generation_utils import GenerationMixin, BeamSearchOutput
from transformers.modeling_outputs import BaseModelOutputWithPast, Seq2SeqLMOutput

from transformers.generation_beam_search import BeamScorer, BeamSearchScorer, BeamSearchScorerTS
from transformers.generation_logits_process import LogitsProcessorList, MinLengthLogitsProcessor
from transformers.file_utils import ModelOutput

from typing import Optional, Union, Iterable, Callable, List, Tuple

# Constants from the performance optimization available in onnxruntime
# It needs to be done before importing onnxruntime
os.environ["OMP_NUM_THREADS"] = str(cpu_count(logical=True))
os.environ["OMP_WAIT_POLICY"] = "ACTIVE"

from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions, get_all_providers

ONNX_CACHE_DIR = Path(os.path.dirname(__file__)).parent.joinpath(".onnx")
logger = logging.getLogger(__name__)

def flatten_past(past):
    # flatten decoder outs
    d = []
    for t in past:
        d = d + list(t)
    past_flatten = tuple(d)
    return past_flatten

def fix_pretrained_model_weight(model):
    for enc_block in model.encoder.block:
        for layer in enc_block.layer:
            try:
                layer.SelfAttention.project_m.k.weight = layer.SelfAttention.k.weight
                layer.SelfAttention.project_m.v.weight = layer.SelfAttention.v.weight
            except:
                pass
            try:
                layer.EncDecAttention.project_m.k.weight = layer.EncDecAttention.k.weight
                layer.EncDecAttention.project_m.v.weight = layer.EncDecAttention.v.weight
            except:
                pass
    for enc_block in model.decoder.block:
        for layer in enc_block.layer:
            try:
                layer.SelfAttention.project_m.k.weight = layer.SelfAttention.k.weight
                layer.SelfAttention.project_m.v.weight = layer.SelfAttention.v.weight
            except:
                pass
            try:
                layer.EncDecAttention.project_m.k.weight = layer.EncDecAttention.k.weight
                layer.EncDecAttention.project_m.v.weight = layer.EncDecAttention.v.weight
            except:
                pass

def create_t5_encoder_decoder(model="t5-base", flatten_decoder_with_past=False):
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
        # assign k, v weight since module structure changed
        # import pdb; pdb.set_trace()
        fix_pretrained_model_weight(model)

    encoder = model.encoder
    decoder = model.decoder
    lm_head = model.lm_head

    t5_encoder = T5Encoder(encoder).eval()
    t5_decoder = T5Decoder(decoder, model.config).eval()
    t5_decoder_with_past = T5Decoder(decoder, model.config, flatten_decoder_with_past).eval()
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
    decoder_outs_with_past = decoder_with_past(decoder_input_ids, encoder_out, attention_mask, has_past, decoder_outs_flatten[1:])
    # NOTE: flatten tuple output.
    d = [decoder_outs_with_past[0]]
    for t in decoder_outs_with_past[1]:
        d = d + list(t)
    decoder_outs_with_past = tuple(d)
    print('decoder out with past: ', len(decoder_outs_with_past), [v.shape for v in decoder_outs_with_past])

    print('trace decoder with past')
    traced_decoder = torch.jit.trace(decoder_with_past, (decoder_input_ids, encoder_out, attention_mask, has_past, decoder_outs_flatten[1:]))

    print('export decoder with past')
    _ = torch.onnx._export(
        traced_decoder,
        (decoder_input_ids, encoder_out, attention_mask, has_past, decoder_outs_flatten[1:]),
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
        match_result = numpy.allclose(decoder_outs_with_past[i].detach().cpu().numpy(), ort_decoder_outputs[i], rtol=1e-03, atol=1e-05)
        print('i all close: ', i, match_result)
        assert match_result

    print("Test 2: test decoder with past not provided")
    has_past = torch.tensor(False)
    unused_decoder_state_inputs = [torch.zeros(v.shape) for v in decoder_outs_flatten[1:]]
    ort_decoder_inputs = {k:v.detach().cpu().numpy() for k, v in zip(["input_ids", "encoder_hidden_states", "attention_mask", "has_past"] + input_states_names, [decoder_input_ids, encoder_out, attention_mask, has_past, *unused_decoder_state_inputs])}
    ort_decoder_outputs = decoder_with_past_session.run(None, ort_decoder_inputs)

    import numpy
    for i in range(len(decoder_outs_flatten)):
        match_result = numpy.allclose(decoder_outs_flatten[i].detach().cpu().numpy(), ort_decoder_outputs[i], rtol=1e-03, atol=1e-05)
        print('i all close: ', i, match_result)
        assert match_result

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
    def __init__(self, decoder, config, flatten_output=False):
        super().__init__()
        self.decoder = decoder
        self.config = config
        self.flatten_output = flatten_output

    def forward(self, input_ids, encoder_hidden_states, attention_mask, has_past, past=None):
        # NOTE: not needed anymore
        # past_arg_key = (
        #     "past_key_value_states"
        #     if "past_key_value_states" in inspect.getfullargspec(self.decoder.forward).args
        #     else "past_key_values"
        # )
        # past_arg = {past_arg_key: past_key_values}
        past_key_values = tuple()
        if past is not None:
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

        # NOTE: flatten tuple output.
        if self.flatten_output and past_key_values is not None:
            d = []
            for t in past_key_values:
                d = d + list(t)
            return sequence_output, d

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


class T5ConfigTS(T5Config, torch.nn.Module):
    def init_module(self):
        torch.nn.Module.__init__(self)

class MinLengthLogitsProcessorTS(torch.nn.Module):
    r"""
    :class:`transformers.LogitsProcessor` enforcing a min-length by setting EOS probability to 0.

    Args:
        min_length (:obj:`int`):
            The minimum length below which the score of :obj:`eos_token_id` is set to :obj:`-float("Inf")`.
        eos_token_id (:obj:`int`):
            The id of the `end-of-sequence` token.
    """

    def __init__(self, min_length: int, eos_token_id: int):
        super().__init__()

        if not isinstance(min_length, int) or min_length < 0:
            raise ValueError(f"`min_length` has to be a positive integer, but is {min_length}")

        if not isinstance(eos_token_id, int) or eos_token_id < 0:
            raise ValueError(f"`eos_token_id` has to be a positive integer, but is {eos_token_id}")

        self.min_length = min_length
        self.eos_token_id = eos_token_id

    def forward(self, input_ids, scores) -> torch.Tensor:
        cur_len = input_ids.shape[-1]
        if cur_len < self.min_length:
            scores[:, self.eos_token_id] = -float("inf")
        return scores

class SimplifiedGenerator(torch.nn.Module, GenerationMixin):
    def __init__(self, model_name_or_path, onnx_path):
        super().__init__()
        self.device = torch.device('cpu')

        self.model_name_or_path = Path(model_name_or_path)
        # NOTE: workaround TorchScript issue with T5Config.
        # self.config = T5Config.from_pretrained(model_name_or_path)
        self.config = T5ConfigTS.from_pretrained(model_name_or_path)
        self.config.init_module()

        self.decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id=None, bos_token_id=self.config.bos_token_id)
        # self.max_length = config.max_length
        # self.min_length = config.min_length
        # self.pad_token_id = config.pad_token_id
        # self.bos_token_id = config.bos_token_id
        # self.eos_token_id = config.eos_token_id
        # self.output_scores = config.output_scores
        # self.output_attentions = config.output_attentions
        # self.output_hidden_states = config.output_hidden_states
        # self.return_dict_in_generate = config.return_dict_in_generate
        # self.is_encoder_decoder = config.is_encoder_decoder
        # self.num_beam_groups = config.num_beam_groups
        # self.do_sample = config.do_sample
        # self.num_return_sequences = config.num_return_sequences
        # self.length_penalty = config.length_penalty
        # self.early_stopping = config.early_stopping

        # NOTE: to create traced modules to run later.
        self.logits_processor = MinLengthLogitsProcessorTS(self.config.min_length, self.config.eos_token_id)
        self.beam_scorer = BeamSearchScorerTS()

        self._trace_modules()

    def _trace_modules(self):
        model = self.model_name_or_path.as_posix()
        simplified_encoder, decoder, decoder_with_past, lm_head = create_t5_encoder_decoder(model, flatten_decoder_with_past=True)

        # Example sequence
        tok = T5Tokenizer.from_pretrained(model)
        enc = tok("42 is the answer", return_tensors="pt")
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        traced_simplified_encoder = torch.jit.trace(simplified_encoder, (input_ids, attention_mask))
        encoder_out = simplified_encoder(input_ids, attention_mask)

        decoder_input_ids = torch.zeros((input_ids.shape[0], 1), dtype=input_ids.dtype, device=input_ids.device)
        attention_mask = input_ids.new_ones(input_ids.shape)

        has_past = torch.tensor(False)
        decoder_outs = decoder(decoder_input_ids, encoder_out, attention_mask, has_past)
        # flatten decoder outs
        d = [decoder_outs[0]]
        for t in decoder_outs[1]:
            d = d + list(t)
        decoder_outs_flatten = d

        has_past = torch.tensor(True)
        print('Tracing decoder with input shape:', decoder_input_ids.shape, encoder_out.shape, attention_mask.shape)
        traced_decoder = torch.jit.trace(decoder_with_past, (decoder_input_ids, encoder_out, attention_mask, has_past, decoder_outs_flatten[1:]))

        print(traced_decoder)
        print('Traced attention graph:', getattr(getattr(traced_decoder.decoder.block, '0').layer, '1').EncDecAttention.graph)
        print('Traced decoder graph:', traced_decoder.graph)
        traced_lm_head = torch.jit.trace(lm_head, (decoder_outs_flatten[0],))

        self.encoder = traced_simplified_encoder
        self.decoder = traced_decoder
        self.lm_head = traced_lm_head

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
        if input_ids is not None and attention_mask is not None:
            return self._encoder_forward(input_ids=input_ids, attention_mask=attention_mask)

        return self._decoder_forward(decoder_input_ids, attention_mask, encoder_outputs, past_key_values)

    def _encoder_forward(self, input_ids, attention_mask):
        last_hidden_state = self.encoder(input_ids, attention_mask)
        return last_hidden_state

    def _decoder_forward(self, decoder_input_ids, attention_mask, encoder_outputs, past_key_values:List[torch.Tensor]):
        if len(past_key_values) > 0:
            has_past = torch.tensor(True)
        else:
            has_past = torch.tensor(False)
            # TODO: fix this hack. This is for working around optional input.
            past_key_values = [torch.ones(1, 8, 1, 64) for _ in range(24)]
        print('calling decoder with input shape:', decoder_input_ids.shape, encoder_outputs.shape, attention_mask.shape)
        decoder_output, past = self.decoder(decoder_input_ids, encoder_outputs, attention_mask, has_past, past_key_values)
        lm_logits = self.lm_head(decoder_output)
        return lm_logits, past
        # return Seq2SeqLMOutput(logits=lm_logits, past_key_values=past)

    def get_encoder(self):
        return self

    def prepare_inputs_for_generation(self, input_ids, attention_mask, use_cache:bool, last_hidden_state, past:List[torch.Tensor]):
        if len(past) > 0:
            input_ids = input_ids[:, -1:]
        return input_ids, past, last_hidden_state, attention_mask, True
        # return {
        #     "decoder_input_ids": input_ids,
        #     "past_key_values": past,
        #     "encoder_outputs": encoder_outputs.last_hidden_state,
        #     "attention_mask": attention_mask,
        #     "use_cache": True,
        # }


    def _reorder_cache(self, past:List[torch.Tensor], beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        print('past for reorder cache size:', len(past))
        reordered_decoder_past = []
        for state in past:
            reordered_decoder_past.append(state.index_select(0, beam_idx))
        return reordered_decoder_past

    def beam_search(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        last_hidden_state: torch.Tensor,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
    ) -> torch.Tensor:
        r"""
        Generates sequences for models with a language modeling head using beam search decoding.

        Parameters:

            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                The sequence used as a prompt for the generation. If :obj:`None` the method initializes it as an empty
                :obj:`torch.LongTensor` of shape :obj:`(1,)`.
            beam_scorer (:obj:`BeamScorer`):
                An derived instance of :class:`~transformers.BeamScorer` that defines how beam hypotheses are
                constructed, stored and sorted during generation. For more information, the documentation of
                :class:`~transformers.BeamScorer` should be read.
            logits_processor (:obj:`LogitsProcessorList`, `optional`):
                An instance of :class:`~transformers.LogitsProcessorList`. List of instances of class derived from
                :class:`~transformers.LogitsProcessor` used to modify the prediction scores of the language modeling
                head applied at each generation step.
            max_length (:obj:`int`, `optional`, defaults to 20):
                The maximum length of the sequence to be generated.
            pad_token_id (:obj:`int`, `optional`):
                The id of the `padding` token.
            eos_token_id (:obj:`int`, `optional`):
                The id of the `end-of-sequence` token.
            output_attentions (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more details.
            output_hidden_states (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return trhe hidden states of all layers. See ``hidden_states`` under returned tensors
                for more details.
            output_scores (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the prediction scores. See ``scores`` under returned tensors for more details.
            return_dict_in_generate (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the :obj:`forward` function of the model. If
                model is an encoder-decoder model the kwargs should include :obj:`encoder_outputs`.

        Return:
            :class:`~transformers.generation_utilsBeamSearchDecoderOnlyOutput`,
            :class:`~transformers.generation_utils.BeamSearchEncoderDecoderOutput` or obj:`torch.LongTensor`: A
            :obj:`torch.LongTensor` containing the generated tokens (default behaviour) or a
            :class:`~transformers.generation_utils.BeamSearchDecoderOnlyOutput` if
            ``model.config.is_encoder_decoder=False`` and ``return_dict_in_generate=True`` or a
            :class:`~transformers.generation_utils.BeamSearchEncoderDecoderOutput` if
            ``model.config.is_encoder_decoder=True``.


        Examples::

            >>> from transformers import (
            ...    AutoTokenizer,
            ...    AutoModelForSeq2SeqLM,
            ...    LogitsProcessorList,
            ...    MinLengthLogitsProcessor,
            ...    BeamSearchScorer,
            ... )
            >>> import torch

            >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
            >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

            >>> encoder_input_str = "translate English to German: How old are you?"
            >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids


            >>> # lets run beam search using 3 beams
            >>> num_beams = 3
            >>> # define decoder start token ids
            >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
            >>> input_ids = input_ids * model.config.decoder_start_token_id

            >>> # add encoder_outputs to model keyword arguments
            >>> model_kwargs = {
            ...     "encoder_outputs": model.get_encoder()(encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True)
            ... }

            >>> # instantiate beam scorer
            >>> beam_scorer = BeamSearchScorer(
            ...     batch_size=1,
            ...     max_length=model.config.max_length,
            ...     num_beams=num_beams,
            ...     device=model.device,
            ... )

            >>> # instantiate logits processors
            >>> logits_processor = LogitsProcessorList([
            ...     MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
            ... ])

            >>> outputs = model.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)

            >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
        """
        # NOTE: encoder_outputs is used in main function
        # NOTE: encoder_outputs is ModelOutput type, not supported by TorchScript.
        #       just return last_hidden_state for now.
        #
        # model_kwargs = {
        #     'attention_mask': attention_mask,
        #     'output_attentions': output_attentions,
        #     'output_hidden_states': output_hidden_states,
        #     'encoder_outputs': encoder_outputs,
        # }

        # init values
        max_length = max_length if max_length is not None else self.config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        batch_size = self.beam_scorer.batch_size

        # NOTE: initialize beam search hypotheses list.
        #       This is a workaround for GetAttr/SetAttr not supported well inside loops.
        _beam_hyps : List[torch.Tensor] = []
        _beam_scores : List[torch.Tensor] = []
        _beam_hyps_count = torch.zeros(batch_size, dtype=torch.long)
        _beam_hyps_worst_scores = torch.zeros(batch_size) + 1e9
        # TODO: fix this
        # _done = torch.tensor([False for _ in range(batch_size)], dtype=torch.bool)
        _done = torch.zeros(batch_size, dtype=torch.bool)

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        # if return_dict_in_generate and self.config.is_encoder_decoder:
        #     encoder_attentions = encoder_outputs.get("attentions") if output_attentions else None
        #     encoder_hidden_states = (
        #         encoder_outputs.get("hidden_states") if output_hidden_states else None
        #     )

        num_beams = self.beam_scorer.num_beams
        batch_beam_size, cur_len = input_ids.shape

        assert (
            num_beams * batch_size == batch_beam_size
        ), "Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))
        next_tokens = torch.zeros((batch_size, num_beams), dtype=torch.long, device=input_ids.device)
        next_indices = torch.zeros((batch_size, num_beams), dtype=torch.long, device=input_ids.device)

        past : List[torch.Tensor] = []
        while cur_len < max_length:
            # NOTE: input_ids, attention_mask, use_cache, encoder_outputs, past=None are used as input
            # Expands to
            # return {
            #     "decoder_input_ids": input_ids,
            #     "past_key_values": past,
            #     "encoder_outputs": encoder_outputs.last_hidden_state,
            #     "attention_mask": attention_mask,
            #     "use_cache": True,
            # }
            # model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            decoder_input_ids, past_key_values, last_hidden_state, attention_mask, use_cache = self.prepare_inputs_for_generation(
                input_ids,
                attention_mask,
                True,
                last_hidden_state,
                past,
            )

            # NOTE: was returning Output of type Seq2SeqLMOutput, but that is not scriptable
            logits, past = self._decoder_forward(
                decoder_input_ids=decoder_input_ids,
                past_key_values=past_key_values,
                encoder_outputs=last_hidden_state,
                attention_mask=attention_mask,
                # return_dict=True,
                # output_attentions=output_attentions,
                # output_hidden_states=output_hidden_states,
            )
            # next_token_logits = outputs.logits[:, -1, :]
            # NOTE: was returning Output of type Seq2SeqLMOutput, but that is not scriptable
            next_token_logits = logits[:, -1, :]

            # adjust tokens for Bart, *e.g.*
            # NOTE: unused
            # next_token_logits = self.adjust_logits_during_generation(
            #     next_token_logits, cur_len=cur_len, max_length=max_length
            # )

            next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

            next_token_scores = self.logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)

            # Store scores, attentions and hidden_states when required
            # NOTE: skipped, not used in production.
            # if return_dict_in_generate:
            #     if output_scores:
            #         scores += (next_token_scores,)
            #     if output_attentions:
            #         decoder_attentions += (
            #             (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
            #         )

            #     if output_hidden_states:
            #         decoder_hidden_states += (
            #             (outputs.decoder_hidden_states,)
            #             if self.config.is_encoder_decoder
            #             else (outputs.hidden_states,)
            #         )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_scores, beam_next_tokens, beam_idx, _beam_hyps, _beam_scores, _beam_hyps_count, _beam_hyps_worst_scores, _done = self.beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                _beam_hyps=_beam_hyps,
                _beam_scores=_beam_scores,
                _beam_hyps_count=_beam_hyps_count,
                _beam_hyps_worst_scores=_beam_hyps_worst_scores,
                _done=_done,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            cur_len = cur_len + 1

            # model_kwargs = self._update_model_kwargs_for_generation(
            #     outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            # )
            # if model_kwargs["past"] is not None:
            #     model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)

            # NOTE: skip this because past is already assigned when returned
            # past = outputs.past_key_values
            if len(past) > 0:
                past = self._reorder_cache(past, beam_idx)

            if self.beam_scorer.is_done(_done):
                break

        sequences, sequence_scores = self.beam_scorer.finalize(
            input_ids, beam_scores, next_tokens, next_indices,
            _beam_hyps=_beam_hyps,
            _beam_scores=_beam_scores,
            _beam_hyps_count=_beam_hyps_count,
            _beam_hyps_worst_scores=_beam_hyps_worst_scores,
            _done=_done,
            pad_token_id=pad_token_id, eos_token_id=eos_token_id,
        )

        # if return_dict_in_generate:
        #     if not output_scores:
        #         sequence_scores = None
        #     if self.config.is_encoder_decoder:
        #         return BeamSearchEncoderDecoderOutput(
        #             sequences=sequences,
        #             sequences_scores=sequence_scores,
        #             scores=scores,
        #             encoder_attentions=encoder_attentions,
        #             encoder_hidden_states=encoder_hidden_states,
        #             decoder_attentions=decoder_attentions,
        #             decoder_hidden_states=decoder_hidden_states,
        #         )
        #     else:
        #         return BeamSearchDecoderOnlyOutput(
        #             sequences=sequences,
        #             sequences_scores=sequence_scores,
        #             scores=scores,
        #             attentions=decoder_attentions,
        #             hidden_states=decoder_hidden_states,
        #         )
        # else:
        return sequences

    def generate_beam_search(self, input_ids, attention_mask, num_beams: int,
                             decoder_input_ids: Optional[torch.Tensor] = None,
                             num_beam_groups: Optional[int] = None,
                             max_length: Optional[int] = None,
                             min_length: Optional[int] = None,
                             do_sample: Optional[bool] = None,
                             num_return_sequences: Optional[int] = None,
                             bos_token_id: Optional[int] = None,
                             pad_token_id: Optional[int] = None,
                             eos_token_id: Optional[int] = None,
                             output_attentions: Optional[bool] = None,
                             output_hidden_states: Optional[bool] = None,
                             output_scores: Optional[bool] = None,
                             return_dict_in_generate: Optional[bool] = None,
                             decoder_start_token_id: Optional[int] = None,
                             use_cache: Optional[bool] = None,
                             repetition_penalty: Optional[float] = None,
                             no_repeat_ngram_size: Optional[int] = None,
                             early_stopping: Optional[bool] = None,
                             length_penalty: Optional[float] = None,
                             encoder_no_repeat_ngram_size: Optional[int] = None,
                             bad_words_ids: Optional[List[int]] = None,
                             diversity_penalty: Optional[float] = None,
                             ):
        # set init values
        num_beam_groups = num_beam_groups if num_beam_groups is not None else self.config.num_beam_groups
        max_length = max_length if max_length is not None else self.config.max_length
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )

        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # model_kwargs["output_attentions"] = output_attentions
        # model_kwargs["output_hidden_states"] = output_hidden_states

        if input_ids is None:
            # init `input_ids` with bos_token_id
            input_ids = self._prepare_input_ids_for_generation(bos_token_id)

        if attention_mask is None:
            # init `attention_mask` depending on `pad_token_id`
            attention_mask = self._prepare_attention_mask_for_generation(
                input_ids, pad_token_id, eos_token_id
            )

        # special case if pad_token_id is not defined
        if pad_token_id is None and eos_token_id is not None:
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            pad_token_id = eos_token_id

        # Storing encoder_input_ids for logits_processor that could use them
        encoder_input_ids = input_ids if self.config.is_encoder_decoder else None


        # add encoder_outputs to model_kwargs
        # NOTE: rewrite below line to avoid using kwargs
        # model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(input_ids, model_kwargs)
        last_hidden_state = self._encoder_forward(input_ids, attention_mask)

        # set input_ids as decoder_input_ids
        if decoder_input_ids is not None:
            # input_ids = model_kwargs.pop("decoder_input_ids")
            input_ids = decoder_input_ids
        else:
            input_ids = self._prepare_decoder_input_ids_for_generation(
                input_ids, decoder_start_token_id=decoder_start_token_id, bos_token_id=bos_token_id
            )

        # if input_ids.shape[-1] >= max_length:
        #     input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
        #     logger.warning(
        #         f"Input length of {input_ids_string} is {input_ids.shape[-1]}, but ``max_length`` is set to {max_length}."
        #         "This can lead to unexpected behavior. You should consider increasing ``config.max_length`` or ``max_length``."
        #     )

        # determine generation mode
        is_greedy_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is False
        is_sample_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is True
        is_beam_gen_mode = (num_beams > 1) and (num_beam_groups == 1) and do_sample is False
        is_beam_sample_gen_mode = (num_beams > 1) and (num_beam_groups == 1) and do_sample is True
        is_group_beam_gen_mode = (num_beams > 1) and (num_beam_groups > 1)
        if num_beam_groups > num_beams:
            raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")
        if is_group_beam_gen_mode and do_sample is True:
            raise ValueError(
                "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
            )

        # set model_kwargs
        # model_kwargs["use_cache"] = use_cache

        # get distribution pre_processing samplers
        # NOTE: callable not supported, remove prefix_allowed_tokens_fn
        #       support only MinLengthLogitsProcessor for the moment
        # logits_processor = self._get_logits_processor(
        #     repetition_penalty=repetition_penalty,
        #     no_repeat_ngram_size=no_repeat_ngram_size,
        #     encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
        #     encoder_input_ids=encoder_input_ids,
        #     bad_words_ids=bad_words_ids,
        #     min_length=min_length,
        #     eos_token_id=eos_token_id,
        #     prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        #     num_beams=num_beams,
        #     num_beam_groups=num_beam_groups,
        #     diversity_penalty=diversity_penalty,
        # )
        min_length = min_length if min_length is not None else self.config.min_length
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        # NOTE: Stock LogitsProcessor is not nn.Module. And neither is nn.Module obj scriptable as method arg.
        #       Work around by storing as model attribute.
        # logits_processor = LogitsProcessorList()
        # logits_processor.append(MinLengthLogitsProcessor(min_length, eos_token_id))

        batch_size = input_ids.shape[0]

        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping

        if num_return_sequences > num_beams:
            raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

        self.beam_scorer.init(
            batch_size=batch_size,
            max_length=max_length,
            num_beams=num_beams,
            device=self.device,
            length_penalty=length_penalty,
            do_early_stopping=early_stopping,
            num_beam_hyps_to_keep=num_return_sequences,
        )
        # interleave with `num_beams`
        input_ids, attention_mask, last_hidden_state = self._expand_inputs_for_generation(
            input_ids, attention_mask, last_hidden_state, expand_size=num_beams,
        )

        return self.beam_search(
            input_ids,
            attention_mask,
            last_hidden_state,
            max_length=max_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
        )

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        last_hidden_state: torch.Tensor,
        expand_size: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        attention_mask = attention_mask.index_select(0, expanded_return_idx)

        last_hidden_state = last_hidden_state.index_select(
            0, expanded_return_idx.to(last_hidden_state.device)
        )
        return input_ids, attention_mask, last_hidden_state

    # NOTE: reason for rewrite is `torch.LongTensor` not supported by torch script
    def _prepare_decoder_input_ids_for_generation(
        self, input_ids: torch.Tensor, decoder_start_token_id: Optional[int] = None, bos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        decoder_start_token_id = self.decoder_start_token_id  # self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
        decoder_input_ids = (
            torch.ones((input_ids.shape[0], 1), dtype=input_ids.dtype, device=input_ids.device)
            * decoder_start_token_id
        )
        return decoder_input_ids

    def forward(self, input_ids, attention_mask, num_beams: int):
        return self.generate_beam_search(input_ids, attention_mask, num_beams)


"""
Trace encoder and decoder at initialization.
Decoder is traced with past_key_value_states as optional inputs to reduce memory consumption.
The generation logic still happens in pytorch.
"""
class OnnxT5_Full(GenerationMixin, torch.nn.Module):
    def __init__(self, model_name_or_path, onnx_path):
        super().__init__()
        self.device = torch.device('cpu')

        self.model_name_or_path = Path(model_name_or_path)
        self.onnx_path = Path(onnx_path)
        self.mode_base_name = self.model_name_or_path.stem
        self.encoder_path = self.onnx_path.joinpath(f"{self.mode_base_name}_encoder.onnx")
        self.decoder_path = self.onnx_path.joinpath(f"{self.mode_base_name}_decoder.onnx")
        self.lm_head_path = self.onnx_path.joinpath(f"{self.mode_base_name}_lm_head.onnx")
        self.config = T5Config.from_pretrained(model_name_or_path)

        # NOTE: to test export successfully
        self._export_onnx_graph()

        # NOTE: to create traced modules to run later.
        self._trace_modules()

    def _trace_modules(self):
        model = self.model_name_or_path.as_posix()
        simplified_encoder, decoder, decoder_with_past, lm_head = create_t5_encoder_decoder(model)

        # Example sequence
        tok = T5Tokenizer.from_pretrained(model)
        enc = tok("42 is the answer", return_tensors="pt")
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        traced_simplified_encoder = torch.jit.trace(simplified_encoder, (input_ids, attention_mask))
        encoder_out = simplified_encoder(input_ids, attention_mask)

        decoder_input_ids = torch.zeros((input_ids.shape[0], 1), dtype=input_ids.dtype, device=input_ids.device)
        attention_mask = input_ids.new_ones(input_ids.shape)

        has_past = torch.tensor(False)
        decoder_outs = decoder(decoder_input_ids, encoder_out, attention_mask, has_past)
        # flatten decoder outs
        d = [decoder_outs[0]]
        for t in decoder_outs[1]:
            d = d + list(t)
        decoder_outs_flatten = tuple(d)

        has_past = torch.tensor(True)
        print('Tracing decoder with input shape:', decoder_input_ids.shape, encoder_out.shape, attention_mask.shape)
        traced_decoder = torch.jit.trace(decoder_with_past, (decoder_input_ids, encoder_out, attention_mask, has_past, decoder_outs_flatten[1:]))

        print(traced_decoder)
        print('Traced attention graph:', getattr(getattr(traced_decoder.decoder.block, '0').layer, '1').EncDecAttention.graph)
        traced_lm_head = torch.jit.trace(lm_head, (decoder_outs_flatten[0],))

        self.encoder = traced_simplified_encoder
        self.decoder = traced_decoder
        # self.decoder_ = decoder_with_past
        self.lm_head = traced_lm_head

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

        if past_key_values is not None:
            has_past = torch.tensor(True)
            past_key_values = flatten_past(past_key_values)
        else:
            has_past = torch.tensor(False)
            # TODO: fix this hack
            past_key_values = [torch.ones(1, 8, 1, 64) for _ in range(24)]

        print('calling decoder with input shape:', decoder_input_ids.shape, encoder_outputs.shape, attention_mask.shape)
        decoder_output, past = self.decoder(decoder_input_ids, encoder_outputs, attention_mask, has_past, past_key_values)
        # decoder_output, past = self.decoder_(decoder_input_ids, encoder_outputs, attention_mask, has_past, *past_key_values)
        lm_logits = self.lm_head(decoder_output)
        return Seq2SeqLMOutput(logits=lm_logits, past_key_values=past)

    def _encoder_forward(self, input_ids=None, attention_mask=None):
        last_hidden_state = self.encoder(input_ids, attention_mask)
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
        print('calling into reorder cache')
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

    def forward(self, input_ids, attention_mask, num_beams, use_cache):
        return self.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=num_beams, use_cache=use_cache)
