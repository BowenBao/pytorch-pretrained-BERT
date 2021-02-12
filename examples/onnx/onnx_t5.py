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
        # Re-assign k, v weight since module structure changed.
        # The change is to support optional k, v, by rewriting the module with TorchScript.
        # In the first iteration for the decoder, past is not provided, hence k, v is None.
        fix_pretrained_model_weight(model)

    encoder = model.encoder
    decoder = model.decoder
    lm_head = model.lm_head

    t5_encoder = T5Encoder(encoder).eval()
    t5_decoder = T5Decoder(decoder, model.config).eval()
    t5_decoder_with_past = T5Decoder(decoder, model.config, flatten_decoder_with_past).eval()
    t5_lm_head = T5LMHead(lm_head).eval()
    return t5_encoder, t5_decoder, t5_decoder_with_past, t5_lm_head


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
        past_key_values = tuple()
        if past is not None:
            for i in range(len(past) // 4):
                past_key_values = past_key_values + (past[i*4:i*4+4],)
        if len(past_key_values) == 0:
            past_key_values = None

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
        self.config = T5ConfigTS.from_pretrained(model_name_or_path)
        self.config.init_module()

        self.decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id=None, bos_token_id=self.config.bos_token_id)

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
        traced_decoder = torch.jit.trace(decoder_with_past, (decoder_input_ids, encoder_out, attention_mask, has_past, decoder_outs_flatten[1:]))
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
            # NOTE: Workaround for optional input, value is only for placeholding, and not used in computation.
            #       ONNX does not have optional input. Emulate optional input by input and boolean flag.
            past_key_values = [torch.ones(1, 8, 1, 64) for _ in range(24)]

        decoder_output, past = self.decoder(decoder_input_ids, encoder_outputs, attention_mask, has_past, past_key_values)
        lm_logits = self.lm_head(decoder_output)
        return lm_logits, past

    def get_encoder(self):
        return self

    def prepare_inputs_for_generation(self, input_ids, attention_mask, use_cache:bool, last_hidden_state, past:List[torch.Tensor]):
        if len(past) > 0:
            input_ids = input_ids[:, -1:]
        return input_ids, past, last_hidden_state, attention_mask, True

    def _reorder_cache(self, past:List[torch.Tensor], beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        reordered_decoder_past = []
        for state in past:
            reordered_decoder_past.append(state.index_select(0, beam_idx))
        return reordered_decoder_past

    # NOTE: Removed a few arguments
    #       1. beam_scorer: custom class not supported by TorchScript.
    #       2. logits_processor: custom class not supported by TorchScript.
    #       3. model_kwargs: kwargs not supported by TorchScript.
    #                        moved attention_mask, last_hidden_state from model_kwargs to arguments.
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

        """
        # initialize values
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
        _done = torch.zeros(batch_size, dtype=torch.bool)

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
            )
            # NOTE: was returning Output of type Seq2SeqLMOutput, but that is not scriptable
            next_token_logits = logits[:, -1, :]

            next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

            next_token_scores = self.logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)

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

        last_hidden_state = self._encoder_forward(input_ids, attention_mask)

        # set input_ids as decoder_input_ids
        if decoder_input_ids is not None:
            input_ids = decoder_input_ids
        else:
            input_ids = self._prepare_decoder_input_ids_for_generation(
                input_ids, decoder_start_token_id=decoder_start_token_id, bos_token_id=bos_token_id
            )

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

        # get distribution pre_processing samplers
        # NOTE: callable not supported, remove prefix_allowed_tokens_fn
        #       support only MinLengthLogitsProcessor for the moment
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
        decoder_start_token_id = self.decoder_start_token_id
        decoder_input_ids = (
            torch.ones((input_ids.shape[0], 1), dtype=input_ids.dtype, device=input_ids.device)
            * decoder_start_token_id
        )
        return decoder_input_ids

    def forward(self, input_ids, attention_mask, num_beams: int):
        return self.generate_beam_search(input_ids, attention_mask, num_beams)
