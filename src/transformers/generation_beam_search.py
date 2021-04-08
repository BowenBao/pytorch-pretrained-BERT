# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from collections import UserDict
from typing import Optional, Tuple, List

import torch

from .file_utils import add_start_docstrings


PROCESS_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size * num_beams, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using any class inheriting from :class:`~transformers.PretrainedTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        next_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, 2 * num_beams)`):
            Current scores of the top :obj:`2 * num_beams` non-finished beam hypotheses.
        next_tokens (:obj:`torch.LongTensor` of shape :obj:`(batch_size, 2 * num_beams)`):
            :obj:`input_ids` of the tokens corresponding to the top :obj:`2 * num_beams` non-finished beam hypotheses.
        next_indices (:obj:`torch.LongTensor` of shape :obj:`(batch_size, 2 * num_beams)`):
            Beam indices indicating to which beam hypothesis the :obj:`next_tokens` correspond.
        pad_token_id (:obj:`int`, `optional`):
            The id of the `padding` token.
        eos_token_id (:obj:`int`, `optional`):
            The id of the `end-of-sequence` token.

    Return:
        :obj:`UserDict`: A dictionary composed of the fields as defined above:

            - **next_beam_scores** (:obj:`torch.FloatTensor` of shape :obj:`(batch_size * num_beams)`) -- Updated
              scores of all non-finished beams.
            - **next_beam_tokens** (:obj:`torch.FloatTensor` of shape :obj:`(batch_size * num_beams)`) -- Next tokens
              to be added to the non-finished beam_hypotheses.
            - **next_beam_indices** (:obj:`torch.FloatTensor` of shape :obj:`(batch_size * num_beams)`) -- Beam indices
              indicating to which beam the next tokens shall be added.

"""

FINALIZE_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size * num_beams, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using any class inheriting from :class:`~transformers.PretrainedTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        final_beam_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size * num_beams)`):
            The final scores of all non-finished beams.
        final_beam_tokens (:obj:`torch.FloatTensor` of shape :obj:`(batch_size * num_beams)`):
            The last tokens to be added to the non-finished beam_hypotheses.
        final_beam_indices (:obj:`torch.FloatTensor` of shape :obj:`(batch_size * num_beams)`):
            The beam indices indicating to which beam the :obj:`final_beam_tokens` shall be added.
        pad_token_id (:obj:`int`, `optional`):
            The id of the `padding` token.
        eos_token_id (:obj:`int`, `optional`):
            The id of the `end-of-sequence` token.

    Return:
        :obj:`torch.LongTensor` of shape :obj:`(batch_size * num_return_sequences, sequence_length)`: The generated
        sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or shorter if all
        batches finished early due to the :obj:`eos_token_id`.

"""


class BeamScorer(ABC):
    """
    Abstract base class for all beam scorers that are used for :meth:`~transformers.PretrainedModel.beam_search` and
    :meth:`~transformers.PretrainedModel.beam_sample`.
    """

    @abstractmethod
    @add_start_docstrings(PROCESS_INPUTS_DOCSTRING)
    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        **kwargs
    ) -> Tuple[torch.Tensor]:
        raise NotImplementedError("This is an abstract method.")

    @abstractmethod
    @add_start_docstrings(FINALIZE_INPUTS_DOCSTRING)
    def finalize(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        **kwargs
    ) -> torch.LongTensor:
        raise NotImplementedError("This is an abstract method.")


class BeamSearchScorer(BeamScorer):
    r"""
    :class:`transformers.BeamScorer` implementing standard beam search decoding.

    Adapted in part from `Facebook's XLM beam search code
    <https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529>`__.

    Reference for the diverse beam search algorithm and implementation `Ashwin Kalyan's DBS implementation
    <https://github.com/ashwinkalyan/dbs/blob/master/dbs/beam_utils.lua>`__

    Args:
        batch_size (:obj:`int`):
            Batch Size of :obj:`input_ids` for which standard beam search decoding is run in parallel.
        max_length (:obj:`int`):
            The maximum length of the sequence to be generated.
        num_beams (:obj:`int`):
            Number of beams for beam search.
        device (:obj:`torch.device`):
            Defines the device type (*e.g.*, :obj:`"cpu"` or :obj:`"cuda"`) on which this instance of
            :obj:`BeamSearchScorer` will be allocated.
        length_penalty (:obj:`float`, `optional`, defaults to 1.0):
            Exponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 in order to encourage the
            model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer
            sequences.
        do_early_stopping (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to stop the beam search when at least ``num_beams`` sentences are finished per batch or not.
        num_beam_hyps_to_keep (:obj:`int`, `optional`, defaults to 1):
            The number of beam hypotheses that shall be returned upon calling
            :meth:`~transformer.BeamSearchScorer.finalize`.
        num_beam_groups (:obj:`int`):
            Number of groups to divide :obj:`num_beams` into in order to ensure diversity among different groups of
            beams. See `this paper <https://arxiv.org/pdf/1610.02424.pdf>`__ for more details.
    """

    def __init__(
        self,
        batch_size: int,
        max_length: int,
        num_beams: int,
        device: torch.device,
        length_penalty: Optional[float] = 1.0,
        do_early_stopping: Optional[bool] = False,
        num_beam_hyps_to_keep: Optional[int] = 1,
        num_beam_groups: Optional[int] = 1,
    ):
        self.max_length = max_length
        self.num_beams = num_beams
        self.device = device
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
        self.num_beam_groups = num_beam_groups
        self.group_size = self.num_beams // self.num_beam_groups

        self._is_init = False
        self._beam_hyps : [List[BeamHypotheses]] = [
            BeamHypotheses(
                num_beams=self.num_beams,
                max_length=self.max_length,
                length_penalty=self.length_penalty,
                early_stopping=self.do_early_stopping,
            )
            for _ in range(batch_size)
        ]
        self._done = torch.tensor([False for _ in range(batch_size)], dtype=torch.bool, device=self.device)

        if not isinstance(num_beams, int) or num_beams <= 1:
            raise ValueError(
                f"`num_beams` has to be an integer strictly greater than 1, but is {num_beams}. For `num_beams` == 1, one should make use of `greedy_search` instead."
            )

        if not isinstance(num_beam_groups, int) or (num_beam_groups > num_beams) or (num_beams % num_beam_groups != 0):
            raise ValueError(
                f"`num_beam_groups` has to be an integer smaller or equal than `num_beams` and `num_beams` "
                f"has to be divisible by `num_beam_groups`, but is {num_beam_groups} with `num_beams` being {num_beams}."
            )

    @property
    def is_done(self) -> bool:
        return self._done.all()

    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor]:
        cur_len = input_ids.shape[-1]
        batch_size = len(self._beam_hyps)
        assert batch_size == (input_ids.shape[0] // self.group_size)

        device = input_ids.device
        next_beam_scores = torch.zeros((batch_size, self.group_size), dtype=next_scores.dtype, device=device)
        next_beam_tokens = torch.zeros((batch_size, self.group_size), dtype=next_tokens.dtype, device=device)
        next_beam_indices = torch.zeros((batch_size, self.group_size), dtype=next_indices.dtype, device=device)

        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                assert (
                    len(beam_hyp) >= self.num_beams
                ), "Batch can only be done if at least {} beams have been generated".format(self.num_beams)
                assert (
                    eos_token_id is not None and pad_token_id is not None
                ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                # pad the batch
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_token_id
                next_beam_indices[batch_idx, :] = 0
                continue

            # next tokens for this sentence
            beam_idx = 0
            for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
            ):
                batch_beam_idx = batch_idx * self.group_size + next_index
                # add to generated hypotheses if end of sentence
                if (eos_token_id is not None) and (next_token.item() == eos_token_id):
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.group_size
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    beam_hyp.add(
                        input_ids[batch_beam_idx].clone(),
                        next_score.item(),
                    )
                else:
                    # add next predicted token since it is not eos_token
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                    beam_idx += 1

                # once the beam for next step is full, don't add more tokens to it.
                if beam_idx == self.group_size:
                    break

            if beam_idx < self.group_size:
                raise ValueError(
                    f"At most {self.group_size} tokens in {next_tokens[batch_idx]} can be equal to `eos_token_id: {eos_token_id}`. Make sure {next_tokens[batch_idx]} are corrected."
                )

            # Check if we are done so that we can save a pad step if all(done)
            self._done[batch_idx] = self._done[batch_idx] or beam_hyp.is_done(
                next_scores[batch_idx].max().item(), cur_len
            )

        return UserDict(
            {
                "next_beam_scores": next_beam_scores.view(-1),
                "next_beam_tokens": next_beam_tokens.view(-1),
                "next_beam_indices": next_beam_indices.view(-1),
            }
        )

    def finalize(
        self,
        input_ids: torch.LongTensor,
        final_beam_scores: torch.FloatTensor,
        final_beam_tokens: torch.LongTensor,
        final_beam_indices: torch.LongTensor,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[torch.LongTensor]:
        batch_size = len(self._beam_hyps)

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                continue

            # all open beam hypotheses are added to the beam hypothesis
            # beam hypothesis class automatically keeps the best beams
            for beam_id in range(self.num_beams):
                batch_beam_idx = batch_idx * self.num_beams + beam_id
                final_score = final_beam_scores[batch_beam_idx].item()
                final_tokens = input_ids[batch_beam_idx]
                beam_hyp.add(final_tokens, final_score)

        # select the best hypotheses
        sent_lengths = input_ids.new(batch_size * self.num_beam_hyps_to_keep)
        best = []
        best_scores = torch.zeros(batch_size * self.num_beam_hyps_to_keep, device=self.device, dtype=torch.float32)

        # retrieve best hypotheses
        for i, beam_hyp in enumerate(self._beam_hyps):
            sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0])
            for j in range(self.num_beam_hyps_to_keep):
                best_hyp_tuple = sorted_hyps.pop()
                best_score = best_hyp_tuple[0]
                best_hyp = best_hyp_tuple[1]
                sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)

                # append to lists
                best.append(best_hyp)
                best_scores[i * self.num_beam_hyps_to_keep + j] = best_score

        # prepare for adding eos
        sent_max_len = min(sent_lengths.max().item() + 1, self.max_length)
        decoded: torch.LongTensor = input_ids.new(batch_size * self.num_beam_hyps_to_keep, sent_max_len)
        # shorter batches are padded if needed
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`pad_token_id` has to be defined"
            decoded.fill_(pad_token_id)

        # fill with hypotheses and eos_token_id if the latter fits in
        for i, hypo in enumerate(best):
            decoded[i, : sent_lengths[i]] = hypo
            if sent_lengths[i] < self.max_length:
                decoded[i, sent_lengths[i]] = eos_token_id
        return UserDict(
            {
                "sequences": decoded,
                "sequence_scores": best_scores,
            }
        )


class BeamHypotheses:
    def __init__(self, num_beams: int, max_length: int, length_penalty: float, early_stopping: bool):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp: torch.LongTensor, sum_logprobs: float):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / (hyp.shape[-1] ** self.length_penalty)
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs: float, cur_len: int) -> bool:
        """
        If there are enough hypotheses and that none of the hypotheses being generated can become better than the worst
        one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret


class BeamSearchScorerTS(torch.nn.Module):
    def __init__(self):
        super().__init__()

    # @property
    def is_done(self, _done) -> torch.Tensor:
        return _done.all()

    def init(
        self,
        batch_size: int,
        max_length: int,
        num_beams: int,
        device: torch.device,
        length_penalty: float = 1.0,
        do_early_stopping: bool = False,
        num_beam_hyps_to_keep: int = 1,
        num_beam_groups: int = 1,
    ):
        self.max_length = max_length
        self.num_beams = num_beams
        self.batch_size = batch_size
        # NOTE: avoid self.device, due to causing segfault in resolving SetAttr.
        # self.device = device
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
        self.num_beam_groups = num_beam_groups
        self.group_size = self.num_beams // self.num_beam_groups

        self._is_init = False
        # NOTE: TorchScript does not support List of Modules
        # self._beam_hyps : List[BeamHypothesesTS] = [
        #     BeamHypothesesTS(
        #         num_beams=self.num_beams,
        #         max_length=self.max_length,
        #         length_penalty=self.length_penalty,
        #         early_stopping=self.do_early_stopping,
        #     )
        #     for _ in range(batch_size)
        # ]

        # self._beam_hyps : List[torch.Tensor] = []
        # self._beam_scores : List[torch.Tensor] = []
        # self._beam_hyps_count = torch.zeros(batch_size, dtype=torch.long)
        # self._beam_hyps_worst_scores = torch.zeros(batch_size) + 1e9
        self._beam_hyps_max_length = max_length - 1  # ignoring bos_token
        # self.max_length = max_length - 1  # ignoring bos_token
        # self.length_penalty = length_penalty
        # self.early_stopping = early_stopping
        # self.num_beams = num_beams
        # # NOTE: add annotation to pass torchscript compile.
        # #   still, not supported due to that lambda is not supported, so cannot sort on beams if element is tuple.
        # # self.beams : List[Tuple[float, torch.Tensor]] = []
        # self.beam_scores : List[torch.Tensor] = []
        # self.beam_hyps : List[torch.Tensor] = []
        # self.worst_score = 1e9

        # self._done = torch.tensor([False for _ in range(batch_size)], dtype=torch.bool, device=device)

        if not isinstance(num_beams, int) or num_beams <= 1:
            raise ValueError(
                f"`num_beams` has to be an integer strictly greater than 1, but is {num_beams}. For `num_beams` == 1, one should make use of `greedy_search` instead."
            )

        if not isinstance(num_beam_groups, int) or (num_beam_groups > num_beams) or (num_beams % num_beam_groups != 0):
            raise ValueError(
                f"`num_beam_groups` has to be an integer smaller or equal than `num_beams` and `num_beams` "
                f"has to be divisible by `num_beam_groups`, but is {num_beam_groups} with `num_beams` being {num_beams}."
            )

    def hypo_len(self, hypo_idx: int,
        _beam_hyps_count: torch.Tensor):
        """
        Number of hypotheses in the list.
        """
        return _beam_hyps_count[hypo_idx]

    def hypo_add(self, hyp: torch.Tensor, sum_logprobs: float, hypo_idx: int,
        _beam_hyps_count: torch.Tensor,
        _beam_hyps_worst_scores: torch.Tensor,
        _beam_scores: List[torch.Tensor],
        _beam_hyps: List[torch.Tensor],):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / (hyp.shape[-1] ** self.length_penalty)
        hyps_count = self.hypo_len(hypo_idx, _beam_hyps_count)
        if hyps_count < self.num_beams or score > _beam_hyps_worst_scores[hypo_idx]:
            # NOTE: work around torch.sum(empty_tensor) = 0, while error in onnx.
            beam_idx = torch.sum(_beam_hyps_count[:hypo_idx]) if hypo_idx != 0 else torch.tensor(0, dtype=torch.long)
            # beam_idx = torch.sum(_beam_hyps_count[:hypo_idx])
            _beam_scores.insert(beam_idx, torch.tensor([score]))
            _beam_hyps.insert(beam_idx, hyp)
            if hyps_count + 1 > self.num_beams:
                # TODO: Slicing on sequence, work around for now.
                # sorted_next_scores, sorted_indices = torch.topk(torch.cat(_beam_scores[beam_idx:beam_idx+hyps_count+2]), hyps_count+1, largest=False)
                sorted_next_scores, sorted_indices = torch.topk(torch.cat(_beam_scores)[beam_idx:beam_idx+hyps_count+2], hyps_count+1, largest=False)
                del _beam_hyps[int((sorted_indices[0] + beam_idx))]
                del _beam_scores[int((sorted_indices[0] + beam_idx))]
                _beam_hyps_worst_scores[hypo_idx] = sorted_next_scores[1]
            else:
                _beam_hyps_worst_scores[hypo_idx] = min(score, _beam_hyps_worst_scores[hypo_idx])
                _beam_hyps_count[hypo_idx] = hyps_count + 1


        # if len(self) < self.num_beams or score > self.worst_score:
        #     self.beam_scores.append(torch.tensor([score]))
        #     self.beam_hyps.append(hyp)
        #     if len(self) > self.num_beams:
        #         sorted_next_scores, sorted_indices = torch.topk(torch.cat(self.beam_scores), len(self.beam_scores), largest=False)
        #         del self.beam_hyps[int(sorted_indices[0].item())]
        #         del self.beam_scores[int(sorted_indices[0].item())]
        #         self.worst_score = float(sorted_next_scores[1].item())
        #     else:
        #         self.worst_score = min(score, self.worst_score)

    def hypo_is_done(self, hypo_idx:int, best_sum_logprobs: float, cur_len: int,
        _beam_hyps_count: torch.Tensor,
        _beam_hyps_worst_scores: torch.Tensor,) -> bool:
        """
        If there are enough hypotheses and that none of the hypotheses being generated can become better than the worst
        one in the heap, then we are done with this sentence.
        """
        if self.hypo_len(hypo_idx, _beam_hyps_count) < self.num_beams:
            return False
        elif self.do_early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = _beam_hyps_worst_scores[hypo_idx].item() >= cur_score
            return ret

    def process(
        self,
        input_ids: torch.Tensor,
        next_scores: torch.Tensor,
        next_tokens: torch.Tensor,
        next_indices: torch.Tensor,
        _beam_hyps : List[torch.Tensor],
        _beam_scores : List[torch.Tensor],
        _beam_hyps_count : torch.Tensor,
        _beam_hyps_worst_scores : torch.Tensor,
        _done : torch.Tensor,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        cur_len = input_ids.shape[-1]
        batch_size = len(_beam_hyps_count)
        assert batch_size == (input_ids.shape[0] // self.group_size)

        device = input_ids.device
        next_beam_scores = torch.zeros((batch_size, self.group_size), dtype=next_scores.dtype, device=device)
        next_beam_tokens = torch.zeros((batch_size, self.group_size), dtype=next_tokens.dtype, device=device)
        next_beam_indices = torch.zeros((batch_size, self.group_size), dtype=next_indices.dtype, device=device)

        for batch_idx in range(batch_size):
            if _done[batch_idx]:
                assert (
                    self.hypo_len(batch_idx, _beam_hyps_count) >= self.num_beams
                ), "Batch can only be done if at least {} beams have been generated".format(self.num_beams)
                assert (
                    eos_token_id is not None and pad_token_id is not None
                ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                # pad the batch
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_token_id
                next_beam_indices[batch_idx, :] = 0
                continue

            # next tokens for this sentence
            beam_idx = 0
            for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
            ):
                batch_beam_idx = batch_idx * self.group_size + next_index
                # add to generated hypotheses if end of sentence
                if (eos_token_id is not None) and (next_token == eos_token_id):
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.group_size
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    self.hypo_add(
                        input_ids[batch_beam_idx].clone(),
                        next_score.item(),
                        batch_idx,
                        _beam_hyps_count,
                        _beam_hyps_worst_scores,
                        _beam_scores,
                        _beam_hyps,
                    )
                else:
                    # add next predicted token since it is not eos_token
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                    beam_idx += 1

                # once the beam for next step is full, don't add more tokens to it.
                if beam_idx == self.group_size:
                    break

            if beam_idx < self.group_size:
                raise ValueError(
                    f"At most {self.group_size} tokens in {next_tokens[batch_idx]} can be equal to `eos_token_id: {eos_token_id}`. Make sure {next_tokens[batch_idx]} are corrected."
                )

            # Check if we are done so that we can save a pad step if all(done)
            _done[batch_idx] = _done[batch_idx] or self.hypo_is_done(
                batch_idx, next_scores[batch_idx].max().item(), cur_len,
                _beam_hyps_count,
                _beam_hyps_worst_scores,
            )

        return next_beam_scores.view(-1), next_beam_tokens.view(-1), next_beam_indices.view(-1),  _beam_hyps, _beam_scores, _beam_hyps_count, _beam_hyps_worst_scores, _done

    def finalize(
        self,
        input_ids: torch.Tensor,
        final_beam_scores: torch.Tensor,
        final_beam_tokens: torch.Tensor,
        final_beam_indices: torch.Tensor,
        _beam_hyps : List[torch.Tensor],
        _beam_scores : List[torch.Tensor],
        _beam_hyps_count : torch.Tensor,
        _beam_hyps_worst_scores : torch.Tensor,
        _done : torch.Tensor,
        pad_token_id: int,
        eos_token_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(_beam_hyps_count)
        # batch_size = len(self._beam_hyps)

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx in range(batch_size):
            if _done[batch_idx]:
                continue

            # all open beam hypotheses are added to the beam hypothesis
            # beam hypothesis class automatically keeps the best beams
            for beam_id in range(self.num_beams):
                batch_beam_idx = batch_idx * self.num_beams + beam_id
                final_score = final_beam_scores[batch_beam_idx].item()
                final_tokens = input_ids[batch_beam_idx]
                self.hypo_add(final_tokens, final_score, batch_idx,
                        _beam_hyps_count,
                        _beam_hyps_worst_scores,
                        _beam_scores,
                        _beam_hyps,)

        # select the best hypotheses
        # sent_lengths = input_ids.new(batch_size * self.num_beam_hyps_to_keep)
        # NOTE: new is not scriptable
        sent_lengths = torch.zeros(batch_size * self.num_beam_hyps_to_keep, dtype=torch.long)
        best = []
        best_scores = torch.zeros(batch_size * self.num_beam_hyps_to_keep, device=input_ids.device, dtype=torch.float32)
        print('beam hyps count:', _beam_hyps_count)
        # retrieve best hypotheses
        for i in range(batch_size):
            # TODO: get rid of lambda and reflect update on beams
            batch_hypo_start = torch.sum(_beam_hyps_count[:i]) if i > 0 else torch.tensor(0, dtype=torch.long)
            batch_hypo_end = torch.sum(_beam_hyps_count[:i+1]) + 1
            print('batch_hypo_start:', batch_hypo_start, ' batch_hypo_end:', batch_hypo_end)
            beam_scores = torch.cat(_beam_scores)[batch_hypo_start:batch_hypo_end]
            print('beam_scores:', beam_scores)
            sorted_next_scores, sorted_indices = torch.topk(beam_scores, len(beam_scores), largest=True)
            for j in range(self.num_beam_hyps_to_keep):
                best_score = beam_scores[sorted_indices[j]]
                best_hyp = _beam_hyps[batch_hypo_start + sorted_indices[j]]
                sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)
                # append to lists
                best.append(best_hyp)
                best_scores[i * self.num_beam_hyps_to_keep + j] = best_score

        # prepare for adding eos
        sent_max_len = min(sent_lengths.max() + 1, self.max_length)
        decoded = torch.zeros(batch_size * self.num_beam_hyps_to_keep, sent_max_len, dtype=torch.long)
        # shorter batches are padded if needed
        if sent_lengths.min() != sent_lengths.max():
            assert pad_token_id is not None, "`pad_token_id` has to be defined"
            decoded.fill_(pad_token_id)

        # fill with hypotheses and eos_token_id if the latter fits in
        for i, hypo in enumerate(best):
            decoded[i, : sent_lengths[i]] = hypo
            if sent_lengths[i] < self.max_length:
                decoded[i, sent_lengths[i]] = eos_token_id

        return decoded, best_scores


@torch.jit.script
class BeamHypothesesTS:
    def __init__(self, num_beams: int, max_length: int, length_penalty: float, early_stopping: bool):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        # NOTE: add annotation to pass torchscript compile.
        #   still, not supported due to that lambda is not supported, so cannot sort on beams if element is tuple.
        # self.beams : List[Tuple[float, torch.Tensor]] = []
        self.beam_scores : List[torch.Tensor] = []
        self.beam_hyps : List[torch.Tensor] = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beam_hyps)

    def add(self, hyp: torch.Tensor, sum_logprobs: float):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / (hyp.shape[-1] ** self.length_penalty)
        if len(self) < self.num_beams or score > self.worst_score:
            # torch.cat((self.beam_scores, torch.tensor([score])))
            self.beam_scores.append(torch.tensor([score]))
            self.beam_hyps.append(hyp)
            if len(self) > self.num_beams:
                sorted_next_scores, sorted_indices = torch.topk(torch.cat(self.beam_scores), len(self.beam_scores), largest=False)
                del self.beam_hyps[int(sorted_indices[0])]
                del self.beam_scores[int(sorted_indices[0])]
                self.worst_score = float(sorted_next_scores[1])
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs: float, cur_len: int) -> bool:
        """
        If there are enough hypotheses and that none of the hypotheses being generated can become better than the worst
        one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret
