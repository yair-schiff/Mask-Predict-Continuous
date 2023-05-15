# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
import math

import ipdb
import torch
from torch.nn import functional as F

from . import DecodingStrategy, register_strategy
from .strategy_utils import generate_step_with_prob, assign_single_value_long, assign_single_value_byte, \
    assign_multi_value_long, convert_tokens


@register_strategy('continuous_mask_predict')
class ContinuousMaskPredict(DecodingStrategy):

    def __init__(self, args):
        super().__init__()
        self.iterations = args.decoding_iterations

    def generate(self, model, encoder_out, tgt_tokens, tgt_dict):
        model.decoder.masking_strategy = "uniform"  # Override this for inference
        bsz, seq_len = tgt_tokens.size()
        pad_mask = tgt_tokens.eq(tgt_dict.pad())
        seq_lens = seq_len - pad_mask.sum(dim=1)

        iterations = seq_len if self.iterations is None else self.iterations

        softmax_out = self.generate_non_autoregressive(model, encoder_out, tgt_tokens)
        vocab_sz = softmax_out.shape[-1]
        unif = (1 / vocab_sz) * torch.ones(vocab_sz, dtype=softmax_out.dtype, device=softmax_out.device)
        pad_prob = torch.zeros(softmax_out.shape[-1], device=softmax_out.device)
        pad_prob[tgt_dict.pad()] = 1.0
        for counter in range(1, iterations):
            new_tgt_tokens = tgt_tokens.clone()
            num_mask = (seq_lens.float() * (1.0 - (counter / iterations))).long()

            # print("Step: ", counter)
            # print("Masking: ", convert_tokens(tgt_dict, softmax_out[0].argmax(dim=-1)))
            softmax_out[pad_mask] = pad_prob
            mask_mask = self.select_most_uncertain(softmax_out, num_mask)
            softmax_out[mask_mask] = unif
            new_tgt_tokens[mask_mask] = tgt_dict.mask()

            decoder_out = model.decoder(
                prev_output_tokens=new_tgt_tokens,
                encoder_out=encoder_out,
                prev_output_probs=softmax_out
            )
            new_softmax_out = F.softmax(decoder_out[0], dim=-1)
            softmax_out[mask_mask] = new_softmax_out[mask_mask]

        lprobs = softmax_out.log().sum(dim=(1, 2))
        tgt_tokens = softmax_out.argmax(dim=-1)
        return tgt_tokens, lprobs

    def generate_non_autoregressive(self, model, encoder_out, tgt_tokens):
        decoder_out = model.decoder(prev_output_tokens=tgt_tokens, encoder_out=encoder_out)
        softmax_out = F.softmax(decoder_out[0], dim=-1)
        return softmax_out

    def select_most_uncertain(self, softmax_out, num_mask):
        bsz, seq_len = softmax_out.shape[:-1]
        neg_kl_to_unif = math.log(softmax_out.shape[-1]) + \
                         (1 / softmax_out.shape[-1]) * torch.sum(torch.log(softmax_out), dim=-1)
        masks = [torch.topk(neg_kl_to_unif[b], k=max(1, num_mask[b]), sorted=False)[1] for b in range(bsz)]
        mask_mask = torch.zeros((bsz, seq_len), dtype=torch.bool, device=softmax_out.device)
        for i, mask in enumerate(masks):
            mask_mask[i, mask] = True
        return mask_mask
