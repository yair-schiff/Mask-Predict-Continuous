# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
import math
from argparse import Namespace
from functools import partial
from os import path as osp

import torch
import yaml
from torch.nn import functional as F

from . import DecodingStrategy, register_strategy
from .strategy_utils import classifier_guidance
from fairseq.data import Dictionary
from fairseq.models.classifier import Classifier


@register_strategy('continuous_mask_predict')
class ContinuousMaskPredict(DecodingStrategy):

    def __init__(self, args):
        super().__init__()
        self.iterations = args.decoding_iterations
        self.refine_all = args.refine_all
        self.ctrl_fn = self.init_ctrl_fn(args)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--refine-all', action='store_true',
                            help='refine all tokens at each iteration')
        parser.add_argument('--ctrl-model', type=str,
                            help='path to trained control model')
        parser.add_argument('--tgt-class', type=int,
                            help='control constraint: target class to maximize')
        parser.add_argument('--ctrl-steps', type=int,
                            help='number of classifier guided drift steps')
        parser.add_argument('--ctrl-lr', type=float,
                            help='magnitude of classifier guided drift steps')

    @staticmethod
    def init_ctrl_fn(args):
        if args.ctrl_model is not None:
            ctrl_ckpt = torch.load(args.ctrl_model, map_location='cpu')
            ctrl_dict = Dictionary.load(ctrl_ckpt['args'].dict_file)
            ctrl_model = Classifier(ctrl_ckpt['args'], ctrl_dict)
            ctrl_model.load_state_dict(ctrl_ckpt['model'])
            ctrl_model.eval()
            if torch.cuda.is_available():
                ctrl_model.cuda()

            ctrl_args = {
                'ctrl_model': ctrl_model,
                'tgt_class': args.tgt_class,
                'ctrl_steps': args.ctrl_steps,
                'ctrl_lr': args.ctrl_lr
            }
            return partial(classifier_guidance, **ctrl_args)
        return None

    def generate(self, model, encoder_out, tgt_tokens, tgt_dict):
        model.decoder.masking_strategy = 'uniform'  # Override this for inference
        bsz, seq_len = tgt_tokens.size()
        pad_mask = tgt_tokens.eq(tgt_dict.pad())
        seq_lens = seq_len - pad_mask.sum(dim=1)

        iterations = seq_len if self.iterations is None else self.iterations

        # t: 0
        decoder_out = model.decoder(
            prev_output_tokens=tgt_tokens,
            encoder_out=encoder_out
        )[0]
        softmax_out = F.softmax(decoder_out, dim=-1)
        vocab_sz = softmax_out.shape[-1]
        unif = (1 / vocab_sz) * torch.ones(vocab_sz, dtype=softmax_out.dtype, device=softmax_out.device)
        pad_prob = torch.zeros(softmax_out.shape[-1], device=softmax_out.device)
        pad_prob[tgt_dict.pad()] = 1.0
        # t: 1 - T
        for counter in range(1, iterations):
            new_tgt_tokens = tgt_tokens.clone()
            num_mask = (seq_lens.float() * (1.0 - (counter / iterations))).long()

            # print("Step: ", counter)
            # print("Masking: ", convert_tokens(tgt_dict, softmax_out[0].argmax(dim=-1)))
            softmax_out[pad_mask] = pad_prob
            if self.refine_all:
                decoder_out = model.decoder(
                    prev_output_tokens=new_tgt_tokens,
                    encoder_out=encoder_out,
                    prev_output_probs=softmax_out
                )[0]
            else:
                mask_mask = self.select_most_uncertain(softmax_out, num_mask)
                softmax_out[mask_mask] = unif
                new_tgt_tokens[mask_mask] = tgt_dict.mask()

                new_decoder_out = model.decoder(
                    prev_output_tokens=new_tgt_tokens,
                    encoder_out=encoder_out,
                    prev_output_probs=softmax_out
                )[0]
                decoder_out[mask_mask] = new_decoder_out[mask_mask]  # Only 'masked' logits get updated
            if self.ctrl_fn is not None:
                decoder_out = self.ctrl_fn(decoder_out, pad_mask)
            softmax_out = F.softmax(decoder_out, dim=-1)
            # new_softmax_out = F.softmax(decoder_out[0], dim=-1)
            # softmax_out[mask_mask] = new_softmax_out[mask_mask]
        softmax_out[pad_mask] = pad_prob
        lprobs = softmax_out.log().sum(dim=(1, 2))
        tgt_tokens = softmax_out.argmax(dim=-1)
        return tgt_tokens, lprobs

    # def iterative_refinement(self, model, encoder_out, tgt_tokens, mask_mask, mask_token,
    #                          softmax_out=None, ctrl_fn=None, ctrl_steps=None, ctrl_lr=None):
    #     new_tgt_tokens = tgt_tokens.clone()
    #     new_tgt_tokens[mask_mask] = mask_token
    #     decoder_out = model.decoder(
    #         prev_output_tokens=new_tgt_tokens,
    #         encoder_out=encoder_out,
    #         prev_output_probs=softmax_out
    #     )
    #     if ctrl_fn is not None:
    #         ctrl_out = self.classifier_guidance(decoder_out, ctrl_fn, ctrl_steps, ctrl_lr)



    @staticmethod
    def select_most_uncertain(softmax_out, num_mask):
        bsz, seq_len = softmax_out.shape[:-1]
        neg_kl_to_unif = math.log(softmax_out.shape[-1]) + \
                         (1 / softmax_out.shape[-1]) * torch.sum(torch.log(softmax_out), dim=-1)
        masks = [torch.topk(neg_kl_to_unif[b], k=max(1, num_mask[b]), sorted=False)[1] for b in range(bsz)]
        mask_mask = torch.zeros((bsz, seq_len), dtype=torch.bool, device=softmax_out.device)
        for i, mask in enumerate(masks):
            mask_mask[i, mask] = True
        return mask_mask
