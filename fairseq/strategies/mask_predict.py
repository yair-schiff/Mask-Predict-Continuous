# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch

from . import DecodingStrategy, register_strategy
from .strategy_utils import generate_step_with_prob, assign_single_value_long, assign_single_value_byte, assign_multi_value_long, convert_tokens


from functools import partial
from .strategy_utils import classifier_guidance
from fairseq.data import Dictionary
from fairseq.models.classifier import Classifier


@register_strategy('mask_predict')
class MaskPredict(DecodingStrategy):
    
    def __init__(self, args):
        super().__init__()
        self.iterations = args.decoding_iterations
        self.ctrl_fn = self.init_ctrl_fn(args)

    @staticmethod
    def add_args(parser):
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
        bsz, seq_len = tgt_tokens.size()
        pad_mask = tgt_tokens.eq(tgt_dict.pad())
        seq_lens = seq_len - pad_mask.sum(dim=1)
        
        iterations = seq_len if self.iterations is None else self.iterations
        
        tgt_tokens, token_probs = self.generate_non_autoregressive(model, encoder_out, tgt_tokens)
        assign_single_value_byte(tgt_tokens, pad_mask, tgt_dict.pad())
        assign_single_value_byte(token_probs, pad_mask, 1.0)
        #print("Initialization: ", convert_tokens(tgt_dict, tgt_tokens[0]))
        
        for counter in range(1, iterations):
            num_mask = (seq_lens.float() * (1.0 - (counter / iterations))).long()

            assign_single_value_byte(token_probs, pad_mask, 1.0)
            mask_ind = self.select_worst(token_probs, num_mask)
            assign_single_value_long(tgt_tokens, mask_ind, tgt_dict.mask())
            assign_single_value_byte(tgt_tokens, pad_mask, tgt_dict.pad())

            #print("Step: ", counter+1)
            #print("Masking: ", convert_tokens(tgt_dict, tgt_tokens[0]))
            decoder_out = model.decoder(tgt_tokens, encoder_out)
            if self.ctrl_fn is not None:
                new_decoder_out = self.ctrl_fn(decoder_out[0], pad_mask)
                decoder_out = (new_decoder_out, decoder_out[1])
            new_tgt_tokens, new_token_probs, all_token_probs = generate_step_with_prob(decoder_out)
            
            assign_multi_value_long(token_probs, mask_ind, new_token_probs)
            assign_single_value_byte(token_probs, pad_mask, 1.0)
            
            assign_multi_value_long(tgt_tokens, mask_ind, new_tgt_tokens)
            assign_single_value_byte(tgt_tokens, pad_mask, tgt_dict.pad())
            #print("Prediction: ", convert_tokens(tgt_dict, tgt_tokens[0]))
        
        lprobs = token_probs.log().sum(-1)
        return tgt_tokens, lprobs
    
    def generate_non_autoregressive(self, model, encoder_out, tgt_tokens):
        decoder_out = model.decoder(tgt_tokens, encoder_out)
        tgt_tokens, token_probs, _ = generate_step_with_prob(decoder_out)
        return tgt_tokens, token_probs

    def select_worst(self, token_probs, num_mask):
        bsz, seq_len = token_probs.size()
        masks = [token_probs[batch, :].topk(max(1, num_mask[batch]), largest=False, sorted=False)[1] for batch in range(bsz)]
        masks = [torch.cat([mask, mask.new(seq_len - mask.size(0)).fill_(mask[0])], dim=0) for mask in masks]
        return torch.stack(masks, dim=0)

