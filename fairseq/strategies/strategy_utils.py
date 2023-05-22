# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
import torch
import torch.nn.functional as F


def duplicate_encoder_out(encoder_out, bsz, beam_size):
    encoder_out['encoder_out'] = encoder_out['encoder_out'].unsqueeze(2).repeat(1, 1, beam_size, 1).view(-1,
                                                                                                         bsz * beam_size,
                                                                                                         encoder_out[
                                                                                                             'encoder_out'].size(
                                                                                                             -1))
    if encoder_out['encoder_padding_mask'] is not None:
        encoder_out['encoder_padding_mask'] = encoder_out['encoder_padding_mask'].unsqueeze(1).repeat(1, beam_size,
                                                                                                      1).view(
            bsz * beam_size, -1)


def generate_step_with_prob(out):
    probs = F.softmax(out[0], dim=-1)
    max_probs, idx = probs.max(dim=-1)
    return idx, max_probs, probs


def assign_single_value_byte(x, i, y):
    x.view(-1)[i.view(-1).nonzero()] = y


def assign_multi_value_byte(x, i, y):
    x.view(-1)[i.view(-1).nonzero()] = y.view(-1)[i.view(-1).nonzero()]


def assign_single_value_long(x, i, y):
    b, l = x.size()
    i = i + torch.arange(0, b * l, l, device=i.device).unsqueeze(1)
    x.view(-1)[i.view(-1)] = y


def assign_multi_value_long(x, i, y):
    b, l = x.size()
    i = i + torch.arange(0, b * l, l, device=i.device).unsqueeze(1)
    x.view(-1)[i.view(-1)] = y.view(-1)[i.view(-1)]


def convert_tokens(dictionary, tokens):
    return ' '.join([dictionary[token] for token in tokens])


def classifier_guidance(out, pad_mask, ctrl_model, tgt_class, ctrl_steps=1, ctrl_lr=0.01):
    """
    Adapted from SSD-LM:
    https://github.com/xhan77/ssd-lm/blob/c0b67612c27928350aa073b303bc9df3a0e87f28/ssd_model_decode_fileio.py#L95
    """
    out_clone = out.clone()
    for ctrl_i in range(ctrl_steps):
        with torch.enable_grad():
            out_clone.requires_grad_()
            input_simplex_4ctr = F.softmax(out_clone, dim=-1)
            input_embeds_4ctr = torch.nn.functional.linear(
                input_simplex_4ctr,
                ctrl_model.base.embed_tokens.weight.t()
            )
            ctrl_loss = -torch.nn.functional.log_softmax(
                ctrl_model(
                    tokens=torch.zeros(input_embeds_4ctr.shape[:-1], device=input_embeds_4ctr.device),
                    inputs_embeds=input_embeds_4ctr,
                    pad_mask=pad_mask
                ),
                dim=-1
            )[:, tgt_class].mean()
            ctr_delta = -torch.autograd.grad(ctrl_loss, out_clone)[0]  # indexing 0 because the return is a tuple
        out_clone += ctrl_lr * ctr_delta
    return out_clone
