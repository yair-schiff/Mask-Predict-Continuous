import math

import ipdb

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('classification')
class ClassificationCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(sample['tokens'])
        loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['label'].size(0)
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'sample_size': sample_size,
            'ntokens': sample['ntokens'],
            'nsentences': sample_size
        }
        return loss, sample_size, logging_output

    @staticmethod
    def compute_loss(model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = sample['label'].view(-1, 1)
        loss = -lprobs.gather(dim=-1, index=target)
        if reduce:
            loss = loss.sum()
        return loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'sample_size': sample_size,
            'ntokens': ntokens,
            'nsentences': sample_size
        }
