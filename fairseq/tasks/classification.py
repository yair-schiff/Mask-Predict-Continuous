# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import os

from fairseq.data import Dictionary
from fairseq.data.classifier_dataset import ClassifierDataset, ImdbSentimentDataset, SentimentClassifiedDataset
from . import FairseqTask, register_task

DATASETS = {
    'imdb_sentiment': ImdbSentimentDataset,
    'wmt16.ro-en': SentimentClassifiedDataset
}


@register_task('classification')
class Classification(FairseqTask):
    """
    Classify text.
    """

    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        return None

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('--dataset', type=str, help='dataset to use',
                            choices=['wmt16.ro-en', 'imdb_sentiment'],
                            )
        parser.add_argument('--dict-file', type=str, help='dictionary filename (absolute path)')
        parser.add_argument('--max-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the sequence')

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary

    @classmethod
    def load_dictionary(cls, filename):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        return Dictionary.load(filename)

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        dictionary = cls.load_dictionary(args.dict_file)
        print('| dictionary: {} types'.format(len(dictionary)))

        return cls(args, dictionary)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        if self.args.dataset not in DATASETS.keys():
            raise NotImplementedError(f'Dataset {self.args.dataset} not implemented.')

        if split == "train":
            train = True
            seed = None
        elif split == "valid":
            train = True
            seed = 1
        elif split == "test":
            train = False
            seed = 1
        else:
            raise Exception('No such split: ' + str(split))

        dataset = DATASETS[self.args.dataset](
            split=split,
            dictionary=self.dictionary
        )
        self.datasets[split] = ClassifierDataset(
            data=dataset,
            data_sizes=dataset.sizes,
            dictionary=self.dictionary,
            max_positions=self.args.max_positions,
            shuffle=False,
            train=train,
            seed=seed
        )

    def max_positions(self):
        """Return the max input length allowed by the task."""
        return None
