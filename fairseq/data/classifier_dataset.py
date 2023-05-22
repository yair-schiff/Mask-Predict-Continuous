import os.path

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import pipeline

from . import data_utils, FairseqDataset, IndexedRawTextDataset


class CachedClassificationDataset(Dataset):
    def __init__(self):
        self.cache = {}

    @property
    def supports_prefetch(self):
        return True

    def prefetch(self, indices):
        if all(i in self.cache.keys() for i in indices):
            return
        indices = sorted(set(indices))
        self.cache.clear()
        for i in indices:
            self.cache[i] = self.unfetched_get_item(i)

    def unfetched_get_item(self, i):
        raise NotImplementedError

    def check_index(self, i):
        if i < 0 or i >= len(self):
            raise IndexError('index out of range')

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, i):
        self.check_index(i)
        return self.cache[i]


class ImdbSentimentDataset(CachedClassificationDataset):
    def __init__(self, split, dictionary):
        super().__init__()
        self.dictionary = dictionary
        file = f'data-bin/imdb_sentiment/{split}.pt'  # TODO: Hack to work quickly
        if os.path.exists(file):
            data_load = torch.load(file)
            self.data = data_load['data']
            self.sizes = data_load['sizes']
            self.cache = data_load['cache']
        else:
            imdb = load_dataset('imdb')
            self.data = imdb[split]
            self.dictionary = dictionary
            self.sizes = [0] * len(imdb[split])
            self.cache = {i: self.unfetched_get_item(i) for i in tqdm(range(len(self)), desc=f'Tokenize {split}')}
            torch.save({'data': self.data, 'cache': self.cache, 'sizes': self.sizes}, file)

    def unfetched_get_item(self, i):
        tokens = self.dictionary.encode_line(self.data[i]['text'], add_if_not_exist=False)
        self.sizes[i] = len(tokens)
        return {
            'tokens': tokens,
            'label': self.data[i]['label']
        }

    def __len__(self):
        return len(self.sizes)


class SentimentClassifiedDataset(CachedClassificationDataset):
    def __init__(self, split, dictionary):
        super().__init__()
        self.dictionary = dictionary
        self.sent_to_num = {'LABEL_0': 0, 'LABEL_1': 1, 'LABEL_2': 2}
        # self.sent_to_num = {'NEG': 0, 'NEU': 1, 'POS': 2}
        file = f'data-bin/wmt16.ro-en/{split}_sent_class.pt'  # TODO: Hack to work quickly
        if os.path.exists(file):
            data_load = torch.load(file)
            self.data = data_load['data']
            self.sizes = data_load['sizes']
            self.cache = data_load['cache']
        else:
            config_name = 'cardiffnlp/twitter-roberta-base-sentiment'
            # config_name = 'finiteautomata/bertweet-base-sentiment-analysis'
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            class_pipe = pipeline(model=config_name, device=device)
            data_file = f'{split}.en'
            data_path = os.path.join('data', 'wmt16.en-ro', data_file)
            print('Loading raw dataset')
            ds = IndexedRawTextDataset(path=data_path, dictionary=dictionary)
            self.data = ds
            self.sizes = [i.shape[0] for i in ds]
            self.cache = {
                i: self.unfetched_get_item(i, class_pipe)
                for i in tqdm(range(len(self)), desc=f'Prepare {split}')
            }
            torch.save({'data': self.data, 'cache': self.cache, 'sizes': self.sizes}, file)

    def unfetched_get_item(self, i, class_pipe):
        tokens = self.data[i]  # [:100]
        text = self.dictionary.string(tokens, bpe_symbol='@@', escape_unk=False)
        text = text.replace(' &apos;', '\'')
        text = text.replace('&quot;', '')
        label = class_pipe(text)
        self.sizes[i] = len(tokens)
        return {
            'tokens': tokens,
            'label': self.sent_to_num[label[0]['label']]
        }

    def __len__(self):
        return len(self.sizes)


class ClassifierDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.
    Args:
        data (torch.utils.data.Dataset): dataset to wrap
        data_sizes (List[int]): sentence lengths
        dictionary (~fairseq.data.Dictionary): vocabulary
        max_positions (int, optional): max number of tokens in the sentence.
            Default: ``2048``
        shuffle (bool, optional): shuffle dataset elements before batching.
            Default: ``True``
        train (bool, optional): training dataset flag.
            Default: ``True``
        seed (int, optional): random seed for shuffling.
            Default: ``None``
    """
    def __init__(
            self, data, data_sizes, dictionary,
            max_positions=2048,
            shuffle=True,
            train=True,
            seed=None,
    ):
        self.data = data
        self.data_sizes = np.array(data_sizes)
        self.dictionary = dictionary
        self.max_positions = max_positions
        self.shuffle = shuffle
        self.train = train
        self.seed = seed
        self.random = np.random.RandomState(seed)
        self.seed = seed

    def __getitem__(self, index):
        return {
            'id': index,
            'tokens': self.data[index]['tokens'],
            'label': self.data[index]['label'],
            'ntokens': self.data[index]['tokens'].ne(self.dictionary.pad()).sum(-1).item()
        }

    def __len__(self):
        return len(self.data)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        """
        def merge(key):
            return data_utils.collate_tokens(
                [s[key] for s in samples], self.dictionary.pad(), left_pad=False,
            )

        if len(samples) == 0:
            return {}

        return {
            'id': torch.LongTensor([s['id'] for s in samples]),
            'ntokens': sum(s['ntokens'] for s in samples),
            'tokens': merge('tokens'),
            'label': torch.LongTensor([s['label'] for s in samples]),
            'nsentences': len(samples),
        }

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.data_sizes[index]

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.data_sizes[index]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle and self.train and self.seed is None:
            return np.random.permutation(len(self))

        indices = np.arange(len(self))
        return indices[np.argsort(self.data_sizes[indices], kind='mergesort')]

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return (
                hasattr(self.data, 'supports_prefetch')
                and self.data.supports_prefetch
        )

    def prefetch(self, indices):
        """Prefetch the data required for this epoch."""
        return self.data.prefetch(indices)
