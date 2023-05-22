import torch.nn.functional as F
from torch import nn

from fairseq.modules import (
    TransformerSentenceEncoder
)
from . import (
    register_model, register_model_architecture, BaseFairseqModel
)


@register_model('classifier')
class Classifier(BaseFairseqModel):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
    """

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--nclasses', type=int, default=2)
        parser.add_argument('--num-encoder-layers', type=int, default=6)
        parser.add_argument('--embedding-dim', type=int, default=768)
        parser.add_argument('--ffn-embedding-dim', type=int, default=3072)
        parser.add_argument('--num-attention-heads', type=int, default=8)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--attention-dropout', type=float, default=0.1)
        parser.add_argument('--activation-dropout', type=float, default=0.1)
        parser.add_argument('--max-seq-len', type=int, default=1024)
        parser.add_argument('--num-segments', type=int, default=0)
        parser.add_argument('--use-position-embeddings', action='store_true', default=True)
        parser.add_argument('--offset-positions-by-padding', action='store_true', default=True)
        parser.add_argument('--encoder-normalize-before', action='store_true', default=False)
        parser.add_argument('--apply-bert-init', action='store_true', default=True)
        parser.add_argument('--activation-fn',type=str, default='relu')
        parser.add_argument('--learned-pos-embedding', action='store_true', default=False)
        parser.add_argument('--add-bias-kv', action='store_true', default=False)
        parser.add_argument('--add-zero-attn', action='store_true', default=False)
        parser.add_argument('--embed-scale', type=float, default=None)
        parser.add_argument('--freeze-embeddings', action='store_true', default=False)
        parser.add_argument('--n-trans-layers-to-freeze', type=int, default=0)
        parser.add_argument('--export', action='store_true', default=False)

    @classmethod
    def build_model(cls, args, task):
        return cls(args, task.dictionary)

    def __init__(self, args, dictionary):
        super().__init__()

        self.base = TransformerSentenceEncoder(
            padding_idx=dictionary.pad(),
            vocab_size=len(dictionary),
            num_encoder_layers=args.num_encoder_layers,
            embedding_dim=args.embedding_dim,
            ffn_embedding_dim=args.ffn_embedding_dim,
            num_attention_heads=args.num_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            num_segments=args.num_segments,
            use_position_embeddings=args.use_position_embeddings,
            offset_positions_by_padding=args.offset_positions_by_padding,
            encoder_normalize_before=args.encoder_normalize_before,
            apply_bert_init=args.apply_bert_init,
            activation_fn=args.activation_fn,
            learned_pos_embedding=args.learned_pos_embedding,
            add_bias_kv=args.add_bias_kv,
            add_zero_attn=args.add_zero_attn,
            embed_scale=args.embed_scale,
            freeze_embeddings=args.freeze_embeddings,
            n_trans_layers_to_freeze=args.n_trans_layers_to_freeze,
            export=args.export
        )

        self.classification_head = nn.Sequential(
            nn.Linear(args.embedding_dim, args.embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(args.embedding_dim // 2, args.nclasses)
        )

    def forward(self, tokens, inputs_embeds=None, pad_mask=None):
        """
        Args:
            tokens (LongTensor): tokens of shape `(batch, len)`

        Returns:
            logits (Tensor): classification head output of shape `(batch, ntargets)`

        """
        cls_embed = self.base(tokens, inputs_embeds=inputs_embeds, pad_mask=pad_mask)[1]
        logits = self.classification_head(cls_embed)

        return logits  # B x K (K = num classes)

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output.float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)


@register_model_architecture('classifier', 'classifier')
def base_architecture(args):
    args.nclasses = getattr(args, 'nclasses', 2)
    args.num_encoder_layers = getattr(args, 'num_encoder_layers', 6)
    args.embedding_dim = getattr(args, 'embedding_dim', 768)
    args.ffn_embedding_dim = getattr(args, 'ffn_embedding_dim', 3072)
    args.num_attention_heads = getattr(args, 'num_attention_heads', 8)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.1)
    args.max_seq_len = getattr(args, 'max_seq_len', 1024)
    args.num_segments = getattr(args, 'num_segments', 0)
    args.use_position_embeddings = getattr(args, 'use_position_embeddings', True)
    args.offset_positions_by_padding = getattr(args, 'offset_positions_by_padding', True)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.apply_bert_init = getattr(args, 'apply_bert_init', True)
    args.activation_fn = getattr(args, 'activation_fn', 'relu')
    args.learned_pos_embedding = getattr(args, 'learned_pos_embedding', False)
    args.add_bias_kv = getattr(args, 'add_bias_kv', False)
    args.add_zero_attn = getattr(args, 'add_zero_attn', False)
    args.embed_scale = getattr(args, 'embed_scale', None)
    args.freeze_embeddings = getattr(args, 'freeze_embeddings', False)
    args.n_trans_layers_to_freeze = getattr(args, 'n_trans_layers_to_freeze', 0)
    args.export = getattr(args, 'export', False)
