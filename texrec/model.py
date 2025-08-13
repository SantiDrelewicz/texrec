from torch import nn
from transformers import BertModel
from .constants import TOKENIZER_EMBEDDING_MODEL_NAME


class Model(nn.Module):
    """
    A Recurrent Neural Network (RNN) model for punctuation and capitalization tasks.
    This model uses a pre-trained BERT embedding layer followed by an RNN layer (LSTM or GRU)
    and three separate linear heads for predicting initial punctuation, final punctuation, and capitalization.

    Args:
        hidden_size (int): The size of the hidden state in the RNN.
        bidirectional (bool): Whether to use a bidirectional RNN.
        lstm (bool): Whether to use LSTM cells instead of GRU cells.
        num_layers (int): The number of recurrent layers.
        dropout (float): Dropout rate applied to the RNN layers.
        freeze (bool): If True, the parameters of the embedding layer will not be updated during training.
    """
    def __init__(
        self,
        hidden_size: int,
        bidirectional: bool = False,
        lstm: bool = False,
        num_layers: int = 1,
        dropout: float = 0.1,
        freeze: bool = True
    ):
        super().__init__()

        self.embedding = BertModel.from_pretrained(TOKENIZER_EMBEDDING_MODEL_NAME).embeddings.word_embeddings

        if freeze:
          for param in self.embedding.parameters():
            param.requires_grad = False

        self._num_directions = 2 if bidirectional else 1

        if lstm:
          self.rnn = nn.LSTM(input_size=self.embedding.embedding_dim,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             batch_first=True,
                             bidirectional=bidirectional,
                             dropout = dropout if num_layers > 1 else 0.0)
        else:
          self.rnn = nn.RNN(input_size=self.embedding.embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional,
                            dropout = dropout if num_layers > 1 else 0.0)

        self._init_punct_head = nn.Linear(hidden_size * self._num_directions, 2)
        self._final_punct_head = nn.Linear(hidden_size * self._num_directions, 4)
        self._capital_head = nn.Linear(hidden_size * self._num_directions, 4)


    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.rnn(emb)

        init_logits = self._init_punct_head(out)
        final_logits = self._final_punct_head(out)
        cap_logits = self._capital_head(out)

        return {"init_punct": init_logits,
                "final_punct": final_logits,
                "capital": cap_logits}
