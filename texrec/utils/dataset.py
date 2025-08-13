import pandas as pd
from tqdm import tqdm
import torch
from typing import Optional
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from texrec.utils.preprocessing import extract_labels, PUNT_INCIAL_MAP, PUNT_FINAL_MAP


def build_df(sentences: list[str], tokenizer) -> pd.DataFrame:
    """
    Builds a DataFrame from the input sentences 
    by extracting labels of punctuation and capitalization.

    Args:
        sentences (list[str]): The list of input sentences.
        tokenizer: The tokenizer to use for encoding the sentences.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted labels.
    """
    rows = []
    for instance_id, sentence in tqdm(enumerate(sentences), total=len(sentences)):
        rows.extend(extract_labels(sentence, instance_id + 1, tokenizer))

    df = pd.DataFrame(rows)
    df["init_punct"] = df["init_punct"].map(PUNT_INCIAL_MAP)
    df["final_punct"] = df["final_punct"].map(PUNT_FINAL_MAP)

    return df


class PunctCapitalDataset(Dataset):
    """
    Dataset for punctuation and capitalization tasks.
    """
    def __init__(
        self,
        sentences: list[str], tokenizer,
        device: Optional[torch.device] = None,
    ):
      self.dataset = build_df(sentences, tokenizer)
      self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.dataset["instance_id"].unique())

    def __getitem__(self, instance_id: int):
        instance = self.dataset[self.dataset["instance_id"] == instance_id + 1]

        return (
            torch.tensor(instance["token_id"].tolist(), dtype=torch.long, device=self.device),
            torch.tensor(instance["init_punct"].tolist(), dtype=torch.long, device=self.device),
            torch.tensor(instance["final_punct"].tolist(), dtype=torch.long, device=self.device),
            torch.tensor(instance["capital"].tolist(), dtype=torch.long, device=self.device)
        )


def collate_fn(batch):
    input_ids, init_punct, final_punct, capital = zip(*batch)

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)

    init_punct = pad_sequence(init_punct, batch_first=True, padding_value=-100)
    final_punct = pad_sequence(final_punct, batch_first=True, padding_value=-100)
    capital = pad_sequence(capital, batch_first=True, padding_value=-100)

    targets = {"init_punct": init_punct, "final_punct": final_punct, "capital": capital}

    return input_ids, targets