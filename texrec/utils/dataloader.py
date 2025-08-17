from .dataset import PunctCapitalDataset, collate_fn
import torch
from torch.utils.data import DataLoader

def create_dataloader(
    sentences: list[str],  tokenizer, batch_size: int, num_workers: int = 0,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> DataLoader:
    """
    Creates a DataLoader for the punctuation and capitalization dataset.

    Args:
        sentences (list[str]): The list of input sentences.
        tokenizer: The tokenizer to use for encoding the sentences.
        batch_size (int): The batch size for the DataLoader.
        num_workers (int): The number of worker processes for the DataLoader.

    Returns:
        DataLoader: A DataLoader for the dataset.
    """
    dataset = PunctCapitalDataset(sentences, tokenizer, device)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers)