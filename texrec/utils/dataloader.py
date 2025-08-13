from .dataset import PunctCapitalDataset, collate_fn
from torch.utils.data import DataLoader

def create_dataloader(sentences: list[str], tokenizer, batch_size: int) -> DataLoader:
    """
    Creates a DataLoader for the punctuation and capitalization dataset.

    Args:
        sentences (list[str]): The list of input sentences.
        tokenizer: The tokenizer to use for encoding the sentences.
        batch_size (int): The batch size for the DataLoader.

    Returns:
        DataLoader: A DataLoader for the dataset.
    """
    dataset = PunctCapitalDataset(sentences, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
