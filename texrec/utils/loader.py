import random
import requests
import gzip
import io
import itertools


def _is_valid_sentence(sentence: str) -> bool:
    # Must contain more than 1 word
    if len(sentence.split()) < 2:
        return False
    # Must start with a capital letter or '¿'
    if not (sentence[0].isupper() or sentence.startswith('¿')):
        return False
    # Must end with ',', '?', '.'
    if not (sentence.endswith((',', '?', '.'))):
        return False
    # Cannot be all uppercase letters
    if sentence.isupper():
        return False
    # Can only contain letters, spaces and the signs ,?¿.'
    if not all(c.isalpha() or c.isspace() or c in ".,?¿'" for c in sentence):
        return False
    # Cannot be two or more punctuation marks together
    if any(
        punct_pairs[0] + punct_pairs[1] in sentence
        for punct_pairs in itertools.product(".,?¿'", ".,?¿'")
    ):
        return False
    return True


def load(n_sentences: int | None = None,
         shuffle: bool = True,
         random_seed: int | None = None) -> list[str]:
    """Load the Lain-American Spanish Open Subtitles corpus
    Args:
        n_sentences (int): Number of valid sentences to load
        shuffle (bool): Whether to shuffle the sentences
        random_seed (int | None): Random seed for reproducibility
    """
    response = requests.get("https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/mono/es_419.txt.gz", stream=True)
    print("Loading corpus...")
    with gzip.open(io.BytesIO(response.content), 'rt', encoding='utf-8') as file:
        lines = file.readlines()

    valid_lines = [line for line in lines if _is_valid_sentence(line.strip())]

    if shuffle:
        random.seed(random_seed)
        random.shuffle(valid_lines)

    if n_sentences is not None:
        valid_lines = valid_lines[:n_sentences]
        
    print(f"Corpus Loaded with {len(valid_lines)} valid sentences.")

    return valid_lines