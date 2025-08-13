import re
from transformers import BertTokenizer


def capitalization(word: str) -> int:
    if word.islower(): return 0
    elif word.istitle(): return 1
    elif word.isupper(): return 3
    else: return 2


def extract_labels(sentence: str, instance_id: int, tokenizer: BertTokenizer) -> list[str]:
    """
    Extracts labels from a sentence for punctuation and capitalization.
    Args:
        sentence (str): The sentence to process.
        instance_id (int): The ID of the instance.
        tokenizer (BertTokenizer): The tokenizer to use.
    Returns:
        list[dict]: A list of dictionaries containing token information.
    """
    all_tokens = re.findall(r"\w+['’]?\w*|¿|\?|,|\.|!|¡", sentence)
    tokens_labels = []
    for i in range(len(all_tokens)):

      try:
        if all_tokens[i] == '¿':
          continue
        else:
          if all_tokens[i-1] == '¿':
            init_punct = all_tokens[i-1]
          else:
            init_punct = ""

        if all_tokens[i] in ['.', ',', '?']:
          final_punct = all_tokens[i]
          tokens_labels[-1]["final_punct"] = final_punct
          continue

        tokens = tokenizer.tokenize(all_tokens[i].lower())
        for j, token in enumerate(tokens):
          token_id = tokenizer.convert_tokens_to_ids(token)
          tokens_labels.append({
            "instance_id": instance_id,
            "token_id": token_id,
            "token": token,
            "init_punct": init_punct if j == 0 else "",
            "final_punct": "",
            "capital": capitalization(all_tokens[i])
          })
      except:
          raise Exception(f"Failed in sentence: '{sentence}'")

    return tokens_labels