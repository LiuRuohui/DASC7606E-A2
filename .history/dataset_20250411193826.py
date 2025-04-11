from datasets import load_dataset
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from pprint import pprint


def align_labels_with_tokens(
    labels: list[int],
    word_ids: list[int | None],
) -> list[int]:
    """
    Align labels with tokenized word IDs, ensuring special tokens are ignored (-100).

    The first rule we’ll apply is that special tokens get a label of -100.
    This is because by default -100 is an index that is ignored in the loss function we will use (cross entropy).
    Then, each token gets the same label as the token that started the word it’s inside, since they are part of the same entity.
    For tokens inside a word but not at the beginning, we replace the B- with I- (since the token does not begin the entity):

    Args:
        labels: List of token labels.
        word_ids: List of word IDs.

    Returns:
        A list of aligned labels.
    """
    # Write your code here.
    previous_word_idx = None

    label_ids = []
    for word_idx in word_ids:
        # Special tokens have a word id that is None. We set the label to -100 so they are ignored in loss
        if word_idx is None:
            label_ids.append(-100)
        elif previous_word_idx is None or word_idx != previous_word_idx:
            # Start of a new word
            label_ids.append(labels[word_idx])
        else:
            # Continuing the same word, repeat the previous label
            label_ids.append(labels[previous_word_idx])
        previous_word_idx = word_idx
    return label_ids


def tokenize_and_align_labels(examples: dict, tokenizer) -> dict:
    """
    Tokenize input examples and align labels for token classification.

    To preprocess our whole dataset, we need to tokenize all the inputs and apply align_labels_with_tokens() on all the labels.
    To take advantage of the speed of our fast tokenizer, it’s best to tokenize lots of texts at the same time,
    so this function will processes a list of examples and return a list of tokenized inputs with aligned labels.

    Args:
        examples: Input examples.
        tokenizer: Tokenizer object.

    Returns:
        Tokenized inputs with aligned labels.
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
    )
    all_labels = examples["ner_tags"]
    aligned_labels = [
        align_labels_with_tokens(
            labels=labels,
            word_ids=tokenized_inputs.word_ids(i),
        )
        for i, labels in enumerate(all_labels)
    ]
    tokenized_inputs["labels"] = aligned_labels
    return tokenized_inputs


def build_dataset() -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    """
    Build the dataset.

    Returns:
        The dataset.


    Below is an example of how to load a dataset.

    ```python
    from datasets import load_dataset

    raw_datasets = load_dataset('tomaarsen/MultiCoNER', 'multi')
    ```

    You can replace this with your own dataset. Make sure to include
    the `test` split and ensure that it is the same as the test split from the MultiCoNER NER dataset,
    Which means that:
        raw_datasets["test"] = load_dataset('tomaarsen/MultiCoNER', 'multi', split="test")
    """
    # Write your code here.
    raw_datasets = load_dataset('tomaarsen/MultiCoNER', 'multi')
    # if 'validation' not in raw_datasets.keys():
    #     # If validation split is not present, create it from the train split
    #     split_dataset = raw_datasets['train'].train_test_split(test_size=0.1)
    #     raw_datasets['train'] = split_dataset['train']
    #     raw_datasets['validation'] = split_dataset['test']
    return raw_datasets
        


def preprocess_data(raw_datasets: DatasetDict, tokenizer) -> DatasetDict:
    """
    Preprocess the data.

    Args:
        raw_datasets: Raw datasets.
        tokenizer: Tokenizer object.

    Returns:
        Tokenized datasets.
    """
    tokenized_datasets: DatasetDict = raw_datasets.map(
        function=lambda examples: tokenize_and_align_labels(
            examples=examples,
            tokenizer=tokenizer,
        ),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    return tokenized_datasets