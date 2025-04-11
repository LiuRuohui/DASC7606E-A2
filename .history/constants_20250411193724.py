SEED = 42  # Random seed for reproducibility
MODEL_CHECKPOINT = "facebookAI/xlm-roberta-base"  # Configurable model checkpoint
OUTPUT_DIR = "checkpoints"  # Configurable output directory

LABEL_TO_ID = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-LOC": 3,
    "I-LOC": 4,
    "B-CORP": 5,
    "I-CORP": 6,
    "B-GRP": 7,
    "I-GRP": 8,
    "B-PROD": 9,
    "I-PROD": 10,
    "B-CW": 11,
    "I-CW": 12,
}
ID_TO_LABEL = {i: label for i, label in enumerate(LABEL_TO_ID)}

# Constants for the test dataset, don't change these
TEST_DATASET_ROW_NUMBER = 471911  # Number of rows of the test dataset
TEST_DATASET_SIZE_IN_BYTES = 205026320  # Size of the test dataset in bytes
TEST_DATASET_FINGERPRINT = "e1bc80da8b43b9f4"  # Fingerprint of the test dataset