from transformers import AutoModelForTokenClassification
from constants import MODEL_CHECKPOINT, ID_TO_LABEL, LABEL_TO_ID

def initialize_model():
    """
    Initialize a model for token classification.

    Returns:
        A model for token classification.

    NOTE: Below is an example of how to initialize a model for token classification.

    from transformers import AutoModelForTokenClassification
    from constants import ID_TO_LABEL, LABEL_TO_ID, MODEL_CHECKPOINT

    model = AutoModelForTokenClassification.from_pretrained(
        pretrained_model_name_or_path=MODEL_CHECKPOINT,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    You are free to change this.
    But make sure the model meets the requirements of the `transformers.Trainer` API.
    ref: https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer
    """
    # Write your code here.

    model = AutoModelForTokenClassification.from_pretrained(
        pretrained_model_name_or_path=MODEL_CHECKPOINT,
        num_labels=len(LABEL_TO_ID),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    return model
