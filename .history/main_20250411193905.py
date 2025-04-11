import os
import torch
from dataset import build_dataset, preprocess_data
from model import initialize_model
from tokenizer import initialize_tokenizer
from trainer import build_trainer
from utils import not_change_test_dataset, set_random_seeds
from constants import OUTPUT_DIR

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_latest_checkpoint(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        return None
    return os.path.join(output_dir, max(checkpoints, key=lambda x: int(x.split("-")[1])))


def main():
    set_random_seeds()
    model = initialize_model()
    tokenizer = initialize_tokenizer()
    raw_datasets = build_dataset()
    assert not_change_test_dataset(raw_datasets), "You should not change the test dataset"
    tokenized_datasets = preprocess_data(raw_datasets, tokenizer)
    trainer = build_trainer(model=model, tokenizer=tokenizer, tokenized_datasets=tokenized_datasets)

    latest_checkpoint = get_latest_checkpoint(OUTPUT_DIR)
    if latest_checkpoint:
        print(f"Resuming from checkpoint: {latest_checkpoint}")
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        print("Starting training from scratch")
        trainer.train()

    best_model_path = trainer.state.best_model_checkpoint
    if best_model_path and os.path.exists(best_model_path):
        trainer.model = initialize_model().from_pretrained(best_model_path)
        trainer.model.to("cuda:0")  # 显式移到 GPU
        print(f"Loaded best model from {best_model_path}")
    else:
        print("Best model not found, using final model")
        trainer.model.to("cuda:0")  # 确保最终模型也在 GPU

    test_metrics = trainer.evaluate(eval_dataset=tokenized_datasets["test"], metric_key_prefix="test")
    print("Test Metrics:", test_metrics)


if __name__ == "__main__":
    main()