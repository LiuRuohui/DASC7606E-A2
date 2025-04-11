from transformers import DataCollatorForTokenClassification, Trainer, TrainingArguments
from constants import OUTPUT_DIR
from evaluation import compute_metrics

def create_training_arguments() -> TrainingArguments:
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        load_best_model_at_end=True,
        push_to_hub=False,
        per_device_train_batch_size=8,  # 修改：从 4 增加到 8，利用更多 GPU 内存
        per_device_eval_batch_size=4,  # 修改：从 2 增加到 4，提高评估效率
        gradient_accumulation_steps=2,  # 修改：从 4 减少到 2，与 batch_size 调整匹配
        fp16=True,  # 保持：启用混合精度训练，加速计算
        gradient_checkpointing=True,  # 保持：节省内存，适合大模型
        eval_accumulation_steps=20,  # 修改：从 50 减少到 20，减少评估时的内存压力
        num_train_epochs=3,  # 修改：从 2 增加到 3，增加训练轮次提升 F1
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,  # 修改：从 1000 减少到 500，更频繁保存检查点
        save_total_limit=20,  # 修改：从 2 减少到 1，减少存储开销
        logging_steps=50,  # 修改：从 100 减少到 50，更频繁记录训练过程
        learning_rate=2e-5,  # 修改：从 3e-5 降低到 2e-5，避免过快收敛影响 F1
        warmup_ratio=0.05,  # 修改：从 0.1 减少到 0.05，缩短预热时间
        weight_decay=0.01,  # 保持：正则化强度适当
        lr_scheduler_type="cosine",  # 保持：余弦调度适合 NER 任务
        label_smoothing_factor=0.1,  # 保持：平滑标签有助于泛化
        metric_for_best_model="f1",
        greater_is_better=True,
        dataloader_num_workers=4,  # 修改：从 8 减少到 4，避免过多线程开销
        dataloader_pin_memory=True,  # 保持：加速数据加载
        dataloader_prefetch_factor=4,  # 修改：从 2 增加到 4，进一步优化数据预取
        save_safetensors=True,  # 保持：安全保存模型
    )
    return training_args


def build_trainer(model, tokenizer, tokenized_datasets) -> Trainer:
    """
    Build and return the trainer object for training and evaluation.

    Args:
        model: Model for token classification.
        tokenizer: Tokenizer object.
        tokenized_datasets: Tokenized datasets.

    Returns:
        Trainer object for training and evaluation.
    """
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,  # 修改：确保与 dataset.py 的动态填充一致
    )
    training_args: TrainingArguments = create_training_arguments()

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )