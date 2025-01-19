
### Fine-tuning a Pretrained Model

Fine-tuning a pretrained model reduces computation costs, carbon footprint, and allows leveraging state-of-the-art models for specific tasks.

**Steps:**

1. **Prepare Dataset:**
   - Load a dataset (e.g., Yelp Reviews dataset) using `datasets` library.

   ```python
   from datasets import load_dataset
   dataset = load_dataset("yelp_review_full")
   ```

2. **Tokenizer:**
   - Use the `AutoTokenizer` to tokenize the text data, and apply padding and truncation strategies to handle varying sequence lengths.

   ```python
   from transformers import AutoTokenizer
   tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
   
   def tokenize_function(examples):
       return tokenizer(examples["text"], padding="max_length", truncation=True)
   
   tokenized_datasets = dataset.map(tokenize_function, batched=True)
   ```

3. **Create Subset (Optional):**
   - Optionally, create a smaller dataset for faster fine-tuning.

   ```python
   small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
   small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
   ```

4. **Fine-tuning with PyTorch**:

   - Load the model for sequence classification with the correct number of labels (e.g., 5 labels for Yelp Review dataset).

   ```python
   from transformers import AutoModelForSequenceClassification
   model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5, torch_dtype="auto")
   ```

5. **Training Arguments:**
   - Define training arguments such as output directory and evaluation strategy.

   ```python
   from transformers import TrainingArguments
   training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch")
   ```

6. **Compute Metrics:**
   - Define how evaluation metrics will be calculated during training.

   ```python
   import numpy as np
   import evaluate

   metric = evaluate.load("accuracy")

   def compute_metrics(eval_pred):
       logits, labels = eval_pred
       predictions = np.argmax(logits, axis=-1)
       return metric.compute(predictions=predictions, references=labels)
   ```

7. **Create Trainer:**
   - Instantiate a `Trainer` to manage the training process.

   ```python
   from transformers import Trainer
   trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=small_train_dataset,
       eval_dataset=small_eval_dataset,
       compute_metrics=compute_metrics
   )
   ```

8. **Train:**
   - Fine-tune the model by calling the `train()` method.

   ```python
   trainer.train()
   ```

9. **TensorFlow Fine-tuning**:

   - Use Keras API for fine-tuning with TensorFlow, which includes tokenizing data and fitting the model.

   ```python
   from transformers import TFAutoModelForSequenceClassification
   from tensorflow.keras.optimizers import Adam

   model = TFAutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased")
   model.compile(optimizer=Adam(3e-5))
   model.fit(tokenized_data, labels)
   ```

10. **Native PyTorch Fine-tuning**:

    - Fine-tune using PyTorch by manually handling tokenized datasets and using a custom training loop.

    ```python
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from transformers import get_scheduler
    import torch
    ```

    Training Loop:

    ```python
    from tqdm.auto import tqdm
    progress_bar = tqdm(range(num_training_steps))
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
    ```

    **Evaluation:**

    ```python
    metric = evaluate.load("accuracy")
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    metric.compute()
    ```

