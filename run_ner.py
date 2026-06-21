# Install necessary libraries if you haven't already
# !pip install transformers datasets seqeval torch accelerate -U

import datasets
import numpy as np
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
import torch
import os # Added for creating directories

# --- Label Mappings for tner/mit_restaurant ---
mit_restaurant_label2id = {
    "O": 0,
    "B-Rating": 1,
    "I-Rating": 2,
    "B-Amenity": 3,
    "I-Amenity": 4,
    "B-Location": 5,
    "I-Location": 6,
    "B-Restaurant_Name": 7,
    "I-Restaurant_Name": 8,
    "B-Price": 9,
    "B-Hours": 10,
    "I-Hours": 11,
    "B-Dish": 12,
    "I-Dish": 13,
    "B-Cuisine": 14,
    "I-Price": 15,
    "I-Cuisine": 16
}
mit_restaurant_id2label = {v: k for k, v in mit_restaurant_label2id.items()}
mit_restaurant_label_names = list(mit_restaurant_label2id.keys())

# --- Label Mappings for tner/mit_movie_trivia ---
mit_movie_trivia_label2id = {
    "O": 0,
    "B-Actor": 1,
    "I-Actor": 2,
    "B-Plot": 3,
    "I-Plot": 4,
    "B-Opinion": 5,
    "I-Opinion": 6,
    "B-Award": 7,
    "I-Award": 8,
    "B-Year": 9,
    "B-Genre": 10,
    "B-Origin": 11,
    "I-Origin": 12,
    "B-Director": 13,
    "I-Director": 14,
    "I-Genre": 15,
    "I-Year": 16,
    "B-Soundtrack": 17,
    "I-Soundtrack": 18,
    "B-Relationship": 19,
    "I-Relationship": 20,
    "B-Character_Name": 21,
    "I-Character_Name": 22,
    "B-Quote": 23,
    "I-Quote": 24
}
mit_movie_trivia_id2label = {v: k for k, v in mit_movie_trivia_label2id.items()}
mit_movie_trivia_label_names = list(mit_movie_trivia_label2id.keys())

# --- Label Mappings for tner/conll2003 ---
conll2003_label2id = {
    "O": 0,
    "B-ORG": 1,
    "B-MISC": 2,
    "B-PER": 3,
    "I-PER": 4,
    "B-LOC": 5,
    "I-ORG": 6,
    "I-MISC": 7,
    "I-LOC": 8
}
conll2003_id2label = {v: k for k, v in conll2003_label2id.items()}
conll2003_label_names = list(conll2003_label2id.keys())

# --- Label Mappings for tner/wnut2017 ---
wnut2017_label2id = {
    "B-corporation": 0,
    "B-creative-work": 1,
    "B-group": 2,
    "B-location": 3,
    "B-person": 4,
    "B-product": 5,
    "I-corporation": 6,
    "I-creative-work": 7,
    "I-group": 8,
    "I-location": 9,
    "I-person": 10,
    "I-product": 11,
    "O": 12
}
wnut2017_id2label = {v: k for k, v in wnut2017_label2id.items()}
wnut2017_label_names = list(wnut2017_label2id.keys())


def main(dataset_name="tner/mit_movie_trivia"):
    # --- Configuration ---
    MODEL_CHECKPOINT = "google-bert/bert-base-cased"
    # DATASET_NAME is now passed as an argument
    BATCH_SIZE = 32
    NUM_EPOCHS = 5
    LEARNING_RATE = 2e-5
    
    # Cleaned up directory naming
    safe_dataset_name_for_paths = dataset_name.replace('/', '_')
    OUTPUT_DIR = f"./ckpts/{safe_dataset_name_for_paths}" 
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    QUICK_TEST_MODE = False # Set to True for quick testing 
    SUBSET_SIZE = 100    

    # --- 1. Load Dataset ---
    print(f"\n--- Starting processing for dataset: {dataset_name} ---") # 开始处理数据集
    print(f"Loading dataset: {dataset_name}...") # 加载数据集
    try:
        raw_datasets = datasets.load_dataset(dataset_name)
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}") # 加载数据集出错
        print("Please ensure you have an internet connection and the dataset name is correct.") # 请确保您已连接互联网且数据集名称正确。
        return

    print("Dataset loaded:") # 数据集已加载
    print(raw_datasets)

    if QUICK_TEST_MODE:
        print(f"Running in QUICK TEST MODE: using {SUBSET_SIZE} samples per split.") # 快速测试模式：每个分割使用 {SUBSET_SIZE} 个样本。
        for split in raw_datasets.keys():
            if len(raw_datasets[split]) > 0: 
                 raw_datasets[split] = raw_datasets[split].select(range(min(SUBSET_SIZE, len(raw_datasets[split]))))
            else:
                print(f"Warning: Split '{split}' in {dataset_name} is empty or has too few samples for QUICK_TEST_MODE.") # 警告：{dataset_name} 中的分割 '{split}' 为空或样本过少，无法进行快速测试模式。
        print("Subset selected:") # 已选择子集
        print(raw_datasets)

    # --- 2. Prepare Labels (Using your provided mappings) ---
    print(f"\nPreparing labels for dataset: {dataset_name}...") # 准备数据集标签
    if dataset_name == "tner/mit_restaurant":
        label_names = mit_restaurant_label_names
        id2label = mit_restaurant_id2label
        label2id = mit_restaurant_label2id
    elif dataset_name == "tner/mit_movie_trivia":
        label_names = mit_movie_trivia_label_names
        id2label = mit_movie_trivia_id2label
        label2id = mit_movie_trivia_label2id
    elif dataset_name == "tner/conll2003": 
        label_names = conll2003_label_names
        id2label = conll2003_id2label
        label2id = conll2003_label2id
    elif dataset_name == "tner/wnut2017": # Added WNUT2017
        label_names = wnut2017_label_names
        id2label = wnut2017_id2label
        label2id = wnut2017_label2id
    else:
        print(f"Warning: Dataset {dataset_name} not recognized for custom label mapping. " # 警告：数据集 {dataset_name} 未被识别用于自定义标签映射。
              "Attempting to derive labels from dataset features.") # 尝试从数据集特征派生标签。
        train_split_present = "train" in raw_datasets and len(raw_datasets["train"]) > 0
        if not train_split_present:
            print("Error: 'train' split is missing or empty. Cannot determine NER features automatically.") # 错误：'train' 分割缺失或为空。无法自动确定 NER 特征。
            available_split = next((s for s in ["validation", "test", "dev"] if s in raw_datasets and len(raw_datasets[s]) > 0), None)
            if available_split:
                print(f"Attempting to use features from '{available_split}' split for {dataset_name}.") # 尝试使用 {dataset_name} 中 '{available_split}' 分割的特征。
                ner_feature = raw_datasets[available_split].features["tags"] 
            else:
                print(f"Error: No suitable split found in {dataset_name} to determine NER features. Exiting.") # 错误：在 {dataset_name} 中找不到合适的分区来确定 NER 特征。正在退出。
                return
        else:
            ner_feature = raw_datasets["train"].features["tags"] 
        
        if hasattr(ner_feature, 'feature') and hasattr(ner_feature.feature, 'names'):
            label_names = ner_feature.feature.names
            id2label = {i: label for i, label in enumerate(label_names)}
            label2id = {label: i for i, label in enumerate(label_names)}
        else:
            print(f"Error: Could not derive label names from dataset features for {dataset_name}. " # 错误：无法从 {dataset_name} 的数据集特征中派生标签名称。
                  "Please ensure the dataset has 'tags' with a 'ClassLabel' feature or provide mappings.") # 请确保数据集具有带 'ClassLabel' 特征的 'tags' 或提供映射。
            return

    num_labels = len(label_names)

    print(f"\nNumber of labels for {dataset_name}: {num_labels}") # {dataset_name} 的标签数量：{num_labels}
    print(f"Label names: {label_names}") # 标签名称

    # --- 3. Load Tokenizer ---
    print(f"\nLoading tokenizer for checkpoint: {MODEL_CHECKPOINT}...") # 加载检查点的分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    # --- 4. Preprocessing Function ---
    # This function assumes 'tokens' is a list of words and 'tags' is a list of numerical IDs.
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"], # Assumes this is the column with lists of words
            truncation=True,
            is_split_into_words=True, 
            padding=False 
        )
        labels = []
        for i, label_sequence in enumerate(examples["tags"]): # Assumes this is the column with lists of numerical tag IDs
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None: 
                    label_ids.append(-100)
                elif word_idx != previous_word_idx: 
                    label_ids.append(label_sequence[word_idx])
                else: 
                    label_ids.append(-100) 
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # --- 5. Apply Preprocessing ---
    print("\nApplying preprocessing to the dataset...") # 对数据集应用预处理
    remove_cols = []
    # Determine columns to remove based on available splits
    for split_name in ["train", "validation", "dev", "test"]: # Common split names
        if split_name in raw_datasets and len(raw_datasets[split_name]) > 0:
            remove_cols = raw_datasets[split_name].column_names
            break # Found columns from the first available split
    if not remove_cols:
         print(f"Warning: Could not determine columns to remove for {dataset_name} as all splits seem empty or missing.") # 警告：由于所有分割似乎都为空或缺失，因此无法确定要为 {dataset_name} 删除的列。


    tokenized_datasets = raw_datasets.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=remove_cols 
    )
    print("\nPreprocessing complete.") # 预处理完成

    # --- 6. Data Collator ---
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # --- 7. Load Model ---
    print(f"\nLoading model for token classification: {MODEL_CHECKPOINT}...") # 加载用于令牌分类的模型
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=num_labels,
        id2label=id2label, 
        label2id=label2id, 
        ignore_mismatched_sizes=True 
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded on device: {device}") # 模型已加载到设备

    # --- 8. Define Metrics Computation ---
    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        predictions = np.argmax(predictions, axis=2)

        # Convert IDs to label strings, ignoring -100
        true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        all_metrics = {
            "precision": precision_score(true_labels, true_predictions, average='macro', zero_division=0),
            "recall": recall_score(true_labels, true_predictions, average='macro', zero_division=0),   
            "f1": f1_score(true_labels, true_predictions, average='macro', zero_division=0),            
        }
        return all_metrics

    # --- 9. Training Arguments ---
    print("\nDefining training arguments...") # 定义训练参数
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch", 
        save_strategy="epoch",       
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True, 
        metric_for_best_model="f1", 
        report_to="none", 
        disable_tqdm=False, 
    )

    # --- 10. Initialize Trainer ---
    print("\nInitializing Trainer...") # 初始化训练器
    
    train_dataset = tokenized_datasets.get("train")
    # WNUT2017 also typically has 'validation' (often called 'dev') and 'test' splits.
    eval_dataset = tokenized_datasets.get("validation") 
    if eval_dataset is None: 
        eval_dataset = tokenized_datasets.get("dev") # 'dev' is common for WNUT2017 validation
    if eval_dataset is None and not QUICK_TEST_MODE : 
        print("Validation/dev split not found, attempting to use test split for evaluation during training.") # 未找到验证/开发拆分，尝试在训练期间使用测试拆分进行评估。
        eval_dataset = tokenized_datasets.get("test")

    if train_dataset is None or len(train_dataset) == 0 :
        print(f"ERROR: Training dataset for {dataset_name} is not available or empty. Skipping training.") # 错误：{dataset_name} 的训练数据集不可用或为空。跳过训练。
        return
    if (eval_dataset is None or len(eval_dataset) == 0) and not QUICK_TEST_MODE:
        print(f"WARNING: Evaluation dataset for {dataset_name} is not available or empty. Training will proceed without intermediate evaluation.") # 警告：{dataset_name} 的评估数据集不可用或为空。训练将在没有中间评估的情况下进行。

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if (eval_dataset and len(eval_dataset) > 0) else None, 
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # --- 11. Train ---
    print(f"\nStarting training for {dataset_name}...") # 开始训练 {dataset_name}
    try:
        if train_dataset and len(train_dataset) > 0:
            train_result = trainer.train()
            print("Training completed.") # 训练完成
            trainer.save_model() 
            trainer.save_state() 
            print(f"Model, tokenizer, and training state saved to {OUTPUT_DIR}") # 模型、分词器和训练状态已保存到 {OUTPUT_DIR}

        else:
            print(f"Skipping training for {dataset_name} as train_dataset is not available or empty.") # 由于 train_dataset 不可用或为空，跳过 {dataset_name} 的训练。

    except Exception as e:
        print(f"An error occurred during training for {dataset_name}: {e}") # {dataset_name} 训练期间发生错误
        if "CUDA out of memory" in str(e):
            print("CUDA out of memory. Try reducing BATCH_SIZE.") # CUDA 内存不足。尝试减少 BATCH_SIZE。
        return

    # --- 12. Evaluate on Test Set (if available) ---
    test_dataset = tokenized_datasets.get("test")
    if test_dataset and len(test_dataset) > 0:
        print(f"\nEvaluating on the test set for {dataset_name}...") # 在 {dataset_name} 的测试集上进行评估
        try:
            test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
            print("Test set evaluation completed.") # 测试集评估完成
            print(f"\n--- Test Set Metrics for {dataset_name} ---") # {dataset_name} 的测试集指标
            print(f"  Test Precision: {test_metrics.get('test_precision', 'N/A'):.4f}") # 测试精度
            print(f"  Test Recall:    {test_metrics.get('test_recall', 'N/A'):.4f}") # 测试召回率
            print(f"  Test F1-Score:  {test_metrics.get('test_f1', 'N/A'):.4f}") # 测试 F1 分数
        except Exception as e:
            print(f"An error occurred during test set evaluation for {dataset_name}: {e}") # {dataset_name} 测试集评估期间发生错误
    else:
        print(f"\nTest set not found or is empty in {dataset_name}. Skipping final evaluation on test set.") # 在 {dataset_name} 中未找到测试集或测试集为空。跳过对测试集的最终评估。

    print(f"\n--- Script Finished for {dataset_name} ---") # {dataset_name} 脚本已完成

if __name__ == "__main__":
    # Run for MIT Movie Trivia dataset
    # main(dataset_name="tner/mit_movie_trivia")
    
    # # Run for MIT Restaurant dataset
    # main(dataset_name="tner/mit_restaurant")

    # # Run for CoNLL2003 dataset
    # main(dataset_name="tner/conll2003")

    # Run for WNUT2017 dataset
    main(dataset_name="tner/mit_movie_trivia")

    # You can comment out lines if you only want to run for specific datasets.
