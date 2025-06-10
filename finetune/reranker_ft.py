import os
import json
import torch
import wandb
from sklearn.model_selection import train_test_split
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CrossEncoderClassificationEvaluator
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, set_seed
from tqdm.auto import tqdm

# ========== 参数 ========== #
MODEL_NAME = "/data/huangruizhi/htmlRAG/bce-reranker-base_v1"
DATA_PATH = "reranker_qa_dataset.jsonl"
SAVE_DIR = "./trained_reranker_single"
PROJECT_NAME = "reranker-bce"
RUN_NAME = "bce-4090-single"
MAX_LENGTH = 512
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 2e-5
EARLY_STOP_PATIENCE = 1
USE_AMP = True

# ========== 设置环境 ========== #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(42)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ========== 数据预处理 ========== #
def truncate_to_max_length(query, doc, max_length=512):
    tokens = tokenizer.encode_plus(query, doc, truncation=True, max_length=max_length)
    decoded = tokenizer.decode(tokens["input_ids"], skip_special_tokens=True)
    parts = decoded.split(tokenizer.sep_token)
    return parts[0], parts[1] if len(parts) > 1 else ""

def custom_collate_fn(batch):
    queries = [item.texts[0] for item in batch]
    docs = [item.texts[1] for item in batch]
    labels = torch.tensor([item.label for item in batch], dtype=torch.float)
    encoded = tokenizer(queries, docs, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    encoded["labels"] = labels
    return encoded

# ========== wandb 初始化 ========== #
wandb.init(project=PROJECT_NAME, name=RUN_NAME)

# ========== 加载数据 ========== #
data = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        q, d = truncate_to_max_length(item["query"], item["doc"], MAX_LENGTH)
        data.append(InputExample(texts=[q, d], label=float(item["label"])))

train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn)
val_dataloader = DataLoader(val_data, shuffle=False, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn)

# ========== 初始化模型 ========== #
model = CrossEncoder(model_name_or_path=MODEL_NAME, num_labels=1, max_length=MAX_LENGTH, device=device)
optimizer = torch.optim.AdamW(model.model.parameters(), lr=LEARNING_RATE)
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

# ========== 验证函数 ========== #
def evaluate_on_val(model, val_data):
    model.eval()
    scorer = CrossEncoderClassificationEvaluator(
        sentence_pairs=[[ex.texts[0], ex.texts[1]] for ex in val_data],
        labels=[ex.label for ex in val_data],
        name="dev-set-eval"
    )
    return scorer(model)

# ========== 训练循环 ========== #
best_acc = 0.0
patience = 0

for epoch in range(EPOCHS):
    print(f"\n🔁 Epoch {epoch + 1}/{EPOCHS}")
    model.model.train()
    total_loss = 0.0

    progress_bar = tqdm(train_dataloader, total=len(train_dataloader))
    for step, batch in enumerate(progress_bar):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            outputs = model.model(**batch)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        progress_bar.set_description(f"Loss: {loss.item():.4f}")

        if step % 10 == 0:
            wandb.log({"train_loss": loss.item(), "epoch": epoch})

    # ========== 验证 + 早停 ========== #
    acc = evaluate_on_val(model, val_data)
    wandb.log({"val_acc": acc['dev-set-eval_accuracy'], "epoch": epoch})
    print(f"✅ Epoch {epoch + 1} 验证准确率: {acc['dev-set-eval_accuracy']:.4f}")

    if acc['dev-set-eval_accuracy'] > best_acc:
        best_acc = acc['dev-set-eval_accuracy']
        patience = 0
        model.save(os.path.join(SAVE_DIR, "best_model"))
        print(f"📌 保存最佳模型 (val acc = {acc['dev-set-eval_accuracy']:.4f})")
    else:
        patience += 1
        if patience >= EARLY_STOP_PATIENCE:
            print("⏹️ 早停触发，结束训练")
            break


# ========== 保存最终模型 ========== #
model.save(SAVE_DIR)
print(f"✅ 最终模型保存至: {SAVE_DIR}")
