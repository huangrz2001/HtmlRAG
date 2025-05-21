import os
import json
import torch
import wandb
from accelerate import Accelerator
from sklearn.model_selection import train_test_split
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CrossEncoderClassificationEvaluator
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# ========== 参数 ==========
MODEL_NAME = "/data/huangruizhi/htmlRAG/bce-reranker-base_v1"
DATA_PATH = "reranker_qa_dataset.jsonl"
SAVE_DIR = "./trained_reranker_accelerated"
PROJECT_NAME = "reranker-bce"
RUN_NAME = "bce-4090-run"
MAX_LENGTH = 512
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 2e-5
WARMUP_STEPS = 500
USE_AMP = True
EARLY_STOP_PATIENCE = 2

# ========== 初始化 ==========
accelerator = Accelerator(mixed_precision="fp16" if USE_AMP else "no")
device = accelerator.device
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def truncate_to_max_length(query, doc, max_length=512):
    tokens = tokenizer.encode_plus(query, doc, truncation=True, max_length=max_length, return_token_type_ids=False, return_attention_mask=False)
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

# ========== wandb ==========
if accelerator.is_main_process:
    wandb.init(project=PROJECT_NAME, name=RUN_NAME)

# ========== 加载数据 ==========
data = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        q, d = truncate_to_max_length(item["query"], item["doc"], MAX_LENGTH)
        data.append(InputExample(texts=[q, d], label=float(item["label"])))

train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn)
val_dataloader = DataLoader(val_data, shuffle=False, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn)

# ========== 初始化模型 ==========
model = CrossEncoder(model_name_or_path=MODEL_NAME, num_labels=1, max_length=MAX_LENGTH, device=device)
optimizer = torch.optim.AdamW(model.model.parameters(), lr=LEARNING_RATE)

# ========== 包装器 ==========
model.model, optimizer, train_dataloader = accelerator.prepare(model.model, optimizer, train_dataloader)

# ========== 验证函数 ==========
def evaluate_on_val(model, val_data):
    model.eval()
    scorer = CrossEncoderClassificationEvaluator(
        sentence_pairs=[[ex.texts[0], ex.texts[1]] for ex in val_data],
        labels=[ex.label for ex in val_data],
        name="dev-set-eval"
    )
    acc = scorer(model)
    return acc

# ========== 训练循环 ==========
best_acc = 0.0
patience = 0
model.model.train()

for epoch in range(EPOCHS):
    accelerator.print(f"🔁 Epoch {epoch + 1}/{EPOCHS}")
    total_loss = 0

    for step, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        outputs = model.model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        total_loss += loss.item()

        if step % 10 == 0 and accelerator.is_main_process:
            wandb.log({"train_loss": loss.item(), "epoch": epoch})
            accelerator.print(f"Step {step}, Loss: {loss.item():.4f}")

    # 验证
    if accelerator.is_main_process:
        acc = evaluate_on_val(model, val_data)
        wandb.log({"val_acc": acc, "epoch": epoch})
        accelerator.print(f"✅ Epoch {epoch+1} 验证准确率: {acc:.4f}")

        # early stopping
        if acc > best_acc:
            best_acc = acc
            patience = 0
            model.save(os.path.join(SAVE_DIR, "best_model"))
            accelerator.print(f"📌 保存最佳模型 (val acc = {acc:.4f})")
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                accelerator.print("⏹️ 早停触发，结束训练")
                break

# ========== 最终模型保存 ==========
if accelerator.is_main_process:
    model.save(SAVE_DIR)
    print(f"✅ 最终模型保存至: {SAVE_DIR}")
