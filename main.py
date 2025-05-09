'''distillation_pipeline.py: Скелет скрипта для дистилляции и оценки ruMTEB

1. Загрузка датасета Gazeta
2. Определение student-трансформера с проекцией на размерность учителя
3. Цикл обучения дистилляции
4. Embedders и evaluate_rumteb
'''

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, T5EncoderModel
from mteb import MTEB, get_tasks

# 1. Загрузка датасета

def load_gazeta(batch_size=32, split='train'):
    ds = load_dataset('IlyaGusev/gazeta', split=split)
    tokenizer = AutoTokenizer.from_pretrained("ai-forever/FRIDA", use_fast=True)
    def tokenize(example):
        return tokenizer(example['text'], truncation=True, padding='max_length', max_length=256)
    ds = ds.map(tokenize, batched=True)
    ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

# 2. StudentTransformer с проекцией
class StudentTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=768, num_layers=6, num_heads=4, teacher_dim=None):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.projection = nn.Linear(embed_dim, teacher_dim) if teacher_dim else None

    def forward(self, input_ids, attention_mask=None):
        x = self.token_emb(input_ids)
        x = x.permute(1, 0, 2)
        attn_mask = ~attention_mask.bool() if attention_mask is not None else None
        x = self.transformer(x, src_key_padding_mask=attn_mask)
        x = x.permute(1, 0, 2)
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1, keepdim=True)
            pooled = (x * attention_mask.unsqueeze(-1)).sum(dim=1) / lengths
        else:
            pooled = x.mean(dim=1)
        return self.projection(pooled) if self.projection else pooled

# 3. Дистилляция

def distillation_train(student, teacher, dataloader, optimizer, device):
    student.to(device)
    teacher.to(device)
    teacher.eval()
    mse = nn.MSELoss()
    kl = nn.KLDivLoss(reduction='batchmean')

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        with torch.no_grad():
            t_out = teacher(input_ids=input_ids, attention_mask=attention_mask)
            t_emb = t_out.last_hidden_state[:, 0]
        s_emb = student(input_ids, attention_mask)
        loss_mse = mse(s_emb, t_emb)
        loss_kl = kl(F.log_softmax(s_emb, dim=-1), F.softmax(t_emb, dim=-1))
        loss = loss_mse + loss_kl
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()

# 4. Embedders и оценка
frida_prompts = {
            "Classification": "categorize: ",
            "MultilabelClassification": "categorize: ",
            "Clustering": "categorize_topic: ",
            "PairClassification": "paraphrase: ",
            "Reranking": "paraphrase: ",
            "Reranking-query": "search_query: ",
            "Reranking-passage": "search_document: ",
            "STS": "paraphrase: ",
            "Summarization": "categorize: ",
            "query": "search_query: ",
            "passage": "search_document: ",
            "CEDRClassification": "categorize_sentiment: ",
            "GeoreviewClassification": "categorize_sentiment: ",
            "HeadlineClassification": "categorize_topic: ",
            "InappropriatenessClassification": "categorize_topic: ",
            "KinopoiskClassification": "categorize_sentiment: ",
            "MassiveIntentClassification": "paraphrase: ",
            "MassiveScenarioClassification": "paraphrase: ",
            "RuReviewsClassification": "categorize_sentiment: ",
            "RuSciBenchGRNTIClassification": "categorize_topic: ",
            "RuSciBenchOECDClassification": "categorize_topic: ",
            "SensitiveTopicsClassification": "categorize_topic: ",
            "TERRa": "categorize_entailment: ",
            "RiaNewsRetrieval": "categorize: ",
            None: "categorize: "
        }

class TeacherEmbedder:
    def __init__(self, model, tokenizer, max_length=512):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def pool(self, hidden, mask, method="cls"):
        return (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).float() if method=="mean" else hidden[:, 0]

    def encode(self, texts, batch_size=128, task_name=None, task_type=None):
        task = task_type or task_name
        prompt = frida_prompts.get(task, "categorize: ")
        all_emb = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Teacher encoding"):
            batch = [f"{prompt}{t}" for t in texts[i:i+batch_size]]
            enc = self.tokenizer(batch, padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt")
            device = next(self.model.parameters()).device
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.model.encoder(input_ids=enc['input_ids'],
                                         attention_mask=enc['attention_mask'])
                emb = self.pool(out.last_hidden_state, enc['attention_mask'], method="cls")
                all_emb.append(F.normalize(emb, p=2, dim=-1).cpu())
        return torch.cat(all_emb, dim=0)

class StudentEmbedder:
    def __init__(self, model, tokenizer, max_length=512):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def encode(self, texts, batch_size=128, task_name=None, task_type=None):
        task = task_type or task_name
        prompt = frida_prompts.get(task, "categorize: ")
        all_emb = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Student encoding"):
            batch = [f"{prompt}{t}" for t in texts[i:i+batch_size]]
            enc = self.tokenizer(batch, padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt")
            device = next(self.model.parameters()).device
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                emb = self.model(enc['input_ids'], enc['attention_mask'])
                all_emb.append(F.normalize(emb, p=2, dim=-1).cpu())
        return torch.cat(all_emb, dim=0)


def evaluate_rumteb(model, tokenizer, limit=50):
    """
    Loads the ruMTEB (Russian part of MTEB) lightweight subset and evaluates both teacher
    and student representations. Each pathway is wrapped in its corresponding embedder
    (which applies prompt addition, tokenization, CLS pooling, and normalization), and then
    the MTEB evaluation object (filtered to Russian tasks via task_langs=["ru"]) is run on each.
    A side-by-side summary of key evaluation scores is printed.
    """
    # Instantiate embedding wrappers for teacher and student with the updated encode() methods.
    teacher_embedder = TeacherEmbedder(model, tokenizer, max_length=512)
    student_embedder = StudentEmbedder(model, tokenizer, max_length=512)

    # Create the MTEB evaluation object for Russian tasks, limiting examples for a lightweight run.
    evaluation = MTEB(tasks=["GeoreviewClusteringP2P"])
    
    print("Evaluating teacher embeddings on ruMTEB lightweight subset...")
    teacher_results = evaluation.run(teacher_embedder, output_folder="results/teacher_rumteb")
    
    print("Evaluating student embeddings on ruMTEB lightweight subset...")
    student_results = evaluation.run(student_embedder, output_folder="results/student_rumteb")
    
    # Print a side-by-side comparison of evaluation metrics for each task.
    print("\nComparison of teacher vs. student on ruMTEB tasks:")
    header = f"{'Task':30} | {'Teacher Score':15} | {'Student Score':15}"
    divider = "-" * len(header)
    print(header)
    print(divider)
    teacher_score = teacher_results[0].scores["test"][0].get("main_score")
    student_score = student_results[0].scores["test"][0].get("main_score")
    print(f"{'':30} | {str(teacher_score):15} | {str(student_score):15}")
    
    return teacher_score, student_score

# 5. main()
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained("ai-forever/FRIDA", use_fast=True)
    teacher = T5EncoderModel.from_pretrained("ai-forever/FRIDA").to(device)
    student = StudentTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_dim=768,
        num_layers=6,
        num_heads=4,
        teacher_dim=teacher.config.hidden_size
    ).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
    dl = load_gazeta()
    for ep in range(1):
        loss = distillation_train(student, teacher, dl, optimizer, device)
        print(f"Epoch {ep+1}: loss={loss:.4f}")
    torch.save(student.state_dict(), 'student_frida.pt')
    evaluate_rumteb(teacher, student, tokenizer)

if __name__ == '__main__':
    main()
