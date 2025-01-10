import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset
import sentencepiece as spm

from tqdm import tqdm
import math
import os

import nltk
nltk.download('punkt')
from nltk.translate.bleu_score import corpus_bleu

from dataset import IWSLT16EnDeDataset
from model.transformer import Transformer

dataset = load_dataset('iwslt2017', 'iwslt2017-en-de')
    
# 데이터셋 및 데이터로더 생성
sp_de = spm.SentencePieceProcessor()
sp_de.load('spm_de.model')

sp_en = spm.SentencePieceProcessor()
sp_en.load('spm_en.model')

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    
    # 패딩
    src_padded = nn.utils.rnn.pad_sequence(src_batch, padding_value=sp_de.pad_id(), batch_first=True)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_batch, padding_value=sp_en.pad_id(), batch_first=True)
    
    return src_padded, tgt_padded

batch_size = 32

train_dataset = IWSLT16EnDeDataset(dataset['train'], sp_de, sp_en, max_len=100)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

validation_dataset = IWSLT16EnDeDataset(dataset['validation'], sp_de, sp_en, max_len=100)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

test_dataset = IWSLT16EnDeDataset(dataset['test'], sp_de, sp_en, max_len=100)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


# 파라미터 설정
n_layer = 6
enc_vocab_size = sp_de.get_piece_size()
dec_vocab_size = sp_en.get_piece_size()
max_len = 100
d_model = 512
ffn_hidden = 2048
n_head = 8
dropout = 0.1
device = torch.device("mps")

# 모델 초기화
model = Transformer(
    n_layer=n_layer,
    enc_vocab_size=enc_vocab_size,
    dec_vocab_size=dec_vocab_size,
    max_len=max_len,
    d_model=d_model,
    ffn_hidden=ffn_hidden,
    n_head=n_head,
    device=device,
    dropout=dropout
).to(device)

# 패딩 토큰 인덱스
pad_idx = sp_en.pad_id()

# 손실 함수 정의
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

# 옵티마이저 정의 (Transformer 논문에서 제안된 Adam 사용)
optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001,
    betas=(0.9, 0.98),
    eps=1e-9
)

def get_lr_schedule(d_model, warmup_steps=4000):
    def lr_lambda(step):
        step += 1  # step 0을 고려하여 1을 더함
        return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)
    return lr_lambda

# 스케줄러 정의
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_schedule(d_model=512, warmup_steps=4000))

def generate_subsequent_mask(size):
    """
    타겟 시퀀스에서 미래 토큰을 참조하지 않도록 마스크를 생성
    Args:
        size: 시퀀스 길이
    Returns:
        마스크 텐서 [1, 1, size, size]
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask.unsqueeze(0).unsqueeze(1).to(device)  # [1, 1, size, size]

def train_epoch(model, dataloader, optimizer, criterion, scheduler, device):
    model.train()
    epoch_loss = 0

    for src, tgt in tqdm(dataloader, desc="Training"):
        src = src.to(device)
        tgt = tgt.to(device)
        
        # 타겟 입력과 출력 분리
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:].contiguous().view(-1)
        
        # 마스크 생성
        tgt_mask = generate_subsequent_mask(tgt_input.size(1))  # [1, 1, tgt_seq_len-1, tgt_seq_len-1]
        
        # 옵티마이저 초기화
        optimizer.zero_grad()
        
        # 포워드 패스
        output = model(src, tgt_input, tgt_mask)  # [batch_size, tgt_seq_len-1, dec_vocab_size]
        
        # 손실 계산
        output = output.view(-1, output.size(-1))  # [batch_size*(tgt_seq_len-1), dec_vocab_size]
        loss = criterion(output, tgt_output)
        
        # 역전파
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 옵티마이저 스텝 및 스케줄러 업데이트
        optimizer.step()
        scheduler.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc="Evaluating"):
            src = src.to(device)
            tgt = tgt.to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:].contiguous().view(-1)
            
            tgt_mask = generate_subsequent_mask(tgt_input.size(1))
            
            output = model(src, tgt_input, tgt_mask)  # [batch_size, tgt_seq_len-1, dec_vocab_size]
            
            output = output.view(-1, output.size(-1))  # [batch_size*(tgt_seq_len-1), dec_vocab_size]
            loss = criterion(output, tgt_output)
            
            epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)

def compute_bleu(model, dataloader, sp_src, sp_tgt, device):
    model.eval()
    references = []
    hypotheses = []

    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc="Computing BLEU"):
            src = src.to(device)
            tgt = tgt.to(device)
            
            tgt_input = tgt[:, :-1]
            # Greedy Decoding: 가장 높은 확률의 토큰 선택
            output = model(src, tgt_input, generate_subsequent_mask(tgt_input.size(1)).to(device))  # [batch_size, tgt_seq_len-1, dec_vocab_size]
            output = output.argmax(dim=-1)  # [batch_size, tgt_seq_len-1]
            
            for ref, hyp in zip(tgt[:, 1:], output):
                ref_tokens = sp_tgt.decode(ref.tolist()).split()
                hyp_tokens = sp_tgt.decode(hyp.tolist()).split()
                references.append([ref_tokens])
                hypotheses.append(hyp_tokens)
    
    bleu = corpus_bleu(references, hypotheses)
    return bleu

num_epochs = 10

for epoch in range(1, num_epochs + 1):
    print(f"Epoch {epoch}/{num_epochs}")
    
    # 학습
    train_loss = train_epoch(model, train_loader, optimizer, criterion, scheduler, device)
    print(f"Training Loss: {train_loss:.4f}")
    
    # 평가
    val_loss = evaluate(model, validation_loader, criterion, device)
    print(f"Validation Loss: {val_loss:.4f}")
    
    # BLEU 점수 계산
    bleu_score = compute_bleu(model, validation_loader, sp_de, sp_en, device)
    print(f"Validation BLEU Score: {bleu_score * 100:.2f}")
    
    # 모델 저장
    torch.save(model.state_dict(), f'transformer_en_de_epoch_{epoch}.pth')