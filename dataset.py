import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

from datasets import load_dataset

dataset = load_dataset('iwslt2017', 'iwslt2017-en-de')

class IWSLT16EnDeDataset(Dataset):
    def __init__(self, dataset_split, sp_src, sp_tgt, max_len=100):
        """
        Args:
            dataset_split: datasets 라이브러리의 데이터 스플릿 (train, validation, test)
            sp_src: 소스 언어 (독일어) SentencePiece 모델
            sp_tgt: 타겟 언어 (영어) SentencePiece 모델
            max_len: 시퀀스의 최대 길이
        """
        self.src_texts = [example['translation']['de'] for example in dataset_split]
        self.tgt_texts = [example['translation']['en'] for example in dataset_split]
        self.sp_src = sp_src
        self.sp_tgt = sp_tgt
        self.max_len = max_len
    
    def __len__(self):
        return len(self.src_texts)
    
    def __getitem__(self, idx):
        src = self.src_texts[idx]
        tgt = self.tgt_texts[idx]
        
        # 토큰화 및 인덱스 변환
        src_ids = self.sp_src.encode(src, out_type=int)
        tgt_ids = self.sp_tgt.encode(tgt, out_type=int)
        
        # BOS와 EOS 토큰 추가 (타겟 시퀀스)
        tgt_ids = [self.sp_tgt.bos_id()] + tgt_ids + [self.sp_tgt.eos_id()]
        
        # 시퀀스 길이 제한
        src_ids = src_ids[:self.max_len]
        tgt_ids = tgt_ids[:self.max_len]
        
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)
