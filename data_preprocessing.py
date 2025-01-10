from datasets import load_dataset
import sentencepiece as spm

# IWSLT 2017 영어-독일어 데이터셋 로드
dataset = load_dataset('iwslt2017', 'iwslt2017-en-de')

# 데이터셋의 스플릿 확인
# print(dataset)

def save_texts(dataset_split, src_lang, tgt_lang, src_file, tgt_file, num_examples=None):
    """
    데이터셋 스플릿에서 소스와 타겟 문장을 추출하여 텍스트 파일로 저장.
    
    Args:
        dataset_split: datasets 라이브러리의 데이터 분할 (train, validation, test)
        src_lang: 소스 언어 코드 (예: 'de')
        tgt_lang: 타겟 언어 코드 (예: 'en')
        src_file: 소스 문장을 저장할 파일 경로
        tgt_file: 타겟 문장을 저장할 파일 경로
        num_examples: 저장할 문장 수 (None이면 전체)
    """
    with open(src_file, 'w', encoding='utf-8') as f_src, open(tgt_file, 'w', encoding='utf-8') as f_tgt:
        for i, example in enumerate(dataset_split):
            if num_examples and i >= num_examples:
                break
            src_text = example['translation'][src_lang]
            tgt_text = example['translation'][tgt_lang]
            f_src.write(src_text + '\n')
            f_tgt.write(tgt_text + '\n')

# 훈련 데이터 문장 저장
save_texts(dataset['train'], 'de', 'en', 'de_train.txt', 'en_train.txt', num_examples=10000)

def train_sentencepiece(input_file, model_prefix, vocab_size=32000):
    """
    전처리 모델 SentencePiece 학습

    Args:
        input_file: 학습에 사용되는 파일 경로
        model_prefix: 모델 파일 접두사
        vocab_size: vocab 단어 개수
    """
    spm.SentencePieceTrainer.Train(
        input = input_file,
        model_prefix = model_prefix,
        vocab_size = vocab_size,
        character_coverage = 1.0,
        model_type = 'bpe',
        pad_id = 0,
        unk_id = 1,
        bos_id = 2,
        eos_id = 3
    )

train_sentencepiece('de_train.txt', 'spm_de')
train_sentencepiece('en_train.txt', 'spm_en')