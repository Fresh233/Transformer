import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

import yaml
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace

from model import Transformer, ModelArgs


# --- 数据集类 (无变动) ---
class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, max_seq_len):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_seq_len = max_seq_len

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = self.max_seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.max_seq_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            return self.__getitem__((idx + 1) % len(self))

        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
        ], dim=0)

        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
        ], dim=0)

        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
        ], dim=0)

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


# --- Tokenizer 训练 (已修改) ---
def get_or_build_tokenizer(config, ds, lang):
    # 修改(1): 从配置文件读取分词器保存路径
    tokenizer_dir = Path(config['tokenizer_path'])
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = tokenizer_dir / f"tokenizer_{lang}.json"

    if not tokenizer_path.exists():
        print(f"Building tokenizer for '{lang}' language and saving to {tokenizer_path}")
        tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordPieceTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], vocab_size=config['vocab_size'],
                                   min_frequency=2)

        def get_all_sentences(ds, lang):
            for item in ds:
                yield item['translation'][lang]

        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        print(f"Loading tokenizer for '{lang}' language from {tokenizer_path}")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


# --- 训练和验证 (无变动) ---
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training Epoch"):
        optimizer.zero_grad()

        encoder_input = batch['encoder_input'].to(device)
        decoder_input = batch['decoder_input'].to(device)
        label = batch['label'].to(device)

        encoder_mask = (encoder_input != 0).unsqueeze(1).unsqueeze(2)
        decoder_mask = (decoder_input != 0).unsqueeze(1).unsqueeze(2) & causal_mask(decoder_input.size(1)).to(device)

        output = model(encoder_input, decoder_input, encoder_mask, decoder_mask)

        loss = criterion(output.view(-1, output.size(-1)), label.view(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation Epoch"):
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            label = batch['label'].to(device)

            encoder_mask = (encoder_input != 0).unsqueeze(1).unsqueeze(2)
            decoder_mask = (decoder_input != 0).unsqueeze(1).unsqueeze(2) & causal_mask(decoder_input.size(1)).to(
                device)

            output = model(encoder_input, decoder_input, encoder_mask, decoder_mask)

            loss = criterion(output.view(-1, output.size(-1)), label.view(-1))
            total_loss += loss.item()
    return total_loss / len(loader)


# --- 主函数  ---
def main():
    parser = argparse.ArgumentParser(description="Train a Transformer model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    args = parser.parse_args()

    # 明确指定使用 utf-8 编码打开配置文件
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    torch.manual_seed(config['seed'])

    # --- 数据加载和预处理 ---
    # 修改(2): 在加载数据集时，指定 cache_dir
    print(f"Loading dataset... This will download to '{config['data_cache_dir']}'")
    raw_datasets = load_dataset(
        'iwslt2017',
        'iwslt2017-de-en',
        split='train',
        cache_dir=config['data_cache_dir'],
        trust_remote_code=True
    )

    split = raw_datasets.train_test_split(test_size=0.1, seed=config['seed'])
    train_ds_raw = split['train']
    val_ds_raw = split['test']

    tokenizer_src = get_or_build_tokenizer(config, train_ds_raw, config['src_lang'])
    tokenizer_tgt = get_or_build_tokenizer(config, train_ds_raw, config['tgt_lang'])

    train_dataset = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['src_lang'], config['tgt_lang'],
                                     config['max_seq_len'])
    val_dataset = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['src_lang'], config['tgt_lang'],
                                   config['max_seq_len'])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # --- 模型和优化器 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_args = ModelArgs(
        n_embd=config['n_embd'],
        n_heads=config['n_heads'],
        dim=config['dim'],
        dropout=config['dropout'],
        max_seq_len=config['max_seq_len'],
        vocab_size=tokenizer_src.get_vocab_size(),
        block_size=config['max_seq_len'],
        n_layer=config['n_layer']
    )
    model = Transformer(model_args).to(device)

    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'))

    # --- 训练循环 ---
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    # 修改(3): 确保 results 目录存在
    Path("results").mkdir(exist_ok=True)

    for epoch in range(config['epochs']):
        print(f"--- Epoch {epoch + 1}/{config['epochs']} ---")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)

        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Model saved.")

    # --- 绘制并保存损失曲线 ---
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('results/loss_curve.png')
    print("Loss curve saved to results/loss_curve.png")


if __name__ == "__main__":
    main()