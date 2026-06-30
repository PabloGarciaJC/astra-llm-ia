import os
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# ==============================
# HIPERPARÁMETROS
# ==============================
EPOCHS      = int(os.getenv('EPOCHS', 50))
BATCH_SIZE  = int(os.getenv('BATCH_SIZE', 32))
LR          = float(os.getenv('LEARNING_RATE', 3e-4))
BLOCK_SIZE  = int(os.getenv('BLOCK_SIZE', 64))
N_EMBED     = int(os.getenv('N_EMBED', 128))
N_HEAD      = int(os.getenv('N_HEAD', 4))
N_LAYER     = int(os.getenv('N_LAYER', 4))
DROPOUT     = float(os.getenv('DROPOUT', 0.1))
EVAL_EVERY  = 10
GEN_LEN     = 200
CORPUS_PATH = 'data/corpus.txt'
MODEL_PATH  = 'output/model.pt'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ==============================
# TOKENIZER (nivel carácter)
# ==============================
class Tokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, tokens):
        return ''.join(self.itos[t] for t in tokens)


# ==============================
# DATASET
# ==============================
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.block_size]
        y = self.data[idx + 1: idx + self.block_size + 1]
        return x, y


# ==============================
# ARQUITECTURA — TRANSFORMER
# ==============================
class Head(nn.Module):
    """Un cabezal de self-attention."""

    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        scale = C ** -0.5
        att = (q @ k.transpose(-2, -1)) * scale
        att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        v = self.value(x)
        return att @ v


class MultiHeadAttention(nn.Module):
    """Varios cabezales en paralelo."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads   = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj    = nn.Linear(N_EMBED, N_EMBED)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Red neuronal después de la atención."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBED, 4 * N_EMBED),
            nn.ReLU(),
            nn.Linear(4 * N_EMBED, N_EMBED),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Un bloque transformer: atención + feed-forward."""

    def __init__(self):
        super().__init__()
        head_size = N_EMBED // N_HEAD
        self.att = MultiHeadAttention(N_HEAD, head_size)
        self.ff  = FeedForward()
        self.ln1 = nn.LayerNorm(N_EMBED)
        self.ln2 = nn.LayerNorm(N_EMBED)

    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class LLM(nn.Module):
    """El modelo completo."""

    def __init__(self, vocab_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, N_EMBED)
        self.pos_emb   = nn.Embedding(BLOCK_SIZE, N_EMBED)
        self.blocks    = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln        = nn.LayerNorm(N_EMBED)
        self.head      = nn.Linear(N_EMBED, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=DEVICE))
        x   = tok + pos
        x   = self.blocks(x)
        x   = self.ln(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs  = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx


# ==============================
# ENTRENAMIENTO
# ==============================
def main():
    print("=" * 50)
    print("  Astra LLM AI Trainer")
    print("=" * 50)
    print(f"  Device  : {DEVICE}")
    print(f"  Epochs  : {EPOCHS}")
    print(f"  Embed   : {N_EMBED}  |  Heads: {N_HEAD}  |  Layers: {N_LAYER}")
    print(f"  Block   : {BLOCK_SIZE}  |  Batch: {BATCH_SIZE}")
    print("=" * 50)

    if not os.path.exists(CORPUS_PATH):
        print(f"[ERROR] No se encontró el corpus en: {CORPUS_PATH}")
        print("  Añade texto en llm-trainer/data/corpus.txt y vuelve a entrenar.")
        return

    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"\n[INFO] Corpus cargado: {len(text):,} caracteres")

    tokenizer = Tokenizer(text)
    print(f"[INFO] Vocabulario: {tokenizer.vocab_size} caracteres únicos")

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data   = data[split:]

    train_dataset = TextDataset(train_data, BLOCK_SIZE)
    train_loader  = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )

    model = LLM(tokenizer.vocab_size).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Parámetros del modelo: {params:,}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        if epoch % EVAL_EVERY == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_losses = []
                val_dataset = TextDataset(val_data, BLOCK_SIZE)
                val_loader  = torch.utils.data.DataLoader(
                    val_dataset, batch_size=BATCH_SIZE, drop_last=True
                )
                for x, y in val_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    _, loss = model(x, y)
                    val_losses.append(loss.item())
                val_loss = sum(val_losses) / len(val_losses) if val_losses else float('nan')

            print(f"  Época {epoch:>4}/{EPOCHS}  |  train loss: {avg_loss:.4f}  |  val loss: {val_loss:.4f}")

            context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
            generated = model.generate(context, max_new_tokens=GEN_LEN)
            sample = tokenizer.decode(generated[0].tolist())
            print(f"\n  [MUESTRA]\n  {sample[:200]}\n")
        else:
            print(f"  Época {epoch:>4}/{EPOCHS}  |  train loss: {avg_loss:.4f}")

    os.makedirs('output', exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'vocab_size':  tokenizer.vocab_size,
        'stoi':        tokenizer.stoi,
        'itos':        tokenizer.itos,
        'config': {
            'n_embed':    N_EMBED,
            'n_head':     N_HEAD,
            'n_layer':    N_LAYER,
            'block_size': BLOCK_SIZE,
            'dropout':    DROPOUT,
        }
    }, MODEL_PATH)

    print(f"\n[OK] Modelo guardado en {MODEL_PATH}")
    print("=" * 50)


if __name__ == '__main__':
    main()
