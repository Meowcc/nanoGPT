import torch
import torch.optim as optim
import torch.nn.functional as F
from model import Transformer  
from prepare import prepare_data  
import tiktoken

batch_size = 128
block_size = 256
max_iters = 2400
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 256

torch.manual_seed(42)
enc = tiktoken.get_encoding("gpt2")

data = prepare_data('../dataset/')
vocab_size = enc.n_vocab
data = torch.tensor(enc.encode(data), dtype=torch.long)
n = int(0.9 * len(data)) 
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

def estimate_loss(model):
    model.eval()
    losses = {'train': 0.0, 'val': 0.0}
    for split in ['train', 'val']:
        total_loss = 0.0
        for _ in range(eval_iters):
            x, y = get_batch(split)
            with torch.no_grad():
                logits = model(x, mask=None)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                total_loss += loss.item()
        losses[split] = total_loss / eval_iters
    model.train()
    return losses

model = Transformer(
    embed_size=n_embd, 
    num_layers=6, 
    heads=8, 
    vocab_size=vocab_size, 
    max_length=block_size, 
    forward_expansion=4, 
    dropout=0.1
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"Step {iter}: Train loss = {losses['train']:.4f}, Val loss = {losses['val']:.4f}")
    
    x, y = get_batch('train')

    logits = model(x, mask=None)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def generate_text(model, start_text, max_length, device):
    model.eval()
    input_ids = torch.tensor([enc.encode(start_text)], dtype=torch.long).to(device)

    for _ in range(max_length):
        with torch.no_grad():
            logits = model(input_ids, mask=None)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    return enc.decode(input_ids[0].tolist())

print("Training complete!")
start_text = "Harry Potter stood in the"
generated_text = generate_text(model, start_text, max_length=200, device=device)
print(f"Generated text: {generated_text}")
