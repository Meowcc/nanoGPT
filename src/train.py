import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from model import Transformer  
from prepare import prepare_data  
import tiktoken

batch_size = 128
block_size = 128
max_iters = 2400
eval_interval = 200
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 20
n_embd = 256
accumulation_steps = 4

torch.manual_seed(42)
enc = tiktoken.get_encoding("gpt2")

data = prepare_data('../dataset/')
vocab_size = enc.n_vocab
data = torch.tensor(enc.encode(data), dtype=torch.long)
n = int(0.9 * len(data)) 
train_data = data[:n]
val_data = data[n:]

train_dataset = TensorDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(val_data)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

def get_batch(split):
    if split == 'train':
        return next(iter(train_loader))[0].to(device)
    else :
        return next(iter(val_loader))[0].to(device)

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
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"Step {iter}: Train loss = {losses['train']:.4f}, Val loss = {losses['val']:.4f}")
    
    x, y = get_batch('train')
    logits = model(x, mask=None)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    loss = loss / accumulation_steps
    loss.backward()
    
    if (iter + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
    
    scheduler.step()

def generate_text(model, start_text=None, max_length=100, temperature = 1.0, device = device):
    model.eval()
    if start_text is None:
        input_ids = torch.tensor([enc.encode("<endoftext>"[0])], dtype=torch.long).to(device)
    else:
        input_ids = torch.tensor([enc.encode(start_text)], dtype=torch.long).to(device)

    for _ in range(max_length):
        with torch.no_grad():
            logits = model(input_ids, mask=None)
            next_token_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

    return enc.decode(input_ids[0].tolist())

def generate_multiple_texts(model, max_length, temperature = 1.0, device = device):
    samples = []
    start_texts = [
        "Harry looked around and saw",
        "The dark forest was",
        "In the darkness, a figure appeared",
        "Hermione grabbed her wand and",
        "Ron couldn't believe his eyes when he saw",
        "The room was filled with",
        "A loud noise echoed through the castle",
        "Dumbledore turned to Harry and said",
        "The spell backfired and",
        None,
    ]
    for start_text in start_texts:
        samples.append(generate_text(model, start_text, max_length, temperature, device))
    return samples

# generate text
for i, text in enumerate(generate_multiple_texts(model, 100, 0.8, device)):
    print(f"Text {i + 1}: {text}")
