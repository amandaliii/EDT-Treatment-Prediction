# import dataset from dataprocessing.py
from DataProcessing import load_mimic3_data
# used to count frequency of items (for building vocabulary)
from collections import Counter
# pytorch tools for tensors, dataset handling, and neural network layers
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
# used for shuffling
import random
# stores and exports predictions into Excel file
import pandas as pd

# load dataset from dataprocessing
mimic_data_dir = "/Users/amandali/Downloads/Mimic III"
# loads how many rows of mimic 3 data
result = load_mimic3_data(mimic_data_dir, nrows=100000)

# Flatten all sequences across all HADM_IDs into one list of category sequences
def extract_sequences_with_hadm_ids(data):
    # returns a flat list of (hadm_id, category, sequence) tuples
    sequence_list = []
    for hadm_id, category_dict in data.items():
        for category, items in category_dict.items():
            # makes sure that each category has at least two items
            if len(items) >= 2:
                sequence_list.append((hadm_id, category, items))
    # returns the flat list of tuples
    return sequence_list

# map categorical codes to tokens
def build_vocab(sequences):
    # counts occurrences of all items in a sequence
    item_counts = Counter()
    for seq in sequences:
        item_counts.update(seq)
    vocab = ['<PAD>', '<UNK>'] + [item for item, _ in item_counts.most_common()]
    item2idx = {item: i for i, item in enumerate(vocab)}
    idx2item = {i: item for item, i in item2idx.items()}
    # returns the mapping of item to index and index to item
    return item2idx, idx2item

def collate_fn(batch, pad_idx):
    srcs, tgt_ins, tgt_outs = zip(*batch)
    max_src = max(len(s) for s in srcs)
    max_tgt = max(len(t) for t in tgt_ins)

    src_batch = torch.full((len(batch), max_src), pad_idx, dtype=torch.long)
    tgt_in_batch = torch.full((len(batch), max_tgt), pad_idx, dtype=torch.long)
    tgt_out_batch = torch.full((len(batch), max_tgt), pad_idx, dtype=torch.long)

    for i in range(len(batch)):
        src_batch[i, -len(srcs[i]):] = torch.tensor(srcs[i])
        tgt_in_batch[i, :len(tgt_ins[i])] = torch.tensor(tgt_ins[i])
        tgt_out_batch[i, :len(tgt_outs[i])] = torch.tensor(tgt_outs[i])

    return src_batch, tgt_in_batch, tgt_out_batch

# dataset generator
class EncoderDecoderDataset(Dataset):
    def __init__(self, sequences, item2idx, max_len=64):
        self.data = []
        self.item2idx = item2idx
        self.pad_idx = item2idx['<PAD>']
        self.unk_idx = item2idx['<UNK>']
        self.bos_idx = item2idx.get('<BOS>', len(item2idx))
        self.eos_idx = item2idx.get('<EOS>', len(item2idx) + 1)

        if '<BOS>' not in item2idx:
            self.item2idx['<BOS>'] = self.bos_idx
        if '<EOS>' not in item2idx:
            self.item2idx['<EOS>'] = self.eos_idx

        for seq in sequences:
            # Convert to indices
            idx_seq = [item2idx.get(code, self.unk_idx) for code in seq][:max_len - 2]
            if len(idx_seq) < 2:
                continue
            src = idx_seq[:-1]
            tgt = idx_seq[1:]
            tgt_in = [self.bos_idx] + tgt
            tgt_out = tgt + [self.eos_idx]
            self.data.append((src, tgt_in, tgt_out))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# encoder decoder transformer
class EncoderDecoderTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, nhead=8, num_layers=2, ff_dim=256, max_len=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=ff_dim,
            dropout=0.1,
            batch_first=True,
        )
        self.output = nn.Linear(embed_dim, vocab_size)

    def forward(self, src, tgt_in):
        B, S = src.size()
        B, T = tgt_in.size()
        device = src.device

        pos_src = self.pos_embedding(torch.arange(S, device=device)).unsqueeze(0)
        pos_tgt = self.pos_embedding(torch.arange(T, device=device)).unsqueeze(0)

        src_emb = self.embedding(src) + pos_src
        tgt_emb = self.embedding(tgt_in) + pos_tgt

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(device)
        out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        return self.output(out)

# train the model for epochs times
def train_model(model, dataloader, epochs, lr=1e-3):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=model.embedding.padding_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epoch_losses = []

    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for src, tgt_in, tgt_out in dataloader:
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            optimizer.zero_grad()
            output = model(src, tgt_in)
            # output: (batch_size, seq_len, vocab_size)
            # loss expects (batch_size*seq_len, vocab_size) and targets (batch_size*seq_len)
            loss = loss_fn(output.view(-1, output.size(-1)), tgt_out.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")
        epoch_losses.append(total_loss)
    return epoch_losses

# predict the next term
def predict_next(model, input_seq, item2idx, idx2item, max_len=None):
    model.eval()
    device = next(model.parameters()).device
    input_ids = [item2idx.get(i, item2idx['<UNK>']) for i in input_seq]

    # takes an input sequence, pads or trims it to length max_len, and feeds it to the model
    if max_len is None:
        max_len = len(input_ids)

    input_ids = input_ids[-max_len:]
    if len(input_ids) < max_len:
        input_ids = [item2idx['<PAD>']] * (max_len - len(input_ids)) + input_ids

    src = torch.tensor([input_ids]).to(device)  # (1, seq_len)

    # For a one-token prediction, start tgt_in as just <BOS>
    bos_idx = item2idx.get('<BOS>', len(item2idx))
    tgt_in = torch.tensor([[bos_idx]]).to(device)  # (1, 1)

    with torch.no_grad():
        logits = model(src, tgt_in)  # (1, 1, vocab_size)
        pred_id = logits[0, -1].argmax(dim=-1).item()
        return idx2item[pred_id]

# still use a single vocabulary for all categories
# tuple of sequences, unmodifiable
sequence_tuples = extract_sequences_with_hadm_ids(result)

# --- NEW: Calculate max sequence length for model positional encoding ---
all_train_lens = [len(seq) for seq in sequence_tuples]
all_infer_lens = [len(items) for _, _, items in sequence_tuples]
cfg_max_len = max(all_train_lens + all_infer_lens)
print("Maximum sequence length needed for positional embedding:", cfg_max_len)

# combines all sequences for each category
all_sequences = [seq for _, _, seq in sequence_tuples]
item2idx, idx2item = build_vocab(all_sequences)
# creates dataset of sequences, mapping the items to the hadm_id
dataset = EncoderDecoderDataset(all_sequences, item2idx)
# get the index for <PAD> so we can use it in the collate function
pad_idx = item2idx['<PAD>']
# loads in data
loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    collate_fn=lambda b: collate_fn(b, pad_idx)
)

# store predictions
prediction_rows = []
# store epoch tracking
epoch_tracking = []
# number of hadm_id patients to show
num_hadms = 5
# number of prediction runs
NUM_RUNS = 3

# run the prediction
print("\n=== HADM_ID + Category-wise Predictions ===\n")
for run in range(1, NUM_RUNS + 1):
    print(f"\nRun {run} Predictions...\n")
    # run transformer
    model = EncoderDecoderTransformer(
        vocab_size=len(item2idx),
        embed_dim=64,
        max_len=cfg_max_len + 2  # +2 for BOS/EOS or any extra special tokens
    )
    epoch_losses = train_model(model, loader, epochs=7)
    for epoch_num, loss in enumerate(epoch_losses, 1):
        epoch_tracking.append({
            "Run": run,
            "Epoch": epoch_num,
            "Loss": loss
        })

    # Get all unique HADM_IDs, shuffle, and pick N unique ones
    # filter HADM_IDs to only those with at least one category with ≥ 2 items
    valid_hadm_ids = [
        hadm_id for hadm_id, category_dict in result.items()
        if any(len(items) >= 2 for items in category_dict.values())
    ]

    random.shuffle(valid_hadm_ids)
    hadm_ids_to_process = valid_hadm_ids[:num_hadms]  # use only top N eligible HADM_IDs

    print(f"[Run {run}] Using {len(hadm_ids_to_process)} valid HADM_IDs.")

    for hadm_id in hadm_ids_to_process:
        categories_dict = result[hadm_id]
        print(f"HADM_ID: {hadm_id}")
        for category, items in categories_dict.items():
            print(f"    Category: {category} → Predicting next item from sequence of length {len(items)}...")
            if len(items) < 2:
                continue
            input_seq = items[:]
            predicted_next = predict_next(model, items, item2idx, idx2item, max_len=cfg_max_len)
            prediction_rows.append({
                "Run": run,
                "HADM_ID": hadm_id,
                "Category": category,
                "Input_Sequence": ", ".join(map(str, input_seq)),
                "Predicted_Next_Item": predicted_next
            })

# how many HADM_IDs are being used
unique_ids = set([row["HADM_ID"] for row in prediction_rows])
print(f"\nTotal unique HADM_IDs predicted across all runs: {len(unique_ids)}")
# Save epoch loss history
epoch_df = pd.DataFrame(epoch_tracking)
epoch_df.to_excel("EDT_mimic3_epoch_losses.xlsx", index=False)
print("Epoch losses saved to EDT_mimic3_epoch_losses.xlsx")
# save predictions into an excel sheet
df = pd.DataFrame(prediction_rows)
df.to_excel("EDT_mimic3_predictions.xlsx", index=False)
print("Predictions saved to EDT_mimic3_predictions.xlsx")