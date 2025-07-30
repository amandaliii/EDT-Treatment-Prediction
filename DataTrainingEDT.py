# import dataset from dataprocessing.py
from DataProcessing import load_mimic3_data
# used to count frequency of items (for building vocabulary)
from collections import Counter
# pytorch tools for tensors, dataset handling, and neural network layers
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.utils.data import random_split
# used for shuffling
import random
# stores and exports predictions into Excel file
import pandas as pd
# to calculate precision, recall, and f1 score
from sklearn.metrics import precision_score, recall_score, f1_score
# for validation loss/metrics
from sklearn.model_selection import train_test_split

# load dataset from dataprocessing
mimic_data_dir = "/Users/amandali/Downloads/Mimic III"
# loads how many rows of mimic 3 data
result = load_mimic3_data(mimic_data_dir, nrows=1000000)

# flatten all sequences across all HADM_IDs into one list of category sequences
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
    # add special tokens at start
    # pad: padding token for sequence length normalization
    # unk: unknown token for out of vocab items
    # bos: beginning of sequence token
    # eos: end of sentence token
    special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
    vocab = special_tokens + [item for item, _ in item_counts.most_common()]
    # builds two dictionaries, 1 maps tokens to unique integer indices, the other reverse mapping from indices to tokens
    item2idx = {item: i for i, item in enumerate(vocab)}
    idx2item = {i: item for item, i in item2idx.items()}
    # returns the mapping of item to index and index to item
    return item2idx, idx2item

# combines individual dataset items into batches
# pads all sequences in the batch to same legnth using pad_idx
def collate_fn(batch, pad_idx):
    # source sequences are padded on the left (more space on left), and target sequences are padded on the right (more space on right)
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
        self.bos_idx = item2idx['<BOS>']
        self.eos_idx = item2idx['<EOS>']

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

# determines validation loss
def evaluate_model(model, dataloader, loss_fn, device):
    model.eval()
    total_val_loss = 0
    count = 0
    with torch.no_grad():
        for src, tgt_in, tgt_out in dataloader:
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            output = model(src, tgt_in)
            loss = loss_fn(output.view(-1, output.size(-1)), tgt_out.view(-1))
            total_val_loss += loss.item() * src.size(0)
            count += src.size(0)
    return total_val_loss / count

# train the model for epochs times
def train_model(model, dataloader, val_loader, epochs, lr=1e-3):
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
            total_loss += loss.item() * src.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)
        avg_val_loss = evaluate_model(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        epoch_losses.append({
            "Epoch": epoch + 1,
            "Train Loss": avg_train_loss,
            "Val Loss": avg_val_loss
        })
    return epoch_losses

# predict the next term
def predict_next(model, input_seq, item2idx, idx2item, max_len=None):
    model.eval()
    device = next(model.parameters()).device
    input_ids = [item2idx.get(i, item2idx['<UNK>']) for i in input_seq]

    # takes an input sequence, pads or trims it to length max_len, and feeds it to the model
    if max_len is None:
        max_len = len(input_ids)

    print(f"Example vocab tokens: {[idx2item[i] for i in range(min(10, len(idx2item)))]}")
    print(f"Predicted IDs range: {min(idx2item.keys())}-{max(idx2item.keys())}")

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
        if pred_id not in idx2item:
            # catch out of vocab keys
            print(f"Warning: pred_id {pred_id} not found in idx2item, returning <UNK>")
            return '<UNK>'
        token = idx2item.get(pred_id, '<UNK>')
        return str(token)

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

# creates training and validation datasets
val_fraction = 0.2  # 20% validation
total_size = len(dataset)
val_size = int(total_size * val_fraction)
train_size = total_size - val_size
# splits into training and validation dataset
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
# training dataset
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,  # shuffle training data
    collate_fn=lambda b: collate_fn(b, pad_idx)
)
# validation dataset
val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,  # typically no shuffle on validation data
    collate_fn=lambda b: collate_fn(b, pad_idx)
)

# store predictions
prediction_rows = []
# store epoch tracking
epoch_tracking = []
# number of hadm_id patients to show
num_hadms = 5
# number of prediction runs
NUM_RUNS = 2

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
    epoch_losses = train_model(model, train_loader, val_loader, epochs=15)
    for epoch_loss in epoch_losses:
        epoch_loss.update({"Run": run})
        epoch_tracking.append(epoch_loss)

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
            input_seq = items[:-1] # all except last, for prediction
            true_next = items[-1] # the actual next item
            predicted_next = predict_next(model, input_seq, item2idx, idx2item, max_len=128)
            print(f"Predicted next item token: {predicted_next} (type: {type(predicted_next)})")
            prediction_rows.append({
                "Run": run,
                "HADM_ID": hadm_id,
                "Category": category,
                "Input Sequence": ", ".join(map(str, input_seq)),
                "True Next Item": true_next,
                "Predicted Next Item": predicted_next
            })

# After prediction_rows is populated
y_true = [row["True Next Item"] for row in prediction_rows]
# list of labels/items predicted
y_pred = [row["Predicted Next Item"] for row in prediction_rows]

# convert to numeric indices using item2idx to handle categorical items
y_true_idx = [item2idx.get(i, item2idx['<UNK>']) for i in y_true]
y_pred_idx = [item2idx.get(i, item2idx['<UNK>']) for i in y_pred]
print("Unique labels in true:", set(y_true_idx))
print("Unique labels in pred:", set(y_pred_idx))

# average macro done so metrics are calculated for each class individually, then average is taken across all classes weighted equally
# zero divisions = 0 so that when the metric in undefined, it assigns 0 instead of a warning
precision = precision_score(y_true_idx, y_pred_idx, average="macro", zero_division=0)
recall = recall_score(y_true_idx, y_pred_idx, average="macro", zero_division=0)
f1 = f1_score(y_true_idx, y_pred_idx, average="macro", zero_division=0)
print(f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

# split sequence beforehand
train_seqs, val_seqs = train_test_split(all_sequences, test_size=0.2, random_state=42)

train_dataset = EncoderDecoderDataset(train_seqs, item2idx)
val_dataset = EncoderDecoderDataset(val_seqs, item2idx)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_idx))
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=lambda b: collate_fn(b, pad_idx))

# how many HADM_IDs are being used
unique_ids = set([row["HADM_ID"] for row in prediction_rows])
print(f"\nTotal unique HADM_IDs predicted across all runs: {len(unique_ids)}")
# save epoch loss history
epoch_df = pd.DataFrame(epoch_tracking)
epoch_df.to_excel("EDT_mimic3_epoch_losses.xlsx", index=False)
print("Epoch data saved to EDT_mimic3_epoch_losses.xlsx")
# save predictions
df = pd.DataFrame(prediction_rows)
df.to_excel("EDT_mimic3_predictions.xlsx", index=False)
print("Predictions saved to EDT_mimic3_predictions.xlsx")
