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
# to calculate precision, recall, f1 score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
# reduce learning rate as training progresses
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

# load dataset from dataprocessing
mimic_data_dir = "/Users/amandali/Downloads/Mimic III"
# loads how many rows of mimic 3 data
result = load_mimic3_data(mimic_data_dir, nrows=1000000)

# define a mapping of event categories to their respective item keys, used for categorizing data sequences
CATEGORIES = {
    'chart_events': 'chart_items',
    'input_events': 'input_items',
    'lab_events': 'lab_items',
    'microbiology_events': 'microbiology_items',
    'prescriptions': 'prescriptions_items',
    'procedure_events': 'procedure_items'
}

# function to extract item sequences for a given category from the dataset, keyed by hospital admission IDs (hadm_id)
def extract_sequences(data, category_key):
    # initialize list to hold (hadm_id, items) tuples
    sequence_list = []
    # iterate over admission IDs and their event dictionaries
    for hadm_id, category_dict in data.items():
        # get list of items for the category, or empty if missing
        items = category_dict.get(category_key, [])
        if len(items) >= 2:  # Ensure sequence has at least 2 items
            sequence_list.append((hadm_id, items))
    return sequence_list

# Build vocabulary for a category
def build_vocab(sequences):
    item_counts = Counter()
    for seq in sequences:
        item_counts.update(seq)
    vocab = ['<PAD>', '<UNK>'] + [item for item, _ in item_counts.most_common()]
    item2idx = {item: i for i, item in enumerate(vocab)}
    idx2item = {i: item for item, i in item2idx.items()}
    return item2idx, idx2item

class SequenceDataset(Dataset):
    def __init__(self, sequences, item2idx, max_len=50, oversample_factor=3):  # Reduced oversample_factor
        self.pairs = []
        self.max_len = max_len
        self.item2idx = item2idx
        counts = Counter([item for seq in sequences for item in seq])
        rare_items = {item for item, count in counts.items() if count < 10}
        for seq in sequences:
            idx_seq = [item2idx.get(item, item2idx['<UNK>']) for item in seq]
            for i in range(1, len(idx_seq)):
                input_seq = idx_seq[:i][-max_len:]
                target = idx_seq[i]
                self.pairs.append((input_seq, target))
                if any(item in rare_items for item in seq):
                    for _ in range(oversample_factor - 1):
                        self.pairs.append((input_seq, target))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_seq, target = self.pairs[idx]
        if len(input_seq) < self.max_len:
            input_seq = [0] * (self.max_len - len(input_seq)) + input_seq
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)


def compute_class_weights(sequences, item2idx, device):
    counts = Counter([item for seq in sequences for item in seq])
    total = sum(counts.values())
    weights = {item2idx[item]: min(total / (len(counts) * count), 10.0) for item, count in counts.items()}
    weight_tensor = torch.ones(len(item2idx)).to(device)
    for idx, weight in weights.items():
        weight_tensor[idx] = weight
    return weight_tensor


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_heads=8, num_layers=3, dropout=0.6,
                 max_len=50):  # Increased dropout
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_len, embed_size))
        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(embed_size, vocab_size)
        self.embed_size = embed_size

    def forward(self, src):
        src_emb = self.embedding(src) + self.pos_encoder[:, :src.size(1), :]
        batch_size = src.size(0)
        dec_input = torch.zeros(batch_size, 1, dtype=torch.long).to(src.device)
        dec_emb = self.embedding(dec_input)
        output = self.transformer(src_emb, dec_emb)
        output = self.fc(output[:, -1, :])
        return output


def evaluate_model(model, dataloader, loss_fn, device, idx2item):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            total_loss += loss.item()
            preds = output.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

    if len(dataloader) == 0:
        print("Empty dataloader, skipping evaluation.")
        return 0.0, 0.0, 0.0, 0.0, 0.0

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

    valid_labels = sorted(set(all_targets).union(set(all_preds)))
    valid_names = [str(idx2item[i]) for i in valid_labels if i in idx2item]
    if valid_names:
        print(classification_report(all_targets, all_preds, labels=valid_labels, target_names=valid_names,
                                    zero_division=0, output_dict=False, digits=4))
    else:
        print("No valid labels for classification report.")

    return avg_loss, accuracy, precision, recall, f1


def train_model(model, train_loader, val_loader, epochs, lr=1e-4, class_weights=None, idx2item=None, checkpoint_path="checkpoint.pt"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)  # Increased patience
    metrics = []
    best_f1 = 0
    patience_counter = 0

    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_f1 = checkpoint['best_f1']
        metrics = checkpoint['metrics']
        print(f"Loaded checkpoint from {checkpoint_path}")

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_train_loss += loss.item()
            # Log progress every 100 batches
            if i % 100 == 0:
                print(f"Epoch {epoch + 1}, Batch {i}, Loss: {loss.item():.4f}")
        avg_train_loss = total_train_loss / len(train_loader)

        avg_val_loss, val_accuracy, val_precision, val_recall, val_f1 = evaluate_model(model, val_loader, loss_fn,
                                                                                       device, idx2item)

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'metrics': metrics
            }, checkpoint_path)
            print(f"Saved checkpoint at epoch {epoch + 1}")
        else:
            patience_counter += 1
            if patience_counter >= 3:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        scheduler.step(avg_val_loss)
        metrics.append({
            "Epoch": epoch + 1,
            "Train_Loss": avg_train_loss,
            "Val_Loss": avg_val_loss,
            "Val_Accuracy": val_accuracy,
            "Val_Precision": val_precision,
            "Val_Recall": val_recall,
            "Val_F1": val_f1
        })
        print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
              f"Val Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
    return metrics


def predict_next(model, input_seq, ground_truth, item2idx, idx2item, max_len=50):
    model.eval()
    device = next(model.parameters()).device
    input_ids = [item2idx.get(i, item2idx['<UNK>']) for i in input_seq]
    if len(input_ids) < max_len:
        input_ids = [0] * (max_len - len(input_ids)) + input_ids
    input_ids = input_ids[-max_len:]
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        pred_id = logits.argmax(dim=-1).item()
        predicted_item = idx2item[pred_id]
    return predicted_item, predicted_item == ground_truth


# Main execution
NUM_RUNS = 3
num_hadms = 10
all_prediction_rows = []
all_metrics = []

for category, category_key in CATEGORIES.items():
    print(f"\n=== Processing {category} ===")
    sequence_tuples = extract_sequences(result, category_key)
    if len(sequence_tuples) < 10:  # Skip categories with too few sequences
        print(f"Too few sequences ({len(sequence_tuples)}) for {category}. Skipping.")
        continue

    random.shuffle(sequence_tuples)
    train_size = int(0.8 * len(sequence_tuples))
    val_size = int(0.15 * len(sequence_tuples))
    train_sequences = [seq for _, seq in sequence_tuples[:train_size]]
    val_sequences = [seq for _, seq in sequence_tuples[train_size:train_size + val_size]]
    test_sequences = [seq for _, seq in sequence_tuples[train_size + val_size:]]
    train_hadm_ids = [hadm_id for hadm_id, seq in sequence_tuples[:train_size]]
    val_hadm_ids = [hadm_id for hadm_id, seq in sequence_tuples[train_size:train_size + val_size]]
    test_hadm_ids = [hadm_id for hadm_id, seq in sequence_tuples[train_size + val_size:]]

    # Log sequence statistics
    seq_lengths = [len(seq) for seq in train_sequences + val_sequences + test_sequences]
    print(
        f"Category {category}: {len(train_sequences)} train, {len(val_sequences)} val, {len(test_sequences)} test sequences")
    print(f"Avg seq length: {sum(seq_lengths) / len(seq_lengths):.2f}, Max: {max(seq_lengths)}")

    item2idx, idx2item = build_vocab(train_sequences + val_sequences + test_sequences)
    print(f"Vocabulary size: {len(item2idx)}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    class_weights = compute_class_weights(train_sequences, item2idx, device)

    max_len = min(max(seq_lengths), 50)
    train_dataset = SequenceDataset(train_sequences, item2idx, max_len=max_len, oversample_factor=3)
    val_dataset = SequenceDataset(val_sequences, item2idx, max_len=max_len,
                                  oversample_factor=1)  # No oversampling for val/test
    test_dataset = SequenceDataset(test_sequences, item2idx, max_len=max_len, oversample_factor=1)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Reduced batch size
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = TransformerModel(vocab_size=len(item2idx), embed_size=128, hidden_size=256, num_heads=8, num_layers=3,
                             dropout=0.6, max_len=max_len)
    checkpoint_path = f"checkpoint_{category}.pt"
    metrics = train_model(model, train_loader, val_loader, epochs=5, lr=1e-4, class_weights=class_weights,
                          idx2item=idx2item, checkpoint_path=checkpoint_path)
    for metric in metrics:
        metric['Category'] = category
        all_metrics.append(metric)

    test_loss, test_accuracy, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader,
                                                                                    nn.CrossEntropyLoss(
                                                                                        weight=class_weights.to(
                                                                                            device)), device, idx2item)
    all_metrics.append({
        "Category": category,
        "Epoch": "Test",
        "Train_Loss": None,
        "Val_Loss": test_loss,
        "Val_Accuracy": test_accuracy,
        "Val_Precision": test_precision,
        "Val_Recall": test_recall,
        "Val_F1": test_f1
    })
    print(f"\nTest Results for {category} - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, "
          f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")

    for dataset_name, hadm_ids in [("Train", train_hadm_ids), ("Validation", val_hadm_ids), ("Test", test_hadm_ids)]:
        valid_hadm_ids = [hadm_id for hadm_id in hadm_ids if len(result[hadm_id][category_key]) >= 2]
        print(f"{dataset_name} set: {len(valid_hadm_ids)} valid HADM_IDs with sequences >= 2")
        random.shuffle(valid_hadm_ids)
        hadm_ids_to_process = valid_hadm_ids[:min(num_hadms, len(valid_hadm_ids))]
        print(f"Processing {len(hadm_ids_to_process)} HADM_IDs for {dataset_name} set")
        for run in range(1, NUM_RUNS + 1):
            print(f"\n--Run {run} {dataset_name} Set Predictions for {category}--")
            for hadm_id in hadm_ids_to_process:
                seq = result[hadm_id][category_key]
                input_seq = seq[:-1]
                ground_truth = seq[-1]
                predicted_item, is_correct = predict_next(model, input_seq, ground_truth, item2idx, idx2item,
                                                          max_len=max_len)
                all_prediction_rows.append({
                    "Run": run,
                    "Category": category,
                    "HADM_ID": hadm_id,
                    "Input_Sequence": ", ".join(map(str, input_seq)),
                    "Ground_Truth": ground_truth,
                    "Predicted_Next_Item": predicted_item,
                    "Is_Correct": is_correct,
                    "Dataset": dataset_name
                })

df = pd.DataFrame(all_prediction_rows)
df.to_excel("mimic3_all_categories_predictions.xlsx", index=False)
print("\nPredictions saved to mimic3_all_categories_predictions.xlsx")

metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_excel("mimic3_all_categories_metrics.xlsx", index=False)
print("Metrics saved to mimic3_all_categories_metrics.xlsx")
