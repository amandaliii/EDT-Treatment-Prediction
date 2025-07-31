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
# to calculate precision, recall, and f1 score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# load dataset from dataprocessing
mimic_data_dir = "/Users/amandali/Downloads/Mimic III"
# loads how many rows of mimic 3 data
result = load_mimic3_data(mimic_data_dir, nrows=1000000)

# category configuration
CATEGORIES = {
    'chart_events': 'chart_items',
    'input_events': 'input_items',
    'lab_events': 'lab_items',
    'microbiology_events': 'microbiology_items',
    'prescriptions': 'prescriptions_items',
    'procedure_events': 'procedure_items'
}

# extract sequences for a specific category with HADM_IDs
def extract_sequences(data, category_key):
    sequence_list = []
    for hadm_id, category_dict in data.items():
        items = category_dict.get(category_key, [])
        if len(items) >= 2:  # Ensure sequence has at least 2 items
            sequence_list.append((hadm_id, items))
    return sequence_list

# build vocabulary for a category
def build_vocab(sequences):
    item_counts = Counter()
    for seq in sequences:
        item_counts.update(seq)
    vocab = ['<PAD>', '<UNK>'] + [item for item, _ in item_counts.most_common()]
    item2idx = {item: i for i, item in enumerate(vocab)}
    idx2item = {i: item for item, i in item2idx.items()}
    return item2idx, idx2item

# dataset for sequences
class SequenceDataset(Dataset):
    def __init__(self, sequences, item2idx, max_len=20):
        self.pairs = []
        self.max_len = max_len
        self.item2idx = item2idx
        for seq in sequences:
            idx_seq = [item2idx.get(item, item2idx['<UNK>']) for item in seq]
            for i in range(1, len(idx_seq)):
                input_seq = idx_seq[:i][-max_len:]  # Sliding window
                target = idx_seq[i]
                self.pairs.append((input_seq, target))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_seq, target = self.pairs[idx]
        if len(input_seq) < self.max_len:
            input_seq = [0] * (self.max_len - len(input_seq)) + input_seq
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)

# decoder Model
class decoderModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(decoderModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        embeds = self.embedding(x)
        _, (h_n, _) = self.lstm(embeds)
        out = self.fc(h_n[-1])
        return out

# evaluate model on validation or test set
def evaluate_model(model, dataloader, loss_fn, device):
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
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    return avg_loss, accuracy, precision, recall, f1

# train the model with train/validation split
def train_model(model, train_loader, val_loader, epochs, lr=1e-3):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    metrics = []

    for epoch in range(epochs):
        # training
        model.train()
        total_train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        # validation
        avg_val_loss, val_accuracy, val_precision, val_recall, val_f1 = evaluate_model(model, val_loader, loss_fn, device)
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

# predict next item and compare with ground truth
def predict_next(model, input_seq, ground_truth, item2idx, idx2item, max_len=20):
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

# main execution
NUM_RUNS = 3
num_hadms = 10
all_prediction_rows = []
all_metrics = []

for category, category_key in CATEGORIES.items():
    print(f"\n=== Processing {category} ===")

    # extract sequences
    sequence_tuples = extract_sequences(result, category_key)
    if not sequence_tuples:
        print(f"No valid sequences for {category}. Skipping.")
        continue

    # split into train, validation, and test
    random.shuffle(sequence_tuples)
    train_size = int(0.8 * len(sequence_tuples))
    val_size = int(0.15 * len(sequence_tuples))
    train_sequences = [seq for _, seq in sequence_tuples[:train_size]]
    val_sequences = [seq for _, seq in sequence_tuples[train_size:train_size + val_size]]
    test_sequences = [seq for _, seq in sequence_tuples[train_size + val_size:]]
    train_hadm_ids = [hadm_id for hadm_id, seq in sequence_tuples[:train_size]]
    val_hadm_ids = [hadm_id for hadm_id, seq in sequence_tuples[train_size:train_size + val_size]]
    test_hadm_ids = [hadm_id for hadm_id, seq in sequence_tuples[train_size + val_size:]]

    # build vocabulary
    item2idx, idx2item = build_vocab(train_sequences + val_sequences + test_sequences)

    # create datasets and dataloaders
    train_dataset = SequenceDataset(train_sequences, item2idx)
    val_dataset = SequenceDataset(val_sequences, item2idx)
    test_dataset = SequenceDataset(test_sequences, item2idx)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # initialize and train model
    model = decoderModel(vocab_size=len(item2idx), embed_size=64, hidden_size=128)
    metrics = train_model(model, train_loader, val_loader, epochs=10)
    for metric in metrics:
        metric['Category'] = category
        all_metrics.append(metric)

    # evaluate on test set
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_loss, test_accuracy, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader, nn.CrossEntropyLoss(), device)
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

    # predict for a subset of HADM_IDs from each dataset split
    for dataset_name, hadm_ids in [("Train", train_hadm_ids), ("Validation", val_hadm_ids), ("Test", test_hadm_ids)]:
        random.shuffle(hadm_ids)
        hadm_ids_to_process = hadm_ids[:min(num_hadms, len(hadm_ids))]
        for run in range(1, NUM_RUNS + 1):
            print(f"\nRun {run} {dataset_name} Set Predictions for {category}...")
            for hadm_id in hadm_ids_to_process:
                seq = result[hadm_id][category_key]
                if len(seq) < 2:
                    continue
                # use all but the last item as input, last item as ground truth
                input_seq = seq[:-1]
                ground_truth = seq[-1]
                predicted_item, is_correct = predict_next(model, input_seq, ground_truth, item2idx, idx2item)
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

# save results to excel files
df = pd.DataFrame(all_prediction_rows)
df.to_excel("mimic3_all_categories_predictions.xlsx", index=False)
print("\nPredictions saved to mimic3_all_categories_predictions.xlsx")

metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_excel("mimic3_all_categories_metrics.xlsx", index=False)
print("Metrics saved to mimic3_all_categories_metrics.xlsx")
