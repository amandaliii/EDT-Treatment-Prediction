# import dataset from dataprocessing.py
from DataProcessing import load_mimic3_data
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
# to save vocabulary
import pickle

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

# evaluate model on test set
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

# main execution for testing
NUM_RUNS = 3
num_hadms = 10
all_prediction_rows = []
all_metrics = []

for category, category_key in CATEGORIES.items():
    print(f"\n=== Processing {category} for Testing ===")

    # extract sequences
    sequence_tuples = extract_sequences(result, category_key)
    if not sequence_tuples:
        print(f"No valid sequences for {category}. Skipping.")
        continue

    # split into train, validation, and test
    random.shuffle(sequence_tuples)
    train_size = int(0.8 * len(sequence_tuples))
    val_size = int(0.15 * len(sequence_tuples))
    test_sequences = [seq for _, seq in sequence_tuples[train_size + val_size:]]
    test_hadm_ids = [hadm_id for hadm_id, seq in sequence_tuples[train_size + val_size:]]

    # load saved vocabulary
    try:
        with open(f"vocab_{category}.pkl", "rb") as f:
            vocab_data = pickle.load(f)
            item2idx = vocab_data["item2idx"]
            idx2item = vocab_data["idx2item"]
        print(f"Loaded vocabulary for {category} from vocab_{category}.pkl")
    except FileNotFoundError:
        print(f"Vocabulary file vocab_{category}.pkl not found. Skipping {category}.")
        continue

    # create test dataset and dataloader
    test_dataset = SequenceDataset(test_sequences, item2idx)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # initialize model and load saved weights
    model = decoderModel(vocab_size=len(item2idx), embed_size=64, hidden_size=128)
    try:
        model.load_state_dict(torch.load(f"model_{category}.pth", map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
        print(f"Loaded model for {category} from model_{category}.pth")
    except FileNotFoundError:
        print(f"Model file model_{category}.pth not found. Skipping {category}.")
        continue

    # evaluate on test set
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
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

    # predict for a subset of test HADM_IDs
    random.shuffle(test_hadm_ids)
    hadm_ids_to_process = test_hadm_ids[:min(num_hadms, len(test_hadm_ids))]
    for run in range(1, NUM_RUNS + 1):
        print(f"\nRun {run} Test Set Predictions for {category}...")
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
                "Dataset": "Test"
            })

# save results to excel files
df = pd.DataFrame(all_prediction_rows)
df.to_excel("test_predictions.xlsx", index=False)
print("\nTest Predictions saved to test_predictions.xlsx")

metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_excel("test_metrics.xlsx", index=False)
print("Test Metrics saved to test_metrics.xlsx")
