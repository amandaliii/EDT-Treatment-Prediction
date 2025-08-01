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
result = load_mimic3_data(mimic_data_dir, nrows=2000000)

# category configuration dictionary mapping event types to their keys in the data
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
    sequence_list = []  # Initialize list to store tuples of (hadm_id, item sequence)
    for hadm_id, category_dict in data.items():
        # Iterate over each admission and its categorized event data
        items = category_dict.get(category_key, [])
        # Retrieve the list of items for the specific category, default empty list if not found
        if len(items) >= 2:  # Ensure there are at least 2 items in sequence for prediction
            sequence_list.append((hadm_id, items))
            # Append tuple (HADM_ID, sequence of items) to list
    return sequence_list  # Return all valid sequences for this category

# dataset for sequences
class SequenceDataset(Dataset):
    # Custom PyTorch dataset to handle input-target pairs generated from event sequences
    def __init__(self, sequences, item2idx, max_len=20):
        self.pairs = []  # List to hold (input_sequence, target_item) pairs
        self.max_len = max_len  # Max sequence length for model input
        self.item2idx = item2idx  # Dictionary mapping items to indices
        for seq in sequences:
            # Convert each item in sequence to corresponding index; use <UNK> if not found
            idx_seq = [item2idx.get(item, item2idx['<UNK>']) for item in seq]
            # For each position except first, create input-target pair
            for i in range(1, len(idx_seq)):
                input_seq = idx_seq[:i][-max_len:]  # Take last max_len tokens before i as input (sliding window)
                target = idx_seq[i]  # Target is item at position i
                self.pairs.append((input_seq, target))  # Store the pair

    def __len__(self):
        return len(self.pairs)  # Total samples (input-target pairs) in dataset

    def __getitem__(self, idx):
        input_seq, target = self.pairs[idx]
        # Pad the input sequence on the left with 0 (PAD token index) to ensure fixed length
        if len(input_seq) < self.max_len:
            input_seq = [0] * (self.max_len - len(input_seq)) + input_seq
        # Return input sequence and target as long tensors for PyTorch model consumption
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)

# decoder Model - two-layer LSTM decoder with embedding & final linear classifier
class decoderModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(decoderModel, self).__init__()
        # Embedding layer converts input indices to dense vector representations; ignores padding idx 0 in updates
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        # Two stacked LSTM layers with dropout between layers, batch_first=True means input is (batch, seq, feature)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        # Fully connected linear layer to project from hidden state space to vocabulary size logits
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        embeds = self.embedding(x)  # Get embeddings for input indices
        _, (h_n, _) = self.lstm(embeds)  # Pass embeddings through LSTM, get hidden state of last layer
        out = self.fc(h_n[-1])  # Use last hidden state to predict next item logits
        return out  # Output shape: (batch_size, vocab_size)

# evaluate model on test set
def evaluate_model(model, dataloader, loss_fn, device):
    model.eval()  # Set model to evaluation mode (disables dropout, batchnorm etc.)
    total_loss = 0  # Aggregate loss over batches
    all_preds = []  # Collect predicted indices for all test samples
    all_targets = []  # Collect true target indices for all test samples
    with torch.no_grad():  # Disable gradient calculations for evaluation
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)  # Move to device (cpu/gpu)
            output = model(inputs)  # Forward pass
            loss = loss_fn(output, targets)  # Compute batch loss
            total_loss += loss.item()  # Aggregate loss
            preds = output.argmax(dim=-1).cpu().numpy()  # Get predicted class indices
            all_preds.extend(preds)  # Append predictions
            all_targets.extend(targets.cpu().numpy())  # Append true targets
    avg_loss = total_loss / len(dataloader)  # Average loss over all batches
    # Compute key classification metrics using sklearn weighted macro average, zero_division=0 handles zero divide gracefully
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    return avg_loss, accuracy, precision, recall, f1  # Return evaluation metrics

# predict next item and compare with ground truth
def predict_next(model, input_seq, ground_truth, item2idx, idx2item, max_len=20):
    model.eval()  # Evaluation mode
    device = next(model.parameters()).device  # Get model's device (cpu/gpu)
    # Convert input sequence items to indices, defaulting to <UNK> index for unknown items
    input_ids = [item2idx.get(i, item2idx['<UNK>']) for i in input_seq]

    # Debug print unknown items in input sequence, if any
    if any(item2idx.get(i) is None for i in input_seq):
        print(f"Unknown items in {input_seq}: {[i for i in input_seq if item2idx.get(i) is None]}")

    # Debug print if ground truth item is unknown to vocabulary
    if ground_truth not in item2idx:
        print(f"Ground truth {ground_truth} is unknown")

    # Pad input sequence on left with PAD (index 0) if shorter than max_len
    if len(input_ids) < max_len:
        input_ids = [0] * (max_len - len(input_ids)) + input_ids

    # Use only the last max_len tokens
    input_ids = input_ids[-max_len:]
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)  # Create batch of one input

    with torch.no_grad():
        logits = model(input_tensor)  # Model forward pass
        pred_id = logits.argmax(dim=-1).item()  # Index of most likely next item
        predicted_item = idx2item[pred_id]  # Convert index back to item string

    # Return predicted item and whether prediction matches ground truth
    return predicted_item, predicted_item == ground_truth

# main execution for testing
NUM_RUNS = 3  # Number of repeated inference runs per sample for evaluation stability
num_hadms = 10  # Number of HADM_IDs to test per category

all_prediction_rows = []  # List to collect all prediction records for saving
all_metrics = []  # List to collect overall evaluation metrics

# Log sequence distribution per category before testing
for category, category_key in CATEGORIES.items():
    sequence_tuples = extract_sequences(result, category_key)  # Extract (hadm_id, sequence) tuples
    lengths = [len(seq) for _, seq in sequence_tuples]  # Calculate lengths of each sequence
    print(f"{category}: Sequences={len(sequence_tuples)}, Min length={min(lengths, default=0)}, Avg length={sum(lengths)/len(lengths) if lengths else 0}")
    # Print statistics: number of sequences, minimal and average sequence length

# Process each category to run testing
for category, category_key in CATEGORIES.items():
    print(f"\n=== Processing {category} for Testing ===")

    # Extract sequences for the category
    sequence_tuples = extract_sequences(result, category_key)
    if not sequence_tuples:
        print(f"No valid sequences for {category}. Skipping.")
        continue  # Skip category if no sequences found

    # Split sequences (on HADM_ID level) into train, validation, and test portions (test only here)
    random.seed(42)  # Set random seed for reproducibility
    random.shuffle(sequence_tuples)  # Shuffle admission sequence tuples

    train_size = int(0.8 * len(sequence_tuples))
    val_size = int(0.15 * len(sequence_tuples))
    test_sequences = [seq for _, seq in sequence_tuples[train_size + val_size:]]
    # Keep the last 5% as test sequences

    test_hadm_ids = [hadm_id for hadm_id, seq in sequence_tuples[train_size + val_size:]]
    # Corresponding HADM_IDs for test sequences

    # load saved vocabulary and class weights from disk
    try:
        with open(f"vocab_{category}.pkl", "rb") as f:
            vocab_data = pickle.load(f)
            item2idx = vocab_data["item2idx"]
            idx2item = vocab_data["idx2item"]
            class_weights = vocab_data.get("class_weights")  # Optional: class weights if saved
        print(f"Loaded vocabulary and class weights for {category} from vocab_{category}.pkl")
    except FileNotFoundError:
        print(f"Vocabulary file vocab_{category}.pkl not found. Skipping {category}.")
        continue  # Skip category if vocab file not found

    # create test dataset and dataloader using loaded vocabulary for indexing
    test_dataset = SequenceDataset(test_sequences, item2idx)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # initialize model and load pre-trained weights
    model = decoderModel(vocab_size=len(item2idx), embed_size=64, hidden_size=256)
    try:
        # Load model weights trained earlier from file, map to current device (CPU or GPU)
        model.load_state_dict(torch.load(f"model_{category}.pth", map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
        print(f"Loaded model for {category} from model_{category}.pth")
    except FileNotFoundError:
        print(f"Model file model_{category}.pth not found. Skipping {category}.")
        continue  # Skip category if model file not found

    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Determine device availability
    model = model.to(device)  # Move model to device for computation

    # CrossEntropyLoss with class weights if available, else plain cross entropy
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)

    # Evaluate model on test dataset and compute loss plus all metrics
    test_loss, test_accuracy, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader, loss_fn, device)
    all_metrics.append({
        "Category": category,
        "Epoch": "Test",  # Mark metrics as test phase
        "Train_Loss": None,
        "Val_Loss": test_loss,
        "Val_Accuracy": test_accuracy,
        "Val_Precision": test_precision,
        "Val_Recall": test_recall,
        "Val_F1": test_f1
    })
    print(f"\nTest Results for {category} - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, "
          f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")

    # Predict next item for a subset of test HADM_IDs, multiple runs to confirm prediction consistency
    random.shuffle(test_hadm_ids)  # Shuffle test admission IDs before sampling
    hadm_ids_to_process = test_hadm_ids[:min(num_hadms, len(test_hadm_ids))]  # Select up to num_hadms IDs

    for run in range(1, NUM_RUNS + 1):
        print(f"\nRun {run} Test Set Predictions for {category}...")
        for hadm_id in hadm_ids_to_process:
            seq = result[hadm_id][category_key]  # Get full sequence for this admission and category
            if len(seq) < 2:
                continue  # Skip sequences too short for prediction
            # Use all items but last as input and last item as ground truth for next step prediction
            input_seq = seq[:-1]
            ground_truth = seq[-1]
            predicted_item, is_correct = predict_next(model, input_seq, ground_truth, item2idx, idx2item)
            # Store prediction results in dictionary form to write later to file
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

# save all predictions to an Excel file for analysis
df = pd.DataFrame(all_prediction_rows)  # Convert list of dicts to Pandas DataFrame
df.to_excel("test_predictions.xlsx", index=False)  # Save without row indices
print("\nTest Predictions saved to test_predictions.xlsx")

# save all collected test metrics to Excel for summary and reporting
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_excel("test_metrics.xlsx", index=False)
print("Test Metrics saved to test_metrics.xlsx")
