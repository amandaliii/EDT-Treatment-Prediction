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
# stores and exports metrics into Excel file
import pandas as pd
# to calculate precision, recall, and f1 score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
# to save vocab
import pickle

# load dataset from dataprocessing
mimic_data_dir = "/Users/amandali/Downloads/Mimic III"
# loads how many rows of mimic 3 data
result = load_mimic3_data(mimic_data_dir, nrows=2000000)
# Calls the data loading function from your DataProcessing module to load up to 2 million rows of MIMIC-III data from the specified directory.
# The result is assumed to be a dictionary keyed by HADM_IDs containing category dicts.

# category configuration maps event categories to their item keys in dataset
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
    sequence_list = []  # Will hold tuples (hadm_id, item sequence)
    for hadm_id, category_dict in data.items():
        items = category_dict.get(category_key, [])
        # Get the list of items for the given category in this hospital admission
        if len(items) >= 2:  # Ensure sequence has at least 2 items to be meaningful for prediction
            sequence_list.append((hadm_id, items))
    return sequence_list  # Return list of (hadm_id, items) tuples

# build vocabulary for a category
def build_vocab(sequences):
    item_counts = Counter()  # Initialize counter for item frequencies
    for seq in sequences:
        item_counts.update(seq)  # Update frequency counts with items from sequence
    vocab = ['<PAD>', '<UNK>'] + [item for item, _ in item_counts.most_common()]
    # Vocabulary list starting with special tokens for padding and unknown items, followed by items sorted by frequency
    item2idx = {item: i for i, item in enumerate(vocab)}  # Map from item string to index
    idx2item = {i: item for item, i in item2idx.items()}  # Reverse mapping from index to item string

    # compute class weights for CrossEntropyLoss to handle imbalanced classes
    total = sum(item_counts.values())  # Total number of item occurrences
    class_weights = torch.tensor(
        [total / (len(item_counts) * count) if count > 0 else 1.0 for item, count in item_counts.most_common()],
        dtype=torch.float)
    # Weight inversely proportional to frequency for each item in vocab (excluding PAD and UNK)
    class_weights = torch.cat([torch.tensor([1.0, 1.0]), class_weights])  # Weights for <PAD>, <UNK> tokens set as 1.0
    return item2idx, idx2item, class_weights  # Return vocab dicts and class weights tensor

# dataset for sequences, inherits PyTorch Dataset for batching & shuffling
class SequenceDataset(Dataset):
    def __init__(self, sequences, item2idx, max_len=20):
        self.pairs = []  # List to store (input sequence, target) pairs for training
        self.max_len = max_len  # Maximum length of input sequences
        self.item2idx = item2idx  # Vocabulary mapping
        for seq in sequences:
            # Convert each item into its index in vocab, using <UNK> index if missing
            idx_seq = [item2idx.get(item, item2idx['<UNK>']) for item in seq]
            # Generate pairs: for each item except the first, predict that item given previous items
            for i in range(1, len(idx_seq)):
                input_seq = idx_seq[:i][-max_len:]  # Use sliding window of size max_len for input
                target = idx_seq[i]  # Target is current item
                self.pairs.append((input_seq, target))  # Store pair

    def __len__(self):
        return len(self.pairs)  # Number of training pairs

    def __getitem__(self, idx):
        input_seq, target = self.pairs[idx]
        # Pad input sequence on left with 0 (index of <PAD>) if shorter than max length
        if len(input_seq) < self.max_len:
            input_seq = [0] * (self.max_len - len(input_seq)) + input_seq
        # Return input sequence tensor and target tensor for given index
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)

# model definition using two-layer LSTM with dropout and final linear projection
class decoderModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(decoderModel, self).__init__()
        # Embedding layer maps indices to dense vectors; padding_idx=0 means that index 0 (PAD) is ignored in embedding updates
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        # 2 stacked LSTM layers, output batch_first, with dropout between layers to reduce overfitting
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        # Linear layer maps last hidden state to logits over the vocabulary
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        embeds = self.embedding(x)  # Convert input indices to embeddings
        _, (h_n, _) = self.lstm(embeds)  # Pass embeddings through LSTM, get hidden states from last layer
        out = self.fc(h_n[-1])  # Use the last hidden state of last LSTM layer as input to fully connected layer
        return out  # Return the output logits for prediction

# evaluate model on validation set with metrics calculation
def evaluate_model(model, dataloader, loss_fn, device):
    model.eval()  # Set model to evaluation mode disables dropout etc.
    total_loss = 0
    all_preds = []  # Collect predictions
    all_targets = []  # Collect true targets
    with torch.no_grad():  # Disable gradient calculations for efficiency
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to device (CPU/GPU)
            output = model(inputs)  # Forward pass
            loss = loss_fn(output, targets)  # Calculate loss for the batch
            total_loss += loss.item()  # Accumulate loss
            preds = output.argmax(dim=-1).cpu().numpy()  # Get predicted indices
            all_preds.extend(preds)  # Append to list
            all_targets.extend(targets.cpu().numpy())  # Append ground truth
    avg_loss = total_loss / len(dataloader)  # Average loss over all batches
    # Calculate accuracy, precision, recall and F1 (macro averaged, ignoring divide by zero warnings)
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    return avg_loss, accuracy, precision, recall, f1  # Return metrics

# predict the next item from a given input sequence and check if prediction matches ground truth
def predict_next(model, input_seq, ground_truth, item2idx, idx2item, max_len=20):
    model.eval()  # Evaluation mode
    device = next(model.parameters()).device  # Get current device (cpu/gpu)
    input_ids = [item2idx.get(i, item2idx['<UNK>']) for i in input_seq]  # Map input sequence items to indices, use <UNK> if not found
    # Pad if needed to max_len with PAD (index 0)
    if len(input_ids) < max_len:
        input_ids = [0] * (max_len - len(input_ids)) + input_ids
    input_ids = input_ids[-max_len:]  # Keep only max_len most recent tokens for context
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)  # Create tensor with batch size 1
    with torch.no_grad():
        logits = model(input_tensor)  # Forward pass
        pred_id = logits.argmax(dim=-1).item()  # Select most probable next item index
        predicted_item = idx2item[pred_id]  # Convert predicted index to item string
    return predicted_item, predicted_item == ground_truth  # Return predicted item and correctness boolean

# train the model with train/validation split and class weights for imbalanced classes
def train_model(model, train_loader, val_loader, class_weights, epochs, lr=1e-4):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
    model = model.to(device)  # Move model to device
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))  # Use weighted cross-entropy loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  # Adam optimizer with weight decay for regularization
    metrics = []  # List to collect training/validation stats

    for epoch in range(epochs):
        # training loop
        model.train()  # Set model to training mode
        total_train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # Move batch to device
            optimizer.zero_grad()  # Reset gradients
            output = model(inputs)  # Forward pass
            loss = loss_fn(output, targets)  # Calculate loss
            loss.backward()  # Backpropagation
            # gradient clipping to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()  # Update parameters
            total_train_loss += loss.item()  # Accumulate loss for reporting
        avg_train_loss = total_train_loss / len(train_loader)  # Average loss per batch

        # validation evaluation
        avg_val_loss, val_accuracy, val_precision, val_recall, val_f1 = evaluate_model(model, val_loader, loss_fn, device)
        # Save metrics for this epoch; 'Category' will be assigned later in main code loop
        metrics.append({
            "Epoch": epoch + 1,
            "Category": None,
            "Train_Loss": avg_train_loss,
            "Val_Loss": avg_val_loss,
            "Val_Accuracy": val_accuracy,
            "Val_Precision": val_precision,
            "Val_Recall": val_recall,
            "Val_F1": val_f1
        })
        # Print progress info
        print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
              f"Val Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
    return metrics  # Return history of metrics over epochs

# main execution for training and validation over different event categories
NUM_RUNS = 3  # Number of repeated prediction runs for evaluation
num_hadms = 10  # Number of hospital admissions to predict for per run
all_prediction_rows = []  # Collect prediction results
all_metrics = []  # Collect training metrics

for category, category_key in CATEGORIES.items():
    print(f"\n=== Processing {category} for Training/Validation ===")

    # extract sequences and print stats
    sequence_tuples = extract_sequences(result, category_key)
    lengths = [len(seq) for _, seq in sequence_tuples]
    print(
        f"{category}: Sequences={len(sequence_tuples)}, Min length={min(lengths, default=0)}, Avg length={sum(lengths) / len(lengths) if lengths else 0}")

    if not sequence_tuples:
        print(f"No valid sequences for {category}. Skipping.")
        continue

    # split into train and validation subsets with fixed random seed for reproducibility
    random.seed(42)
    random.shuffle(sequence_tuples)
    train_size = int(0.8 * len(sequence_tuples))
    val_size = int(0.15 * len(sequence_tuples))
    train_sequences = [seq for _, seq in sequence_tuples[:train_size]]
    val_sequences = [seq for _, seq in sequence_tuples[train_size:train_size + val_size]]
    train_hadm_ids = [hadm_id for hadm_id, seq in sequence_tuples[:train_size]]
    val_hadm_ids = [hadm_id for hadm_id, seq in sequence_tuples[train_size:train_size + val_size]]

    # build vocabulary and class weights
    item2idx, idx2item, class_weights = build_vocab(train_sequences + val_sequences)

    # save vocabulary to disk for later reuse
    with open(f"vocab_{category}.pkl", "wb") as f:
        pickle.dump({"item2idx": item2idx, "idx2item": idx2item}, f)
    print(f"Vocabulary for {category} saved to vocab_{category}.pkl")

    # create PyTorch datasets and dataloaders for batching and shuffling
    train_dataset = SequenceDataset(train_sequences, item2idx)
    val_dataset = SequenceDataset(val_sequences, item2idx)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # initialize the model with vocabulary size, embedding dimension, and larger hidden state size
    model = decoderModel(vocab_size=len(item2idx), embed_size=64, hidden_size=256)
    # train the model with weighted loss for class imbalance for 10 epochs
    metrics = train_model(model, train_loader, val_loader, class_weights, epochs=10)
    for metric in metrics:
        metric['Category'] = category  # Assign category for logging and reporting
        all_metrics.append(metric)

    # save the trained model parameters for future loading/evaluation
    torch.save(model.state_dict(), f"model_{category}.pth")
    print(f"Model for {category} saved to model_{category}.pth")

    # perform predictions on a subset of admissions from train and validation sets
    for dataset_name, hadm_ids in [("Train", train_hadm_ids), ("Validation", val_hadm_ids)]:
        random.shuffle(hadm_ids)
        hadm_ids_to_process = hadm_ids[:min(num_hadms, len(hadm_ids))]
        for run in range(1, NUM_RUNS + 1):  # Repeat predictions multiple times per admission
            print(f"\nRun {run} {dataset_name} Set Predictions for {category}...")
            for hadm_id in hadm_ids_to_process:
                seq = result[hadm_id][category_key]
                if len(seq) < 2:
                    continue  # Skip too short sequences
                # Use all but last item as input sequence, last item as ground truth label for prediction
                input_seq = seq[:-1]
                ground_truth = seq[-1]
                predicted_item, is_correct = predict_next(model, input_seq, ground_truth, item2idx, idx2item)
                # Record prediction results in dictionary for output
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

# save all collected predictions to Excel for analysis
df = pd.DataFrame(all_prediction_rows)
df.to_excel("train_val_predictions.xlsx", index=False)
print("\nTraining and Validation Predictions saved to train_val_predictions.xlsx")

# save metrics collected during training/validation to Excel
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_excel("train_val_metrics.xlsx", index=False)
print("Training and Validation Metrics saved to train_val_metrics.xlsx")
