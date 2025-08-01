# import dataset from dataprocessing file
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
# import ITEMID-to-label mapping function from itemID.py
from itemID import create_itemid_label_mapping, load_labitems_labels

# load dataset from dataprocessing
mimic_data_dir = "/Users/amandali/Downloads/Mimic III"
# loads how many rows of mimic 3 data
result = load_mimic3_data(mimic_data_dir, nrows=2000000)
# load itemID to label mapping and check for duplicates (from D_ITEMS.csv)
itemid_label_mappings, duplicates = create_itemid_label_mapping(f"{mimic_data_dir}/D_ITEMS.csv")

# load lab item labels separately (from D_LABITEMS.csv)
lab_items_labels = load_labitems_labels(f"{mimic_data_dir}/D_LABITEMS.csv")

# merge lab labels into the 'labevents' category in your main itemid_label_mappings:
if 'labevents' not in itemid_label_mappings:
    itemid_label_mappings['labevents'] = {}
for itemid, label in lab_items_labels.items():
    itemid_label_mappings['labevents'][itemid] = label

# debugging: print mapping details to verify content including merged lab labels
print("\nLoaded itemid_label_mappings (including lab items):")
for category, items in itemid_label_mappings.items():
    print(f"  Category: {category}, Number of ITEMIDs: {len(items)}")
    # Print a few sample ITEMIDs and labels for each category
    sample_items = list(items.items())[:3]
    for itemid, label in sample_items:
        print(f"    ITEMID: {itemid} -> Label: {label}")

# print duplicates if any
if duplicates:
    print("\nDuplicate ITEMIDs found in D_ITEMS.csv:")
    for dup in duplicates:
        print(f"  ITEMID: {dup['ITEMID']}")
        print(f"    First occurrence: Label='{dup['First_Label']}', Category='{dup['First_Category']}'")
        print(f"    Second occurrence: Label='{dup['Second_Label']}', Category='{dup['Second_Category']}'")
else:
    print("\nNo duplicate ITEMIDs found in D_ITEMS.csv.")

# category configuration maps event categories to their item keys in dataset
CATEGORIES = {
    'chart_events': 'chart_items',
    'input_events': 'input_items',
    'lab_events': 'lab_items',
    'microbiology_events': 'microbiology_items',
    'prescriptions': 'prescriptions_items',
    'procedure_events': 'procedure_items'
}

# map model categories to D_ITEMS.csv categories for label lookup
CATEGORY_TO_D_ITEMS = {
    'chartevents': 'CHART',
    'inputevents': 'INPUT',
    'labevents': 'LAB',
    'microbiologyevents': 'MICROBIOLOGY',
    'prescriptions': 'PRESCRIPTIONS',
    'procedureevents': 'PROCEDURE'
}

# combine all category mappings into a single ITEMID -> label dictionary
def flatten_itemid_label_mappings(itemid_label_mappings):
    combined_mapping = {}
    for category_dict in itemid_label_mappings.values():
        combined_mapping.update(category_dict)
    return combined_mapping

# extract sequences for a specific category with HADM_IDs
def extract_sequences(data, category_key):
    # will hold tuples (hadm_id, item sequence)
    sequence_list = []
    for hadm_id, category_dict in data.items():
        items = category_dict.get(category_key, [])
        # get the list of items for the given category in this hospital admission
        # ensure sequence has at least 2 items to be meaningful for prediction
        if len(items) >= 2:
            sequence_list.append((hadm_id, items))
    # return list of (hadm_id, items) tuples
    return sequence_list

# build vocabulary for a category
def build_vocab(sequences):
    # initialize counter for item frequencies
    item_counts = Counter()
    for seq in sequences:
        # convert items to strings
        item_counts.update(str(item) for item in seq)
        # update frequency counts with items from sequence
        item_counts.update(seq)
    # vocabulary list starting with special tokens for padding and unknown items, followed by items sorted by frequency
    vocab = ['<PAD>', '<UNK>'] + [item for item, _ in item_counts.most_common()]
    # map from item string to index
    item2idx = {item: i for i, item in enumerate(vocab)}
    # reverse mapping from index to item string
    idx2item = {i: item for item, i in item2idx.items()}

    # compute class weights for CrossEntropyLoss to handle imbalanced classes
    total = sum(item_counts.values())  # total number of item occurrences
    class_weights = torch.tensor(
        [total / (len(item_counts) * count) if count > 0 else 1.0 for item, count in item_counts.most_common()],
        dtype=torch.float)
    # weight inversely proportional to frequency for each item in vocab (excluding PAD and UNK)
    # weights for <PAD>, <UNK> tokens set as 1.0
    class_weights = torch.cat([torch.tensor([1.0, 1.0]), class_weights])
    # return vocab dicts and class weights tensor
    return item2idx, idx2item, class_weights

# dataset for sequences, inherits PyTorch Dataset for batching & shuffling
class SequenceDataset(Dataset):
    def __init__(self, sequences, item2idx, max_len=20):
        # list to store (input sequence, target) pairs for training
        self.pairs = []
        # maximum length of input sequences
        self.max_len = max_len
        # vocabulary mapping
        self.item2idx = item2idx
        for seq in sequences:
            # convert each item into its index in vocab, using <UNK> index if missing
            # convert items to strings
            idx_seq = [item2idx.get(str(item), item2idx['<UNK>']) for item in seq]
            # generate pairs: for each item except the first, predict that item given previous items
            for i in range(1, len(idx_seq)):
                # use sliding window of size max_len for input
                input_seq = idx_seq[:i][-max_len:]
                # target is current item
                target = idx_seq[i]
                # store pair
                self.pairs.append((input_seq, target))

    def __len__(self):
        # number of training pairs
        return len(self.pairs)

    def __getitem__(self, idx):
        input_seq, target = self.pairs[idx]
        # pad input sequence on left with 0 (index of <PAD>) if shorter than max length
        if len(input_seq) < self.max_len:
            input_seq = [0] * (self.max_len - len(input_seq)) + input_seq
        # return input sequence tensor and target tensor for given index
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)

# model definition using two-layer LSTM with dropout and final linear projection
class decoderModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(decoderModel, self).__init__()
        # embedding layer maps indices to dense vectors; padding_idx=0 means that index 0 (PAD) is ignored in embedding updates
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        # 2 stacked LSTM layers, output batch_first, with dropout between layers to reduce overfitting
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        # linear layer maps last hidden state to logits over the vocabulary
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        # convert input indices to embeddings
        embeds = self.embedding(x)
        # pass embeddings through LSTM, get hidden states from last layer
        _, (h_n, _) = self.lstm(embeds)
        # use the last hidden state of last LSTM layer as input to fully connected layer
        out = self.fc(h_n[-1])
        # return the output logits for prediction
        return out

# evaluate model on validation set with metrics calculation
def evaluate_model(model, dataloader, loss_fn, device):
    # set model to evaluation mode disables dropout etc.
    model.eval()
    total_loss = 0
    # collect predictions
    all_preds = []
    # collect true targets
    all_targets = []
    # disable gradient calculations for efficiency
    with torch.no_grad():
        for inputs, targets in dataloader:
            # move data to device (CPU/GPU)
            inputs, targets = inputs.to(device), targets.to(device)
            # forward pass
            output = model(inputs)
            # calculate loss for the batch
            loss = loss_fn(output, targets)
            # calculate total loss
            total_loss += loss.item()
            # get predicted indices
            preds = output.argmax(dim=-1).cpu().numpy()
            # append to list
            all_preds.extend(preds)
            # append ground truth
            all_targets.extend(targets.cpu().numpy())
    # average loss over all batches
    avg_loss = total_loss / len(dataloader)
    # calculate accuracy, precision, recall and F1 (macro averaged, ignoring divide by zero warnings)
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    return avg_loss, accuracy, precision, recall, f1

# flattened mapping of item IDs to label
flat_itemid_label_mapping = flatten_itemid_label_mappings(itemid_label_mappings)

# predict the next item from a given input sequence and check if prediction matches ground truth
def predict_next(model, input_seq, ground_truth, item2idx, idx2item, max_len=20, category=None, itemid_label_mappings_flat=None):
    # evaluation mode
    model.eval()
    # get current device (cpu/gpu)
    device = next(model.parameters()).device
    # convert input sequence items to strings to match itemid_label_mappings
    # map input sequence items to indices
    input_ids = [item2idx.get(str(i), item2idx['<UNK>']) for i in input_seq]
    # pad if needed to max_len with PAD (index 0)
    if len(input_ids) < max_len:
        input_ids = [0] * (max_len - len(input_ids)) + input_ids
    # keep only max_len most recent tokens for context
    input_ids = input_ids[-max_len:]
    # create tensor with batch size 1
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    with torch.no_grad():
        # forward pass
        logits = model(input_tensor)
        # select most probable next item index
        pred_id = logits.argmax(dim=-1).item()
        # convert predicted index to item string
        predicted_item = idx2item[pred_id]

    # get labels for predicted and ground truth ITEMIDs
    d_items_category = CATEGORY_TO_D_ITEMS.get(category, 'Uncategorized')
    # debug: print lookup details
    print(f"\nPredicting for Category: {category}, D_ITEMS Category: {d_items_category}")
    print(f"  Ground Truth ITEMID: {ground_truth}, Predicted ITEMID: {predicted_item}")

    # lookup labels from the flat mapping (fallback to 'Unknown')
    predicted_label = itemid_label_mappings_flat.get(str(predicted_item), 'Unknown')
    ground_truth_label = itemid_label_mappings_flat.get(str(ground_truth), 'Unknown')

    print(f"\nPredicted ITEMID: {predicted_item}, Label: {predicted_label}")
    print(f"Ground Truth ITEMID: {ground_truth}, Label: {ground_truth_label}")

    return predicted_item, predicted_label, ground_truth_label, str(predicted_item) == str(ground_truth)

# train the model with train/validation split and class weights for imbalanced classes
def train_model(model, train_loader, val_loader, class_weights, epochs, lr=1e-4):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
    model = model.to(device)  # Move model to device
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))  # Use weighted cross-entropy loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  # Adam optimizer with weight decay for regularization
    # list to collect training/validation stats
    metrics = []

    for epoch in range(epochs):
        # set model to training mode
        model.train()
        total_train_loss = 0
        for inputs, targets in train_loader:
            # move batch to device
            inputs, targets = inputs.to(device), targets.to(device)
            # reset gradients
            optimizer.zero_grad()
            # forward pass
            output = model(inputs)
            # calculate loss
            loss = loss_fn(output, targets)
            # backpropagation
            loss.backward()
            # gradient clipping to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # update parameters
            optimizer.step()
            # accumulate loss for reporting
            total_train_loss += loss.item()
        # average loss per batch
        avg_train_loss = total_train_loss / len(train_loader)

        # validation evaluation
        avg_val_loss, val_accuracy, val_precision, val_recall, val_f1 = evaluate_model(model, val_loader, loss_fn, device)
        # save metrics for this epoch; 'Category' will be assigned later in main code loop
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
        # print progress info
        print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
              f"Val Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
    return metrics

# main execution for training and validation over different event categories
# number of repeated prediction runs for evaluation
NUM_RUNS = 3
# number of hospital admission IDs to predict for per run
num_hadms = 10
# collect prediction results
all_prediction_rows = []
# collect training metrics
all_metrics = []

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

    # debugging: print sample ITEMIDs from sequences
    if train_sequences:
        print(f"Sample ITEMIDs from {category}: {train_sequences[0][:3]}")

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
        # assign category for logging and reporting
        metric['Category'] = category
        all_metrics.append(metric)

    # save the trained model parameters for future loading/evaluation
    torch.save(model.state_dict(), f"model_{category}.pth")
    print(f"Model for {category} saved to model_{category}.pth")

    # perform predictions on a subset of admissions from train and validation sets
    for dataset_name, hadm_ids in [("Train", train_hadm_ids), ("Validation", val_hadm_ids)]:
        random.shuffle(hadm_ids)
        hadm_ids_to_process = hadm_ids[:min(num_hadms, len(hadm_ids))]
        # repeat predictions multiple times per admission
        for run in range(1, NUM_RUNS + 1):
            print(f"\nRun {run} {dataset_name} Set Predictions for {category}...")
            for hadm_id in hadm_ids_to_process:
                seq = result[hadm_id][category_key]
                # skip too short sequences
                if len(seq) < 2:
                    continue
                # use all but last item as input sequence, last item as ground truth label for prediction
                input_seq = seq[:-1]
                ground_truth = seq[-1]
                predicted_item, predicted_label, ground_truth_label, is_correct = predict_next(
                    model, input_seq, ground_truth, item2idx, idx2item, category=category, itemid_label_mappings_flat=flat_itemid_label_mapping)
                # record prediction results in dictionary for output
                all_prediction_rows.append({
                    "Run": run,
                    "Category": category,
                    "HADM_ID": hadm_id,
                    "Input_Sequence": ", ".join(map(str, input_seq)),
                    "Ground_Truth_ITEMID": ground_truth,
                    "Ground_Truth_Label": ground_truth_label,
                    "Predicted_ITEMID": predicted_item,
                    "Predicted_Label": predicted_label,
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
