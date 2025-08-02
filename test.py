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
    # initialize list to store tuples of (hadm_id, item sequence)
    sequence_list = []
    for hadm_id, category_dict in data.items():
        # iterate over each admission and its categorized event data
        items = category_dict.get(category_key, [])
        # retrieve the list of items for the specific category, default empty list if not found
        # ensure there are at least 2 items in sequence for prediction
        if len(items) >= 2:
            sequence_list.append((hadm_id, items))
            # append tuple (HADM_ID, sequence of items) to list
    return sequence_list

# dataset for sequences
class SequenceDataset(Dataset):
    # custom PyTorch dataset to handle input-target pairs generated from event sequences
    def __init__(self, sequences, item2idx, max_len=20):
        # list to hold (input_sequence, target_item) pairs
        self.pairs = []
        # max sequence length for model input
        self.max_len = max_len
        # dictionary mapping items to indices
        self.item2idx = item2idx
        for seq in sequences:
            # convert each item in sequence to corresponding index; use <UNK> if not found
            idx_seq = [item2idx.get(item, item2idx['<UNK>']) for item in seq]
            # for each position except first, create input-target pair
            for i in range(1, len(idx_seq)):
                # take last max_len tokens before i as input (sliding window)
                input_seq = idx_seq[:i][-max_len:]
                # target is item at position i
                target = idx_seq[i]
                # store the pair
                self.pairs.append((input_seq, target))

    def __len__(self):
        # total samples (input-target pairs) in dataset
        return len(self.pairs)

    def __getitem__(self, idx):
        input_seq, target = self.pairs[idx]
        # pad the input sequence on the left with 0 (PAD token index) to ensure fixed length
        if len(input_seq) < self.max_len:
            input_seq = [0] * (self.max_len - len(input_seq)) + input_seq
        # return input sequence and target as long tensors for PyTorch model consumption
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)

# decoder Model - two-layer LSTM decoder with embedding & final linear classifier
class decoderModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(decoderModel, self).__init__()
        # embedding layer converts input indices to dense vector representations; ignores padding idx 0 in updates
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        # two stacked LSTM layers with dropout between layers, batch_first=True means input is (batch, seq, feature)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        # fully connected linear layer to project from hidden state space to vocabulary size logits
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        # get embeddings for input indices
        embeds = self.embedding(x)
        # pass embeddings through LSTM, get hidden state of last layer
        _, (h_n, _) = self.lstm(embeds)
        # use last hidden state to predict next item logits
        out = self.fc(h_n[-1])
        # output shape: (batch_size, vocab_size)
        return out

# evaluate model on test set
def evaluate_model(model, dataloader, loss_fn, device):
    # set model to evaluation mode (disables dropout, batchnorm etc.)
    model.eval()
    # aggregate loss over batches
    total_loss = 0
    # collect predicted indices for all test samples
    all_preds = []
    # collect true target indices for all test samples
    all_targets = []
    # disable gradient calculations for evaluation
    with torch.no_grad():
        for inputs, targets in dataloader:
            # move to device (cpu/gpu)
            inputs, targets = inputs.to(device), targets.to(device)
            # forward pass
            output = model(inputs)
            # compute batch loss
            loss = loss_fn(output, targets)
            # aggregate loss
            total_loss += loss.item()
            # get predicted class indices
            preds = output.argmax(dim=-1).cpu().numpy()
            # append predictions
            all_preds.extend(preds)
            # append true targets
            all_targets.extend(targets.cpu().numpy())
    # average loss over all batches
    avg_loss = total_loss / len(dataloader)
    # compute key classification metrics using sklearn weighted macro average, zero_division=0 handles zero divide gracefully
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    return avg_loss, accuracy, precision, recall, f1

# flattened mapping of item IDs to label
flat_itemid_label_mapping = flatten_itemid_label_mappings(itemid_label_mappings)

# predict next item and compare with ground truth
def predict_next(model, input_seq, ground_truth, item2idx, idx2item, max_len=20, category=None, itemid_label_mappings_flat=None):
    # evaluation mode
    model.eval()
    # get model's device (cpu/gpu)
    device = next(model.parameters()).device
    # convert input sequence items to indices, defaulting to <UNK> index for unknown items
    input_ids = [item2idx.get(i, item2idx['<UNK>']) for i in input_seq]

    # debug print unknown items in input sequence, if any
    if any(item2idx.get(i) is None for i in input_seq):
        print(f"Unknown items in {input_seq}: {[i for i in input_seq if item2idx.get(i) is None]}")

    # debug print if ground truth item is unknown to vocabulary
    if ground_truth not in item2idx:
        print(f"Ground truth {ground_truth} is unknown")

    # pad input sequence on left with PAD (index 0) if shorter than max_len
    if len(input_ids) < max_len:
        input_ids = [0] * (max_len - len(input_ids)) + input_ids

    # use only the last max_len tokens
    input_ids = input_ids[-max_len:]
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    # forward pass
    output_logits = model(input_tensor)
    # predicted index
    predicted_idx = torch.argmax(output_logits, dim=-1).item()
    # map index back to item ID or label
    predicted_item = idx2item.get(predicted_idx, '<UNK>')

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


# main execution for testing
# number of prediction runs
NUM_RUNS = 3
# number of ids to use
num_hadms = 10

# list to collect all prediction records for saving
all_prediction_rows = []
# list to collect overall evaluation metrics
all_metrics = []

# log sequence distribution per category before testing
for category, category_key in CATEGORIES.items():
    # extract (hadm_id, sequence) tuples
    sequence_tuples = extract_sequences(result, category_key)
    # calculate lengths of each sequence
    lengths = [len(seq) for _, seq in sequence_tuples]
    # print statistics: number of sequences, minimal and average sequence length
    print(f"{category}: Sequences={len(sequence_tuples)}, Min length={min(lengths, default=0)}, Avg length={sum(lengths)/len(lengths) if lengths else 0}")

# process each category to run testing
for category, category_key in CATEGORIES.items():
    print(f"\n=== Processing {category} for Testing ===")

    # extract sequences for the category
    sequence_tuples = extract_sequences(result, category_key)
    if not sequence_tuples:
        # Skip category if no sequences found
        print(f"No valid sequences for {category}. Skipping.")
        continue

    # split sequences (on HADM_ID level) into train, validation, and test portions (test only here)
    # set random seed for reproducibility
    random.seed(42)
    # shuffle admission sequence tuples
    random.shuffle(sequence_tuples)

    train_size = int(0.8 * len(sequence_tuples))
    val_size = int(0.15 * len(sequence_tuples))
    # keep the last 5% as test sequences
    test_sequences = [seq for _, seq in sequence_tuples[train_size + val_size:]]

    # corresponding HADM_IDs for test sequences
    test_hadm_ids = [hadm_id for hadm_id, seq in sequence_tuples[train_size + val_size:]]

    # load saved vocabulary and class weights from disk
    try:
        with open(f"vocab_{category}.pkl", "rb") as f:
            vocab_data = pickle.load(f)
            item2idx = vocab_data["item2idx"]
            idx2item = vocab_data["idx2item"]
            # optional: class weights if saved
            class_weights = vocab_data.get("class_weights")
        print(f"Loaded vocabulary and class weights for {category} from vocab_{category}.pkl")
    except FileNotFoundError:
        # skip category if vocab file not found
        print(f"Vocabulary file vocab_{category}.pkl not found. Skipping {category}.")
        continue

    # create test dataset and dataloader using loaded vocabulary for indexing
    test_dataset = SequenceDataset(test_sequences, item2idx)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # initialize model and load pre-trained weights
    model = decoderModel(vocab_size=len(item2idx), embed_size=64, hidden_size=256)
    try:
        # load model weights trained earlier from file, map to current device (CPU or GPU)
        model.load_state_dict(torch.load(f"model_{category}.pth", map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
        print(f"Loaded model for {category} from model_{category}.pth")
    except FileNotFoundError:
        # skip category if model file not found
        print(f"Model file model_{category}.pth not found. Skipping {category}.")
        continue

    # determine device availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # move model to device for computation
    model = model.to(device)

    # CrossEntropyLoss with class weights if available, else plain cross entropy
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)

    # evaluate model on test dataset and compute loss plus all metrics
    test_loss, test_accuracy, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader, loss_fn, device)
    all_metrics.append({
        "Category": category,
        "Epoch": "Test",  # mark metrics as test phase
        "Train_Loss": None,
        "Val_Loss": test_loss,
        "Val_Accuracy": test_accuracy,
        "Val_Precision": test_precision,
        "Val_Recall": test_recall,
        "Val_F1": test_f1
    })
    print(f"\nTest Results for {category} - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, "
          f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")

    # predict next item for a subset of test HADM_IDs, multiple runs to confirm prediction consistency
    # shuffle test admission IDs before sampling
    random.shuffle(test_hadm_ids)
    hadm_ids_to_process = test_hadm_ids[:min(num_hadms, len(test_hadm_ids))]  # Select up to num_hadms IDs

    for run in range(1, NUM_RUNS + 1):
        print(f"\nRun {run} Test Set Predictions for {category}...")
        for hadm_id in hadm_ids_to_process:
            # get full sequence for this admission and category
            seq = result[hadm_id][category_key]
            # skip sequences too short for prediction
            if len(seq) < 2:
                continue
            # use all items but last as input and last item as ground truth for next step prediction
            input_seq = seq[:-1]
            ground_truth = seq[-1]
            predicted_item, predicted_label, ground_truth_label, is_correct = predict_next(
                model, input_seq, ground_truth, item2idx, idx2item, category=category, itemid_label_mappings_flat=flat_itemid_label_mapping)
            # store prediction results in dictionary form to write later to file
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
                "Dataset": "Test"
            })

# save all predictions to an Excel file for analysis
df = pd.DataFrame(all_prediction_rows)
df.to_excel("test_predictions.xlsx", index=False)
print("\nTest Predictions saved to test_predictions.xlsx")

# save all collected test metrics to Excel for summary and reporting
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_excel("test_metrics.xlsx", index=False)
print("Test Metrics saved to test_metrics.xlsx")
