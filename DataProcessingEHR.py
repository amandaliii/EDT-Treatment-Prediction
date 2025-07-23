import pandas as pd
import os
from collections import Counter

# mimic data directory (stored locally)
mimic_data_dir = '/Users/amandali/Downloads/Mimic III'

# mimic_3data is the directory path for the data in my local file
def load_mimic3_data(mimic_3data, nrows):
    # data dictionary mapping - file name, sort columns, and grouping column
    data_files = {
        'chart_events': ('CHARTEVENTS.csv.gz', ['HADM_ID', 'CHARTTIME'], 'ITEMID'),
        'input_events': ('INPUTEVENTS_MV.csv.gz', ['HADM_ID', 'STARTTIME'], 'ITEMID'),
        'lab_events': ('LABEVENTS.csv.gz', ['HADM_ID', 'CHARTTIME'], 'ITEMID'),
        'microbiology_events': ('MICROBIOLOGYEVENTS.csv.gz', ['HADM_ID', 'CHARTTIME'], 'SPEC_ITEMID'),
        'prescriptions': ('PRESCRIPTIONS.csv.gz', ['HADM_ID', 'STARTDATE'], 'DRUG'),
        'procedure_events': ('PROCEDUREEVENTS_MV.csv.gz', ['HADM_ID', 'STARTTIME'], 'ITEMID'),
    }

    # store grouped data for each category
    data_dicts = {}
    for key, (file_name, sort_cols, group_col) in data_files.items():
        file_path = os.path.join(mimic_3data, file_name)
        # check if file exists at specified path - if not, assign empty and move on
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            data_dicts[key] = {}
            continue
        # try and read and process other files that are found
        try:
            df = pd.read_csv(file_path, compression='gzip', nrows=nrows, usecols=sort_cols + [group_col])
            # check if dataframe is empty
            if df.empty:
                print(f"Empty DataFrame: {file_path}")
                # if dataframe is empty then assign empty and move on
                data_dicts[key] = {}
            else:
                # check if required columns exist
                missing_cols = [col for col in sort_cols + [group_col] if col not in df.columns]
                # if any required columns are missing
                if missing_cols:
                    print(f"Missing columns: {missing_cols} in {file_name}")
                    data_dicts[key] = {}
                else:
                    # sort dataframe by specified columns
                    sorted_df = df.sort_values(by=sort_cols)
                    # group by HADM_ID and collect the group_col into lists, convert to dict
                    data_dicts[key] = sorted_df.groupby('HADM_ID')[group_col].apply(list).to_dict()
                    print(f"Loaded {file_name} with {len(df)} rows")
        # catch exceptions/errors
        except Exception as e:
            print(f"Error reading {file_name} : {e}")
            data_dicts[key] = {}


    # empty dict for merged results
    merged_dict = {}
    # stores all unique hadm_ids
    all_hadm_ids = set()
    # add all HADM_IDs from the dict to the set
    for d in data_dicts.values():
        all_hadm_ids.update(d.keys())

    for hadm_id in all_hadm_ids:
        current_dict = {
            # get list of ITEMID for chart events, default to empty list if not found
            'chart_items': data_dicts['chart_events'].get(hadm_id, []),
            'input_items': data_dicts['input_events'].get(hadm_id, []),
            'lab_items': data_dicts['lab_events'].get(hadm_id, []),
            'microbiology_items': data_dicts['microbiology_events'].get(hadm_id, []),
            'prescriptions_items': data_dicts['prescriptions'].get(hadm_id, []),
            'procedure_items': data_dicts['procedure_events'].get(hadm_id, [])
        }
        # only include the HADM_ID if all categories have non-empty lists
        # Original strict version
        if all(len(items) > 0 for items in current_dict.values()):
            merged_dict[hadm_id] = current_dict

        # Try relaxed optionally:
        if any(len(items) > 0 for items in current_dict.values()):
            merged_dict[hadm_id] = current_dict

    return merged_dict

result = load_mimic3_data(mimic_data_dir, nrows=1000)

# 1. Build vocab from merged_dict
def build_vocab(merged_dict):
    all_codes = []
    for hadm_data in merged_dict.values():
        for cat in ['chart_items', 'input_items', 'lab_items',
                    'microbiology_items', 'prescriptions_items', 'procedure_items']:
            all_codes.extend(str(code) for code in hadm_data[cat])

    counter = Counter(all_codes)
    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
        "<BOS>": 2,
        "<EOS>": 3
    }
    # Offset existing indices by 4
    for idx, code in enumerate(counter):
        vocab[code] = idx + 4
    return vocab

print(f"Loaded {len(result)} patient admissions from MIMIC-III")
print(f"\nTotal merged HADM IDs: {len(result)}")

# Prepare sequences as (encoder_input, decoder_target)
def build_encoder_decoder_sequences(merged_dict, vocab, max_len=128):
    examples = []
    for hadm_id, data in merged_dict.items():
        token_list = []

        # Merge all categories chronologically (flat event sequence)
        for cat in ['chart_items', 'input_items', 'lab_items',
                    'microbiology_items', 'prescriptions_items', 'procedure_items']:
            token_list.extend(str(code) for code in data[cat])

        # basic cleanup
        if len(token_list) < 3:
            continue

        print(f"HADM_ID {hadm_id}: Total merged events = {len(token_list)}")

        # Map to vocab indexes
        token_ids = [vocab.get(code, vocab['<UNK>']) for code in token_list]

        # Truncate to max_len - 2 for BOS/EOS
        token_ids = token_ids[:max_len - 2]

        # Create encoder and decoder sequences
        encoder_input = token_ids[:-1]  # e.g., [A B C]
        decoder_target = token_ids[1:]  # e.g., [B C D]
        decoder_input = [vocab['<BOS>']] + decoder_target  # [<BOS> B C]
        decoder_output = decoder_target + [vocab['<EOS>']]  # [B C D <EOS>]

        examples.append((hadm_id, encoder_input, decoder_input, decoder_output))

    return examples

# build vocab
vocab = build_vocab(result)

# build encoder-decoder sequences
examples = build_encoder_decoder_sequences(result, vocab, max_len=128)

# reverse vocab for decoding indices back to tokens
idx2token = {idx: token for token, idx in vocab.items()}

# print first few examples for inspection
if examples:
    print("\n===== Sample Encoder-Decoder Sequences =====\n")
    for i, (hadm_id, enc, dec_in, dec_out) in enumerate(examples[:5]):
        print(f"[Example {i + 1}] HADM_ID: {hadm_id}")
        print("Encoder input:     ", enc)
        print("Decoder input:     ", dec_in)
        print("Decoder target:    ", dec_out)
        print("Decoder input (str): ", [idx2token[idx] for idx in dec_in])
        print("Decoder target (str):", [idx2token[idx] for idx in dec_out])
        print("-" * 60)
else:
    print("\nNo examples generated. Check data filtering or event length thresholds.")