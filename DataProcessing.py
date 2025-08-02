import pandas as pd
import os
from collections import Counter

# mimic data directory (stored locally)
mimic_data_dir = '/Users/amandali/Downloads/Mimic III'

# mimic_3data is the directory path for the data in my local file
def load_mimic3_data(mimic_3data, nrows):
    # chart events: observations and measurements
    # input events: administration of fluids, medications, or other substances
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
        # at least one item in the lists
        if any(len(items) > 0 for items in current_dict.values()):
            merged_dict[hadm_id] = current_dict

    return merged_dict

# loads the dict
result = load_mimic3_data(mimic_data_dir, nrows=1000000)

# build vocab from merged_dict
def build_vocab(merged_dict):
    all_codes = []
    for hadm_data in merged_dict.values():
        for cat in ['chart_items', 'input_items', 'lab_items',
                    'microbiology_items', 'prescriptions_items', 'procedure_items']:
            # adds string version of all codes from that category into all_codes
            all_codes.extend(str(code) for code in hadm_data[cat])

    # counts occurrence of each unique code
    counter = Counter(all_codes)
    vocab = {
        "<PAD>": 0, # padding
        "<UNK>": 1, # unknown token
        "<BOS>": 2, # beginning of sentence
        "<EOS>": 3  # end of sentence
    }
    # offset existing indices by 4
    for idx, code in enumerate(counter):
        # 0-3 are reserved for special tokens; start index at 4
        vocab[code] = idx + 4
    return vocab

# show how many hadm_ids were merged/logged
print(f"\nTotal HADM IDs: {len(result)}")

# Prepare sequences as (encoder_input, decoder_target)
def build_encoder_decoder_sequences(merged_dict, vocab):
    encoded_sequences = []
    # iterates over each hadm_id and its data
    for hadm_id, data in merged_dict.items():
        token_list = []

        # turns the items from each category into strings
        for cat in ['chart_items', 'input_items', 'lab_items',
                    'microbiology_items', 'prescriptions_items', 'procedure_items']:
            token_list.extend(str(code) for code in data[cat])

        # has to have at least three tokens - less than 3 is too short to work with
        if len(token_list) < 3:
            continue

        print(f"HADM_ID {hadm_id}: Total merged events = {len(token_list)}")

        # converts each event code to its integer vocab index; if code isn't in vocab, use index for 'UNK'
        token_ids = [vocab.get(code, vocab['<UNK>']) for code in token_list]

        # Create encoder and decoder sequences
        encoder_input = token_ids[:-1]  # e.g., [A B C] leave last to be predicted
        decoder_target = token_ids[1:]  # e.g., [B C D] gives the last one that should be predicted
        decoder_input = [vocab['<BOS>']] + decoder_target  # [<BOS> B C] signals start index for decoding
        decoder_output = decoder_target + [vocab['<EOS>']]  # [B C D <EOS>] signals where decoded sequence should end

        encoded_sequences.append((hadm_id, encoder_input, decoder_input, decoder_output))

    return encoded_sequences

# build vocab from result data dict
vocab = build_vocab(result)

# build encoder-decoder sequences, use all tokens
encoded_sequences = build_encoder_decoder_sequences(result, vocab)

# reverse vocab for decoding indices back to tokens
idx2token = {idx: token for token, idx in vocab.items()}

# check if any encoded sequences were generated
if encoded_sequences:
    print("\n===== Sample Encoder-Decoder Sequences =====\n")
    # loops over how many tuples
    for i, (hadm_id, enc, dec_in, dec_out) in enumerate(encoded_sequences[:10]):
        # print out information for verification
        print(f"[HADM_ID {i + 1}] HADM_ID: {hadm_id}")
        print("Encoder input:     ", enc)
        print("Decoder input:     ", dec_in)
        print("Decoder target:    ", dec_out)
        print("Decoder input (str): ", [idx2token[idx] for idx in dec_in])
        print("Decoder target (str):", [idx2token[idx] for idx in dec_out])
        print("-" * 60)
else:
    print("\nNo examples generated. Check data filtering or event length thresholds.")

# convert encoded_sequences to a DataFrame
def save_to_excel(encoded_sequences, output_file='encoded_sequences.xlsx'):
    # prepare data for DataFrame
    data = []
    for hadm_id, enc, dec_in, dec_out in encoded_sequences:
        # convert lists to strings for Excel compatibility
        enc_str = ','.join(map(str, enc))
        dec_in_str = ','.join(map(str, dec_in))
        dec_out_str = ','.join(map(str, dec_out))
        data.append({
            'HADM_ID': hadm_id,
            'Encoder_Input': enc_str,
            'Decoder_Input': dec_in_str,
            'Decoder_Output': dec_out_str
        })

    # create DataFrame
    df = pd.DataFrame(data)

    # save to Excel
    df.to_excel(output_file, index=False, engine='openpyxl')
    print(f"Saved encoded sequences to {output_file}")

# call the function to save the sequences
if encoded_sequences:
    save_to_excel(encoded_sequences, output_file='encoded_sequences.xlsx')
else:
    print("No sequences to save to Excel.")
