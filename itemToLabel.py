import csv
from collections import defaultdict
import pandas as pd

def create_itemid_label_mapping(csv_file_path):
    # define the allowed categories for LINKSTO
    allowed_categories = {
        'chartevents',
        'inputevents',
        'labevents',
        'microbiologyevents',
        'prescriptions',
        'procedureevents'
    }

    # initialize a defaultdict for category mappings
    category_mappings = defaultdict(dict)
    # initialize a dictionary to track ITEMIDs and their details
    itemid_tracker = {}
    # list to store duplicates
    duplicates = []

    # read the CSV file
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            #  reads CSV rows into dictionaries keyed by column headers
            reader = csv.DictReader(file)

            # process each row
            for row in reader:
                itemid = row['ITEMID']
                label = row['LABEL']
                category = row['LINKSTO'].lower() if row['LINKSTO'] else 'Uncategorized'

                # only check if category is part of the allowed categories
                if category not in allowed_categories:
                    continue

                # check for duplicate ITEMID
                if itemid in itemid_tracker:
                    # store duplicate info
                    duplicates.append({
                        'ITEMID': itemid,
                        'First_Label': itemid_tracker[itemid]['label'],
                        'First_Category': itemid_tracker[itemid]['category'],
                        'Second_Label': label,
                        'Second_Category': category
                    })
                else:
                    # store ITEMID details
                    itemid_tracker[itemid] = {'label': label, 'category': category}

                # add itemid-label pair to the corresponding category
                category_mappings[category][itemid] = label

    # if the file doesn't exist print an error and return empty results
    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found. Please check the file path.")
        return {}, []
    except Exception as e:
        print(f"Error reading file: {e}")
        return {}, []

    return category_mappings, duplicates

# get labels for lab item IDs
def load_labitems_labels(csv_file_path):
    # empt dict to store lab item labels
    labitem_labels = {}
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                itemid = row['ITEMID']
                label = row['LABEL']
                # store label keyed by ITEMID in dictionary
                labitem_labels[itemid] = label
    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found. Please check the file path.")
    except Exception as e:
        print(f"Error reading file: {e}")
    return labitem_labels


def main():
    # path for the two files
    d_items_path = '/Users/amandali/Downloads/Mimic III/D_ITEMS.csv'
    d_labitems_path = '/Users/amandali/Downloads/Mimic III/D_LABITEMS.csv'

    # load mappings from main items file
    mappings, duplicates = create_itemid_label_mapping(d_items_path)
    # load lab item labels from lab items file
    lab_items_labels = load_labitems_labels(d_labitems_path)

    # incorporate lab item labels, possibly overriding or adding to existing labels:
    if 'labevents' not in mappings:
        mappings['labevents'] = {}
    for itemid, label in lab_items_labels.items():
        # add or update label from lab item file for lab events
        mappings['labevents'][itemid] = label

    # dictionary to convert to pandas dataframe later
    mapping_rows = []
    for category, items in mappings.items():
        for itemid, label in items.items():
            # add item and labels to dictionary
            mapping_rows.append({
                'Category': category,
                'ItemID': itemid,
                'Label': label
            })

    # dataframe mapping
    df_mappings = pd.DataFrame(mapping_rows)
    # dataframe of duplicates (if any found)
    df_duplicates = pd.DataFrame(duplicates)

    # store mapping into excel sheet 
    output_file = 'filtered_item_mappings.xlsx'
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_mappings.to_excel(writer, sheet_name='Mappings', index=False)
        if not df_duplicates.empty:
            df_duplicates.to_excel(writer, sheet_name='Duplicates', index=False)

    print(f"Data successfully saved to '{output_file}'.")

if __name__ == "__main__":
    main()
