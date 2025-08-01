import csv
from collections import defaultdict

def create_itemid_label_mapping(csv_file_path):
    # initialize a defaultdict for category mappings
    category_mappings = defaultdict(dict)
    # initialize a dictionary to track ITEMIDs and their details
    itemid_tracker = {}
    # list to store duplicates
    duplicates = []

    # read the CSV file
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)

            # process each row
            for row in reader:
                itemid = row['ITEMID']
                label = row['LABEL']
                category = row['CATEGORY'] if row['CATEGORY'] else 'Uncategorized'

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

    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found. Please check the file path.")
        return {}, []
    except Exception as e:
        print(f"Error reading file: {e}")
        return {}, []

    return category_mappings, duplicates

def main():
    csv_file_path = '/Users/amandali/Downloads/Mimic III/D_ITEMS.csv'
    mappings, duplicates = create_itemid_label_mapping(csv_file_path)

    # check and print duplicates
    if duplicates:
        print("\nDuplicate ITEMIDs found:")
        for dup in duplicates:
            print(f"  ITEMID: {dup['ITEMID']}")
            print(f"    First occurrence: Label='{dup['First_Label']}', Category='{dup['First_Category']}'")
            print(f"    Second occurrence: Label='{dup['Second_Label']}', Category='{dup['Second_Category']}'")
    else:
        print("\nNo duplicate ITEMIDs found.")

    # print the mappings for each category
    for category, items in mappings.items():
        print(f"\nCategory: {category}")
        for itemid, label in items.items():
            print(f"  ItemID: {itemid} -> Label: {label}")

if __name__ == "__main__":
    main()
