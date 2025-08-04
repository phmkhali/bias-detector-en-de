import csv

"""
This script reads a CSV file with columns: english, german, and label.
It counts the total number of entries in the file.
It checks that each label is either '0' or '1'.
It counts how many labels are '0' and how many are '1'.
It prints any lines where the label is invalid.
You can run the script on multiple files to get these statistics for each file.
"""

def is_valid_label(label):
    return label.strip() in {'0', '1'}

def check_csv(input_file):
    total = 0
    label_counts = {'0': 0, '1': 0}
    invalid_label_lines = []

    with open(input_file, newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        for i, row in enumerate(reader, start=2):  # start=2 for header + first line offset
            total += 1
            label = row.get('label', '').strip()
            
            if not is_valid_label(label):
                invalid_label_lines.append(i)
            else:
                label_counts[label] += 1

    print(f'Total entries: {total}')
    print(f"Label '0' count: {label_counts['0']}")
    print(f"Label '1' count: {label_counts['1']}")

    if invalid_label_lines:
        print(f'Invalid labels found on lines: {invalid_label_lines}')
    else:
        print('No invalid labels found.')

if __name__ == "__main__":
    for filename in ['lardelli_final.csv', 'mgente_final.csv', 'tatoeba_final.csv','dataset.csv']:
        print(f'Checking file: {filename}')
        check_csv(filename)
        print()

