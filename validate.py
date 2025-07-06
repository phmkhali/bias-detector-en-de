import csv

def is_quoted(text):
    text = text.strip()
    return text.startswith('"') and text.endswith('"')

def is_valid_label(label):
    return label.strip() in {'0', '1'}

def check_csv(input_file):
    with open(input_file, newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        for i, row in enumerate(reader, start=2):  # start=2 for header + first line offset
            errors = []
            english = row.get('english', '')
            german = row.get('german', '')
            label = row.get('label', '')

            if not is_valid_label(label):
                errors.append(f'invalid label: {label}')

            if errors:
                print(f'Line {i}: {", ".join(errors)}')

if __name__ == "__main__":
    check_csv('datasets/dataset.csv')
