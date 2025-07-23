import csv
import random

def tsv_to_csv_processed_sample(input_filepath, output_filepath, sample_size, seed):
    """
    Reads a TSV file, filters out rows with 'Tom', samples a subset,
    and writes them as CSV with 'english,german,label' format.
    """
    try:
        with open(input_filepath, 'r', encoding='utf-8') as infile:
            reader = csv.reader(infile, delimiter='\t')
            all_rows = []
            for row in reader:
                if len(row) >= 2:
                    english = row[0].strip()
                    german = row[1].strip()
                    if "Tom" not in english and "Tom" not in german:
                        all_rows.append([english, german])

        random.seed(seed)
        actual_sample_size = min(sample_size, len(all_rows))
        sampled_rows = random.sample(all_rows, actual_sample_size)

        with open(output_filepath, 'w', encoding='utf-8', newline='') as outfile:
            writer = csv.writer(outfile, delimiter=',')
            writer.writerow(['english', 'german', 'label'])
            for row in sampled_rows:
                writer.writerow([row[0], row[1], '0'])

        print(f"Sampled {actual_sample_size} rows (excluding 'Tom') and saved to '{output_filepath}'.")
    except FileNotFoundError:
        print(f"Error: Input file '{input_filepath}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Usage
input_file = 'deu.txt'
output_file = 'deu_processed_500.csv'
sample_count = 550
random_seed = 10

tsv_to_csv_processed_sample(input_file, output_file, sample_count, random_seed)
