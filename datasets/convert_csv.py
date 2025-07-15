import csv
import random

def tsv_to_csv_processed_sample(input_filepath, output_filepath, sample_size, seed):
    """
    Reads a tab-separated values (TSV) file, randomly picks a sample of lines,
    extracts only English and German columns, adds a 'label' column with 0,
    and converts them to a comma-separated values (CSV) file with a header.

    Args:
        input_filepath (str): The path to the input TSV file.
        output_filepath (str): The path where the output CSV file will be saved.
        sample_size (int): The number of random instances to pick.
        seed (int): The seed for the random number generator to ensure reproducibility.
    """
    try:
        with open(input_filepath, 'r', encoding='utf-8') as infile:
            reader = csv.reader(infile, delimiter='\t')
            all_rows = []
            for row in reader:
                if len(row) >= 2:
                    all_rows.append([row[0], row[1]]) 

        random.seed(seed)
        actual_sample_size = min(sample_size, len(all_rows))
        sampled_rows = random.sample(all_rows, actual_sample_size)

        with open(output_filepath, 'w', encoding='utf-8', newline='') as outfile:
            writer = csv.writer(outfile, delimiter=',')
            writer.writerow(['english', 'german', 'label']) 
            for row in sampled_rows:
                writer.writerow([row[0], row[1], '0']) 

        print(f"Successfully processed and converted a random sample of {actual_sample_size} instances from '{input_filepath}' to '{output_filepath}' with 'english,german,label' structure.")
    except FileNotFoundError:
        print(f"Error: Input file '{input_filepath}' not found. Please check the file path.")
    except Exception as e:
        print(f"An error occurred: {e}")

input_file = 'deu.txt' 
output_file = 'deu_processed_sampled.csv' 
sample_count = 500
random_seed = 10

tsv_to_csv_processed_sample(input_file, output_file, sample_count, random_seed)