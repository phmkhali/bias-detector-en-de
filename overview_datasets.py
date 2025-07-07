import pandas as pd

def analyze_dataset(file_path):
    """
    Loads a CSV file, prints the total number of entries,
    and counts the occurrences of 0 and 1 in the 'label' column.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"\n--- Analysis for: {file_path} ---")
        print(f"Total entries: {len(df)}")

        if 'label' in df.columns:
            label_counts = df['label'].value_counts().sort_index()
            print("Label counts:")
            if 0 in label_counts:
                print(f"  Label 0: {label_counts[0]} entries")
            else:
                print("  Label 0: 0 entries")
            if 1 in label_counts:
                print(f"  Label 1: {label_counts[1]} entries")
            else:
                print("  Label 1: 0 entries")
            # Also print any other labels if they exist
            for label in label_counts.index:
                if label not in [0, 1]:
                    print(f"  Label {label}: {label_counts[label]} entries (other labels)")
        else:
            print("No 'label' column found in this dataset.")
    except FileNotFoundError:
        print(f"\nError: File not found at {file_path}")
    except Exception as e:
        print(f"\nAn error occurred while processing {file_path}: {e}")

# List of files to analyze
files_to_analyze = [
    "datasets/lardelli_and_gpt_data.csv",
    "datasets/wikipedia_biographies_transformed.csv",
    "datasets/mgente_transformed.csv" 
]

# Run analysis for each file
for file in files_to_analyze:
    analyze_dataset(file)
