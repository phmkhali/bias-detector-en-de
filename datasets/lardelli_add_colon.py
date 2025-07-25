import csv

input_file = 'lardelli_synthetically_extended.csv'
output_file = 'lardelli_final.csv'

with open(input_file, newline='', encoding='utf-8') as f_in, \
     open(output_file, 'w', newline='', encoding='utf-8') as f_out:

    reader = csv.DictReader(f_in)
    fieldnames = reader.fieldnames
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        writer.writerow(row)  # write original row
        german_text = row['german']
        if '*' in german_text:
            new_row = row.copy()
            new_row['german'] = german_text.replace('*', ':')
            writer.writerow(new_row)
