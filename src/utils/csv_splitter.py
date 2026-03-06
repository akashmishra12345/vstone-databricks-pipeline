#!/usr/bin/env python3
import csv
import os
from pathlib import Path

def detect_header(file_path, delimiter=','):
    """Detect if CSV file has a header row using Sniffer."""
    with open(file_path, 'r', encoding='utf-8') as file:
        sample = file.read(1024)
        file.seek(0)
        sniffer = csv.Sniffer()
        try:
            return sniffer.has_header(sample)
        except:
            return False

def count_rows(file_path, has_header=False, delimiter=','):
    """Count total data rows in CSV file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=delimiter)
        if has_header:
            next(reader)
        return sum(1 for _ in reader)

def split_csv(input_file, percentages, output_dir=None, has_header=None, delimiter=','):
    """
    Main function called by Databricks Notebook.
    Splits CSV into 4 chunks based on percentages.
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    output_dir = Path(output_dir) if output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect header
    if has_header is None:
        has_header = detect_header(input_file, delimiter)
        print(f"INFO: Auto-detected header: {has_header}")

    total_rows = count_rows(input_file, has_header, delimiter)
    print(f"INFO: Total data rows: {total_rows}")

    # Calculate exact row counts for each chunk
    chunk_sizes = []
    for i, pct in enumerate(percentages):
        if i == len(percentages) - 1:
            chunk_sizes.append(total_rows - sum(chunk_sizes))
        else:
            chunk_sizes.append(int(total_rows * pct / 100))

    # Read original header
    header_row = None
    if has_header:
        with open(input_file, 'r', encoding='utf-8') as f:
            header_row = next(csv.reader(f, delimiter=delimiter))

    # Splitting Logic
    with open(input_file, 'r', encoding='utf-8') as infile:
        reader = csv.reader(infile, delimiter=delimiter)
        if has_header: next(reader) # Skip source header

        for chunk_idx, size in enumerate(chunk_sizes):
            # Target filename format: 1_main_chunk_1.csv
            chunk_name = f"1_main_chunk_{chunk_idx + 1}.csv"
            chunk_path = output_dir / chunk_name
            
            print(f"INFO: Writing {size} rows to {chunk_name}...")
            
            with open(chunk_path, 'w', newline='', encoding='utf-8') as outfile:
                writer = csv.writer(outfile, delimiter=delimiter)
                if header_row:
                    writer.writerow(header_row)
                
                # Write only the calculated number of rows for this chunk
                for _ in range(size):
                    try:
                        writer.writerow(next(reader))
                    except StopIteration:
                        break

    print(f"SUCCESS: Created {len(chunk_sizes)} chunks in {output_dir}")
    return True