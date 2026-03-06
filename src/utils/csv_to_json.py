import csv
import json
import os

def convert(csv_file, json_file):
    """
    Modular function to convert CSV to JSON.
    Can be called directly from Databricks notebooks.
    """
    try:
        # File existence check for better error handling
        if not os.path.exists(csv_file):
            print(f"ERROR: Source file not found: {csv_file}")
            return False

        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = list(reader)
            
        if not data:
            print(f"INFO: No data found in {csv_file}")
            return False
            
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            
        print(f"SUCCESS: Converted {csv_file} to {json_file}")
        return True
        
    except Exception as e:
        print(f"ERROR in csv_to_json utility: {e}")
        raise e

# Taaki aap ise terminal se bhi test kar sakein (if needed)
if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        convert(sys.argv[1], sys.argv[2])