import csv
import xml.etree.ElementTree as ET
import os
import pandas as pd

def convert(input_data, xml_file):
    """
    Modular function to convert CSV file or Pandas DataFrame to XML.
    """
    try:
        # Check if input is a file path (string) or a DataFrame
        if isinstance(input_data, str):
            if not os.path.exists(input_data):
                print(f"ERROR: Source file not found: {input_data}")
                return False
            with open(input_data, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                data = list(reader)
        elif isinstance(input_data, pd.DataFrame):
            data = input_data.to_dict(orient='records')
        else:
            print("ERROR: Invalid input type for XML conversion.")
            return False

        if not data:
            print("INFO: No data available for XML conversion.")
            return False

        # Create XML structure
        root = ET.Element("car_market_data")
        
        for row in data:
            record = ET.SubElement(root, "listing")
            for key, value in row.items():
                # Clean column names for XML tags (No spaces/special chars)
                clean_key = str(key).replace(" ", "_").replace("(", "").replace(")", "")
                elem = ET.SubElement(record, clean_key)
                elem.text = str(value)
                
        # Pretty print and write to file
        ET.indent(root, space="  ")
        tree = ET.ElementTree(root)
        tree.write(xml_file, encoding='utf-8', xml_declaration=True)
        
        print(f"SUCCESS: Converted data to {xml_file}")
        return True
        
    except Exception as e:
        print(f"ERROR in csv_to_xml utility: {e}")
        raise e

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        convert(sys.argv[1], sys.argv[2])