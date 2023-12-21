from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()
metadata_path = os.getenv('METADATA_PATH')
metadata = pd.read_csv(metadata_path, sep='|', header=None, names=['ID', 'Text', 'Normalized Text', 'Source'])
print(metadata_path)
print(metadata.head())
