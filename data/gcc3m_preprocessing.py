import pandas as pd
import warnings
import os


original_file_path = './data/Train_GCC-training.tsv'

new_file_path = './data/Train_GCC-training_with_header.tsv'

if not os.path.exists(new_file_path):
    try:
        print(f"Reading the original file {original_file_path}...")
        df = pd.read_csv(original_file_path, sep='\t', header=None)
        
        print(f"File read completed, {len(df)} rows.")

        df.columns = ['caption', 'url']
        print("Successfully added headers ['caption', 'url'] to the data.")

        print(f"Starting to write data to the new file {new_file_path} ...")
        df.to_csv(new_file_path, sep='\t', index=False, encoding='utf-8')
    except Exception as e:
        print(f"Error occurred during processing: {e}")
else:
    print(f"New file {new_file_path} already exists, skipping creation step.")

# --- Preparation ---
warnings.filterwarnings("ignore", message=".*promote_options.*")
warnings.filterwarnings("ignore", message=".*precompiled_charsmap.*")

TSV_FILE_PATH = new_file_path  

def get_gcc3m(num_samples):
    """
    Read the first num_samples prompts from the TSV file of GCC3M.
    """
    df = pd.read_csv(TSV_FILE_PATH, sep='\t', encoding='utf-8')
    print(f"Successfully read the TSV file: {TSV_FILE_PATH}")
    print(f"Total number of rows: {len(df)}")
    print(f"Column names: {list(df.columns)}")
    prompts = df['caption'].head(num_samples).tolist()
    print(f"Successfully extracted {len(prompts)} prompts from GCC3M")
        
    return prompts
    

def get_loaders(name, num_samples=50):
    if name == 'gcc3m':
        return get_gcc3m(num_samples)
    raise ValueError(f"Unknown dataset: {name}")

print("Python functions and configurations are ready.")

# Simple test
num_to_sample = 5
print(f"--- Trying to load the first {num_to_sample} samples ---")

prompts = get_loaders(name='gcc3m', num_samples=num_to_sample)

print("--- Demo output results ---")
if prompts:
    print(f"Successfully got {len(prompts)} prompts:")
    for i, p in enumerate(prompts, 1):
        print(f"{i}: {p}")
else:
    print("Failed to get prompts.")