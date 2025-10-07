from datasets import load_dataset
import pandas as pd
import warnings
warnings.filterwarnings("ignore", message=".*promote_options.*")
warnings.filterwarnings("ignore", message=".*precompiled_charsmap.*")

TSV_FILE_PATH = "./data/Train_GCC-training_with_header.tsv"

def get_gcc3m(num_samples):
    try:
        df = pd.read_csv(TSV_FILE_PATH, sep='\t', encoding='utf-8')
        print(f"Successfully read TSV file: {TSV_FILE_PATH}")
        print(f"Total number of rows: {len(df)}")
        print(f"Column names: {list(df.columns)}")
        
        prompts = df['caption'].head(num_samples).tolist()
        print(f"Extracted {len(prompts)} prompts from GCC3M")
        
        return prompts
    
    except FileNotFoundError:
        print(f"Error: File {TSV_FILE_PATH} does not exist.")
        return []
    except Exception as e:
        print(f"Error occurred while reading the file: {str(e)}")
        return []

def get_loaders(name, num_samples=50):
    if name == 'gcc3m':
        return get_gcc3m(num_samples)  
    raise ValueError(f"Unknown dataset: {name}")

