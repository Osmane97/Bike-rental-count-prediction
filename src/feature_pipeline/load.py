"""
Load and split the raw dataset
Store it in a file
"""

from pathlib import Path
import pandas as pd

BASE_DIR = Path("data")
RAW_DIR = BASE_DIR  / 'original_data'
DATA_DIR =  BASE_DIR  /'data_splitted'

def load_and_split_data(
        raw_path: Path = RAW_DIR / "hour.csv" , #Loading the raw dataset, either by specifying the path or by default
        output_path: Path | str = DATA_DIR #The folder where you want to save the split files
):
    
    """Load raw dataset, split into train/eval/holdout by date, and save to output_dir."""
    df = pd.read_csv(raw_path)

    df['dteday'] = pd.to_datetime(df['dteday'])
    df = df.sort_values(by = ['dteday', 'hr'])

    #cutoffs
    cutoff_date = pd.Timestamp('2012-01-01')
    holdout_date =  pd.Timestamp('2012-10-01')

    # Split
    train_df = df[df['dteday']< cutoff_date]
    valid_df = df[(df['dteday']>= cutoff_date) & (df['dteday']< holdout_date)]
    holdout_df = df[df['dteday']>= holdout_date]

    #Save
    outdir = Path(output_path)
    outdir.mkdir(parents= True, exist_ok= True) # create  prents folder if needed, if exist then it doesnt crash
    
    train_df.to_csv(outdir/ 'train_df.csv', index = False)
    valid_df.to_csv(outdir/ 'valid_df.csv', index = False)
    holdout_df.to_csv(outdir/ 'holdout_df.csv', index = False)

    return train_df, valid_df, holdout_df

if __name__ == '__main__':
    load_and_split_data()

