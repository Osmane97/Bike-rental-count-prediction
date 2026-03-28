"""
- Reads train/eval/holdout CSVs from data/raw/.
- handling duplicates and null values.
- Saves cleaned splits to data/processed/.
"""


from pathlib import Path
import pandas as pd



BASE_DIR = Path("data")# the path of project's data folder in the directory project
INPUT_DIR = BASE_DIR  / 'data_splitted'
OUTPUT_DIR =  BASE_DIR  /'data_processed'

def preprocess_data (
        input_dir: Path = INPUT_DIR,
        output_dir: Path = OUTPUT_DIR 
):

     """
    Load train/valid/holdout datasets from a folder,
    clean them, and save cleaned versions to output_dir.
    """
     
     input_dir = Path(input_dir)
     output_dir = Path(output_dir)
     output_dir.mkdir(parents= True, exist_ok= True) # create  prents folder if needed, if exist then it doesnt crash

    #expected file names
     file_map = {"train": input_dir / "train_df.csv",
                 "valid": input_dir / "valid_df.csv",
                 "holdout": input_dir / "holdout_df.csv"
                 }
     
     datasets = {}

     #Load datasets
     for name, path in file_map.items():
          if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
          datasets[name] = pd.read_csv(path)
    

     rename_map = {'weathersit':'weather_situation',
                              'workingday': 'working_day',
                              'windspeed':'wind_speed',
                              'weekday': 'week_day',
                              'cnt': 'num_rentals', 
                              'temp':'temp_norm',
                              'atemp': 'feels_like_temp_norm',
                              'hum': 'humidity_norm',
                              'mnth': 'month',
                              'yr': 'year',
                              'dteday': 'date',
                              'hr': 'hour',
                              }
     

    #cleaning
     for name, df in datasets.items():

        # Standardize column names
        df.columns = (df.columns.
                      str.strip().
                      str.lower().
                      str.replace(' ', '_'))
        
        # Rename    
        df.rename(columns= rename_map, inplace= True)

        # Convert date
        df['date'] = pd.to_datetime(df['date'])
        # Drop column
        if "instant" in df.columns:
            df.drop(columns= 'instant', inplace= True)

        # Remove duplicates
        df.drop_duplicates(inplace=True)

        # Remove nulls
        df.dropna(inplace=True)

        # Logical integrity check

        imposible_mask = df['num_rentals'] < (df['casual'] + df['registered'])
        if imposible_mask.any():
            print(f"{name} has logical inconsistencies:")
            print(df.loc[imposible_mask,
                            ["date", "num_rentals", "casual", "registered"]])  
            df.drop(index= df[imposible_mask].index, inplace = True )
            


        #save Data
        df.to_csv(output_dir / f"{name}_preprocessed.csv", index=False)

    
if __name__  == "__main__":
    preprocess_data()     
