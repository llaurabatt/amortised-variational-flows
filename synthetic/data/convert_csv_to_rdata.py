import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import os

# Enable the automatic conversion between pandas DataFrames and R dataframes
pandas2ri.activate()

# Define file paths
csv_file_path = 'synthetic_data_SEED3_50obs_50groups.csv'
rdata_file_path = 'synthetic_data_SEED3_50obs_50groups.RData'

def read_csv_and_save_to_rdata(csv_file, rdata_file):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file, dtype='float64')

    # Convert pandas DataFrame to R DataFrame
    r_dataframe = pandas2ri.py2rpy(df)

    # Assign the R dataframe to a variable in the R environment
    ro.globalenv['dataframe_in_r'] = r_dataframe

    # Save the R DataFrame to an .RData file
    r_save = ro.r['save']
    r_save('dataframe_in_r', file=rdata_file)

    # Check if the file was saved
    if os.path.exists(rdata_file):
        print(f"Data has been successfully saved to {rdata_file}")
    else:
        print("There was an issue saving the file.")

def load_rdata_and_verify(rdata_file):
    # Load the .RData file
    r_load = ro.r['load']
    r_load(rdata_file)

    # Access the dataframe object in R environment (assuming it's still named 'dataframe_in_r')
    r_dataframe = ro.globalenv['dataframe_in_r']

    # Convert the R dataframe back to a pandas dataframe
    df_loaded = pandas2ri.rpy2py(r_dataframe)

    # Print a summary or some values to verify
    print("Loaded DataFrame from .RData:")
    print(df_loaded.head())  # Display the first few rows of the dataframe

# Execute the functions
read_csv_and_save_to_rdata(csv_file_path, rdata_file_path)
load_rdata_and_verify(rdata_file_path)