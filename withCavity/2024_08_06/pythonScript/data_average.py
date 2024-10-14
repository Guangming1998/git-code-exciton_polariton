import pandas as pd
import glob
import os
import numpy as np
import shutil
import sys

# Define the path pattern to search for CSV files in subdirectories
path = r'tr*'  # Modify as needed

# Use glob to find all matching files
all_files = glob.glob(os.path.join(path, 'CJJ_cavity.csv'))

# List to store data from each file
data_frames = []
data_frames_1 = []
data_frames_2 = []

# Loop through each file and read the data
for file in all_files:
    # Read the CSV file as strings
    df = pd.read_csv(file, usecols=['CJJ_mol'], dtype=str)
    df_1 = pd.read_csv(file, usecols=['CJJ_cav'], dtype=str)
    df_2 = pd.read_csv(file, usecols=['CJJ'], dtype=str)
    

    # Convert the 'CJJ_mol', 'CJJ_cav', 'CJJ' column to complex numbers
    df['CJJ_mol'] = df['CJJ_mol'].apply(lambda x: complex(x.replace(' ', '').replace('+-', '-')))
    df_1['CJJ_cav'] = df_1['CJJ_cav'].apply(lambda x: complex(x.replace(' ', '').replace('+-', '-')))
    df_2['CJJ'] = df_2['CJJ'].apply(lambda x: complex(x.replace(' ', '').replace('+-', '-')))
    

    data_frames.append(df)
    data_frames_1.append(df_1)
    data_frames_2.append(df_2)

    # Remove the file after reading
    os.remove(file)

# Concatenate all dataframes along the columns (axis=1)
concatenated_df = pd.concat(data_frames, axis=1, ignore_index=True)
concatenated_df_1 = pd.concat(data_frames_1, axis=1, ignore_index=True)
concatenated_df_2 = pd.concat(data_frames_2, axis=1, ignore_index=True)

# Convert the concatenated dataframe to a numpy array
data_array = concatenated_df.to_numpy()
data_array_1 = concatenated_df_1.to_numpy()
data_array_2 = concatenated_df_2.to_numpy()

# Calculate the average across the columns (axis=1)
average_array = np.mean(data_array, axis=1)
average_array_cav = np.mean(data_array_1, axis=1)
average_array_total = np.mean(data_array_2, axis=1)


# Convert the average array back to a dataframe
average_df = pd.DataFrame(average_array, columns=['CJJ_mol_average'])
average_df_1 = pd.DataFrame(average_array_cav, columns=['CJJ_cav_average'])
average_df_2 = pd.DataFrame(average_array_total, columns=['CJJ_average'])

# Save the averaged data to a new CSV file
average_df.to_csv('CJJmol_average.csv', index=False, header=False)
average_df_1.to_csv('CJJcav_average.csv', index=False, header=False)
average_df_2.to_csv('CJJ_average.csv', index=False, header=False)

#delete subdirectory tr/
for i in range(1,data_array[0,:].size+1):
    #directory = sys.argv[1]
    path = os.path.join(os.getcwd(), f'tr{i}')
    # if os.path.exists(path):
    shutil.rmtree(path)
