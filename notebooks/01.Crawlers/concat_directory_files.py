import pandas as pd
import os
from tqdm import tqdm
import time

start = time.time()
file_dict = 'news-title-bias\data\df_stories_raw_urls_contents_20_proxy'
# Get a list of the file names in the directory
file_list = os.listdir(file_dict)

# Initialize an empty list to store the dataframes
df_list = []

# Iterate through the file names
for file in tqdm(file_list):
    # Read the CSV file into a dataframe
    file = os.path.join(file_dict, file)
    df = pd.read_csv(file, encoding='utf_8_sig', header=0)
    # Add the dataframe to the list
    df_list.append(df)

# Concatenate the dataframes into a single dataframe
df_concat = pd.concat(df_list)

# Write the concatenated dataframe to a CSV file
df_concat.to_csv(file_dict + '.csv', index=False)
print('Time overall: %s'%(time.time()-start))