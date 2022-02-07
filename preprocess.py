import numpy as np
import pandas as pd
import random

input_directory = r"C:\I24 Motion Project\YB Outlier Detection\csv_data"

def load_data(filename):
    x_0, y_0 = process_data(filename, 0.0)
    x_10, y_10 = process_data(filename, 0.1)
    x_20, y_20 = process_data(filename, 0.2)
    x_30, y_30 = process_data(filename, 0.3)
    x_40, y_40 = process_data(filename, 0.4)
    
    x_train = np.concatenate((x_0,x_10,x_20,x_30,x_40))
    y_train = np.concatenate((y_0,y_10,y_20,y_30,y_40))
    
    return x_train, y_train


def process_data(filename, pollute_coefficient) :
    df = read_data(filename)

    x_train = []
    car_ids = np.unique(np.array(df['ID']))
    batch_size = 100

    for one_id in car_ids:
        one_car_df = df.loc[df['ID'] == one_id]
        one_car_df = one_car_df.sort_values(by='Frame #').reset_index(drop=True)

        start_row, end_row = 0, int(one_car_df.size / 3)

        while(end_row - start_row >= batch_size):
            df_batch = one_car_df.iloc[start_row : start_row + batch_size]
            batch_ar = np.array(df_batch['x'])
            
            if(pollute_coefficient > 0.0) :
                batch_ar = pollute_random(batch_ar, batch_size, pollute_coefficient)
                
            x_train.append(batch_ar)
            start_row += batch_size
        
    x_train = np.array(x_train)
    num_of_batch = int(x_train.size / batch_size)
    
    return x_train, np.full((num_of_batch), 1.0 - pollute_coefficient)


def pollute_random(batch_ar, batch_size, pollute_coefficient) :
    pollute_size = int(batch_size * pollute_coefficient)
    random_index = random.sample(range(0, batch_size), pollute_size)

    for i in range(len(random_index)) :
        batch_ar[random_index[i]] = -1

    return batch_ar
    
    
def read_data(filename):
    data_path = input_directory + "\\" + filename

    df = pd.read_csv(data_path, skiprows=0, index_col=False)
    df = df.reindex(columns=['Frame #', 'ID', 'x'])

    return df
