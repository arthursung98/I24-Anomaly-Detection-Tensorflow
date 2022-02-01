import pathlib
import numpy as np
import pandas as pd


def load_data(filename):
  df = read_data(filename)

  x_train = []
  car_ids = np.unique(np.array(df['ID']))

  for one_id in car_ids:
    one_car_df = df.loc[df['ID'] == one_id]
    one_car_df = one_car_df.sort_values(
        by='Frame #').reset_index(drop=True)

    start_frame, max_frame = 0, int(one_car_df.size / 3)

    while(max_frame - start_frame >= 50):
      df_50batch = one_car_df.iloc[start_frame: start_frame + 50]
      df_50batch_xcoord = np.array(df_50batch['x'])
      x_train.append(df_50batch_xcoord)
      start_frame += 50

  x_train = np.array(x_train)
  num_of_batch = int(x_train.size / 50)

  return x_train, np.full((num_of_batch), 1.0)


def read_data(filename):
  data_path = pathlib.Path().absolute().joinpath('../gt_data')
  file_name = data_path.joinpath(filename)

  df = pd.read_csv(file_name, skiprows=0, index_col=False)
  df = df.reindex(columns=['Frame #', 'ID', 'x'])

  return df
