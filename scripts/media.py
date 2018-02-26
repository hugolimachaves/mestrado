import numpy as np
import os
import argparse
import pandas as pd

encode = "utf-8"

parser = argparse.ArgumentParser()
parser.add_argument("name", help="Video file name")

if(os.path.isfile(parser.parse_args().name)):
	df = pd.read_csv(parser.parse_args().name, header=None)
	df.replace([np.inf, -np.inf], np.nan)
	norm = np.linalg.norm(df.values, ord=2, axis=1)
	mean = norm.mean()
	norm = np.append(norm, mean)
	new_df = pd.DataFrame()
	new_df = new_df.assign(norm=norm)
	out_file = parser.parse_args().name + "_" + str(df.isnull().values.any()) + ".stats"
	new_df.to_csv(out_file, sep=',', line_terminator='\n', encoding=encode, index=False)
