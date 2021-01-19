import glob
import numpy as np
import argparse
import os
import pandas as pd

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='Checkbox Detection')
parser.add_argument('--ocr_folder', default='output/csvs/', type=str, help='folder path to ocr results')
parser.add_argument('--checkbox_folder', default='CBoutput/', type=str, help='folder path to checkbox results')
parser.add_argument('--pred_cb', default=True, type=str2bool, help='Predict checkboxes')
parser.add_argument('--pred_table', default=True, type=str2bool, help='Predict table')
parser.add_argument('--output_folder', default='output/ocr_cb', type=str, help='folder path to results')

args = parser.parse_args()

def closest_node_to_right(node, nodes):
    tr = (np.transpose(nodes)[0])
    val_nodes = tr < node[0]
    nodes[val_nodes] = 0 
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist_2)

def preprocess(df):
    x1 = df[['startX', 'endX']].values
    x3 = df[['startY', 'endY']].values
    x4 = df[['startX', 'endY']].values
    x2 = df[['endX', 'startY']].values
    x1 = np.expand_dims(x1, axis=1)
    x2 = np.expand_dims(x2, axis=1)
    x3 = np.expand_dims(x3, axis=1)
    x4 = np.expand_dims(x4, axis=1)
    me = np.hstack((x1,x2,x3,x4))
    print(me.shape)
    x = df[['startX', 'endX']]
    y = df[['startY', 'endY']]
    x_mean = x.mean(axis=1)
    y_mean = y.mean(axis=1)
    df['x_mean'] = x_mean
    df['y_mean'] = y_mean
    x_m = x_mean.values
    y_m = y_mean.values
    me = np.stack((x_m,y_m))
    mean = np.transpose(me)
    return mean

if __name__ == "__main__":
    if not os.path.isdir(args.output_folder):
        os.mkdir(args.output_folder)
    cb_files = glob.glob(args.checkbox_folder+'*_cb.csv')
    for f in cb_files:
        print(f)
        filename = f.split('/')[-1]
        csv = args.ocr_folder + filename[:-7] + '.csv'
        df_cb = pd.read_csv(f)
        df = pd.read_csv(csv)
        df.dropna(inplace=True)
        mean_cb = preprocess(df_cb)
        mean = preprocess(df)
        
        for i,node in enumerate(mean_cb):
            # print(df_cb.iloc[[i]])
            argmin = closest_node_to_right(node, mean)
            break
            # print(df.iloc[[argmin]])
        break