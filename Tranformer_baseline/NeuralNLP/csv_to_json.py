# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 23:51:35 2021

@author: 76504
"""
# convert training data into json format {'doc_label':[], 'doc_token':[]}

import pandas as pd
import json,re,tqdm,os



def read_csv(path,file_name, column):
    df = pd.read_csv(path+file_name,encoding='utf8')
    return df[column]


def csv_json(path,file_name,output_file):
    df = read_csv(path,file_name, column)
    with open(path+output_file,"w+",encoding='utf-8') as f:
        for indexs in tqdm.tqdm(range(len(df))):
            dict = {}
            if '|' in str(df.loc[indexs].values[0]):
                lab=str(df.loc[indexs].values[0]).split('|')
                dict['doc_label']=lab
            else:
                dict['doc_label'] = [str(df.loc[indexs].values[0])]
            doc_token = df.loc[indexs].values[1].rstrip()
            # remove \n \r \t
            reg = "[\n\r\t]"
            doc_token =re.sub(reg, '', doc_token)
            content = doc_token.split(' ')
            dict['doc_token'] = content
            dict['doc_keyword'] = []
            dict['doc_topic'] = []
            # convert into json
            json_str = json.dumps(dict, ensure_ascii=False)
            f.write('%s\n' % json_str)
            

if __name__ == '__main__':
    path = './'
    file_name = 'data/train_clean.csv'
    column = ['grammar','full_text']
    df = read_csv(path,file_name, column)
    output_file = 'data/train.json'
    csv_json(path,file_name,output_file)
 