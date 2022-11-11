import torch,json
import torch.nn as nn
import torch.nn.functional as f



with open('/Users/yililuo/Desktop/AI_Proj/data/test.json', encoding = 'utf8') as f:
    data = f.readlines()
    label = []
    for i in data:
        i = eval(json.loads(i).get('doc_label')[0])
        label.append(i)  

with open('/Users/yililuo/Desktop/AI_Proj/data/output/predict.txt') as f:
    data = f.readlines()
    input = []
    for i in data:
        input.append(eval(i.strip('\n')))
    
def compute_mes(input,label):
    label = torch.tensor(label)
    input = torch.tensor(input)
    loss = f.mse_loss(input,label).data
    return loss

compute_mes(input,label)