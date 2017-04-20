# -*- coding: utf-8 -*-
import json
from bleu_eval_2 import *

def compare(can,ref):
    score = []
    can = can.strip(".")
    for i in range(len(ref)):
        ref[i] = ref[i].strip(".")
        score.append(BLEU_2(can,ref[i]))
    return score

##load json    
with open('testing_public_label.json','r') as f:
    data = json.load(f)

##generate label
data2 = []
for i in range(len(data)):
    a = [str(data[i]['id'])]
    b = data[i]['caption']
    for j  in range(len(b)):
        b[j] = str(b[j])
    data2.append([a,b])

##output data    // not load yet
our_output = "a b c d e"
out_data = []
for i in range(len(data)):
    a = [i]
    b = [our_output]
    out_data.append([a,b])

### start BLEU
#store in :
#score_list
#max_score_index 
#max_score_sentence 
score_list = []
max_score_index  = []
max_score_sentence = []
for i in range(len(data)):
    score = compare(out_data[i][1][0],data2[i][1])
    score_list.append(score)
    tmp = score.index(max(score))
    max_score_index.append(tmp)
    max_score_sentence.append(data2[i][1][tmp])


    

