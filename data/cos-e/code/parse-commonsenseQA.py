import jsonlines
import sys
import pandas as pd

expl = {}
with open(sys.argv[2], 'rb') as f:
    for item in jsonlines.Reader(f):
        expl[item['id']] = item['explanation']['open-ended']

print(f"reading {sys.argv[2]} complete")

with open(sys.argv[1], 'rb') as f:
    # wfw = csv.writer(wf,delimiter=',',quotechar='"')
    # wfw = csv.writer(wf,delimiter=',',quotechar='"')

    datalist = []
    non_existence_count = 0
    exception_count = 0
    for item in jsonlines.Reader(f):
        if item['id'] not in expl:
            non_existence_count +=1
            continue
        label = -1
        if(item['answerKey'] == 'A'):
            label = 0
        elif(item['answerKey'] == 'B'):
            label = 1
        elif(item['answerKey'] == 'C'):
            label = 2
        elif(item['answerKey'] == 'D'):
            label = 3
        else:
            label = 4
        try:
            datalist.append([item['id'],item['question']['stem'],item['question']['choices'][0]['text'],item['question']['choices'][1]['text'],item['question']['choices'][2]['text'],item['question']['choices'][3]['text'],item['question']['choices'][4]['text'],label,expl[item['id']]])
        except:
            exception_count += 1
    wfw = pd.DataFrame(data=datalist, columns=['id','question','choice_0','choice_1','choice_2','choice_3','choice_4','label','human_expl_open-ended'])
    wfw.to_csv(sys.argv[3], index=False)
    
        
print(f"non-existence-count {non_existence_count}")
print(f"exceptions {exception_count}")