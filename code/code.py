"""
Created on Fri May  1 23:15:12 2020

@author: rabia
"""
import os
import json
import pandas as pd
import plotly
import plotly.graph_objects as go
import spacy
from collections import Counter
import operator
nlp = spacy.load('en_core_web_sm')   

# path defined for all the corpa
essays_path =  'D:/UPB/SS2020/CA/Assignment/1/Data/ArgumentAnnotatedEssays-2.0/ArgumentAnnotatedEssays-2.0/brat-project-final/brat-project-final'
labels_path = 'D:/UPB/SS2020/CA/Assignment/1/Data/UKP-OpposingArgumentsInEssays_v1.0/UKP-OpposingArgumentsInEssays_v1.0/labels.tsv'
paragraphs_path = 'D:/UPB/SS2020/CA/Assignment/1/Data/UKP-InsufficientArguments_v1.0/UKP-InsufficientArguments_v1.0/data-tokenized.tsv'
train_test_path = 'D:/UPB/SS2020/CA/Assignment/1/Data/ArgumentAnnotatedEssays-2.0/ArgumentAnnotatedEssays-2.0/train-test-split.csv'
unified_json_path = 'unified_data.json'

class major_claim:
  def __init__(self, span, text):
    self.span = span
    self.text=' '.join(text) 
    
class claim:
  def __init__(self, span, text):
    self.span = span
    self.text=' '.join(text)
    
class premis:
  def __init__(self, span, text):
    self.span = span
    self.text=' '.join(text) 

class paragraph:
  def __init__(self, text,annotation):
    self.text = text 
    if(annotation=='sufficient'):
        self.annotation=True
    else:
        self.annotation=False  
    
class confirmation_bias:
  def __init__(self, docId, label):
    self.docId=docId
    if(label=='positive'):
        self.label=True
    else:
        self.label=False    
        
class argument:
  def __init__(self, docId, text, major_claim, claim, premis,confirmation_bias,paragraphs):
    self.docId = docId
    self.text = text 
    self.major_claim=major_claim
    self.claim=claim
    self.premis=premis
    self.confirmation_bias=confirmation_bias
    self.paragraphs=paragraphs
    
def getFileId(fileName):
    if '.txt' in fileName:
        fileName=fileName.replace('.txt', '')
    if '.ann' in fileName:
        fileName=fileName.replace('.ann', '')
    if 'essay00' in fileName:
        fileName=fileName.replace('essay00', '')
    if 'essay0' in fileName:
        fileName=fileName.replace('essay0', '')
    if 'essay' in fileName:
        fileName=fileName.replace('essay', '')
    return fileName 

# filter the train data
def filterTrainData():    
    print('Reading train-test-split.csv ...')
    df = pd.read_csv(train_test_path, delimiter=";")
    trainIds=df[df['SET'] == 'TRAIN']['ID']
    print('Done reading train-test-split.csv!')
    return trainIds

# method to check if the essay is part of train data
def isTrainData(data, element):
    element=element.replace('.ann','')
    for item in data:
        if item==element:
            return True
    return False

# method to read all the essays
def readEssays(trainFiles):
    print('Reading essays...')
    essays={}
    for root, dirs, files in os.walk(essays_path+'/'):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'r') as f:
                    text = f.read()
                    fileName=getFileId(file)
                    essays[fileName] = text
    print('Done reading essays!')                
    return essays  

# method to read confirmation bias label              
def readLabels():
    labels=[]
    print('Reading labels.tsv...')
    df = pd.read_csv(labels_path,sep='\t', header=0,engine='python')
    for index, row in df.iterrows():
        docId=getFileId(df['id'][index])
        labels.append(confirmation_bias(docId,df['label'][index])) 
    print('Done reading labels.tsv!')    
    return labels 
   
# method to read paragraphs    
def readParagraphs():
    print('Reading data-tokenized.tsv...')
    paragraphs={}
    df = pd.read_csv(paragraphs_path,sep='\t', header=0,engine='python')
    df = df.fillna('sufficient')  
    for index1 , row1 in df.iterrows():
        paras=[]
        essayId=df['ESSAY'][index1]
        same=df[df['ESSAY'] == essayId]
        for i, row in same.iterrows():
            paras.append(paragraph(row['TEXT'],row['ANNOTATION']))
        paragraphs[essayId]=paras  
    print('Done reading data-tokenized.tsv!')   
    return paragraphs 


# method to create argument object into json onject                   
def createJSONobj(arg):
    argJSON = {}
    premisObj = []
    majorClaimsObj = []
    claimsObj = []
    paragraphsObj = []

    for p in arg.premis:
        premisObj.append({'span': p.span, 'text': p.text})
    for p in arg.major_claim:
        majorClaimsObj.append({'span': p.span, 'text': p.text})
    for p in arg.claim:
        claimsObj.append({'span': p.span, 'text': p.text})
    for p in arg.paragraphs:
        paragraphsObj.append({'text': p.text, 'sufficient': p.annotation})    
        
    argJSON["id"] = arg.docId
    argJSON['text'] = arg.text    
    argJSON['major_claims'] = majorClaimsObj
    argJSON['claims'] = claimsObj
    argJSON['premises'] = premisObj
    argJSON['confirmation_bias'] = arg.confirmation_bias.label
    argJSON['paragraphs'] = paragraphsObj
    return argJSON

def calculateStatistics():
    data = pd.read_json(unified_json_path)
    df = pd.DataFrame(data)
    stats={}

    print('Stats Calculation Started....')
    mcCount = mcTokens = claimsCount = claimsTokens = premisCount = premisTokens = 0
    mcTokensSum = cTokensSum = premisTokensSum = 0
    mcRoots = []
    claimsRoots = []
    premisRoots = []
    mcDict = {}
    cDict = {}
    premisDict = {}
    
    stats['essay']=len(df['id'])
    stats['major_claim'] = df['major_claims'].str.len().sum()
    stats['claim'] = df['claims'].str.len().sum()
    stats['premis'] = df['premises'].str.len().sum()
    stats['paragraph'] = df['paragraphs'].str.len().sum()
    stats['essay_with_conformation'] = df[df.confirmation_bias==True]['confirmation_bias'].count()
    stats['essay_without_conformation'] = df[df.confirmation_bias==False]['confirmation_bias'].count()

    for index , row in df.iterrows():
        for mC in list(row['major_claims']):
            mcCount += 1
            doc = nlp(str(mC['text']))
            mcTokensSum += len(doc)
            mcList = list(filter(lambda d: d.dep_ == "ROOT" and not d.is_stop, doc))
            if mcList:
                if str(mcList[0]) in mcDict.keys():
                    mcDict[str(mcList[0])] = mcDict[str(mcList[0])] + 1
                else:
                    mcDict[str(mcList[0])] = 1   
        for claims in list(row['claims']):
            claimsCount += 1
            doc = nlp(str(claims['text']))
            cTokensSum += len(doc)
            claimList = list(filter(lambda d: d.dep_ == "ROOT" and not d.is_stop, doc))
            if claimList:
                if str(claimList[0]) in cDict.keys():
                    cDict[str(claimList[0])] = cDict[str(claimList[0])] + 1
                else:
                    cDict[str(claimList[0])] = 1            
        for premis in list(row['premises']):
            premisCount += 1
            doc = nlp(str(premis['text']))
            premisTokensSum += len(doc) 
            premisList = list(filter(lambda d: d.dep_ == "ROOT" and not d.is_stop, doc))
            if premisList:
                if str(premisList[0]) in premisDict.keys():
                    premisDict[str(premisList[0])] = premisDict[str(premisList[0])] + 1
                else:
                    premisDict[str(premisList[0])] = 1
    topMajorClaim=sorted(mcDict.items(), key=lambda x: x[1], reverse=True)[0:9]
    topClaim=sorted(cDict.items(), key=lambda x: x[1], reverse=True)[0:9]      
    topPremis=sorted(premisDict.items(), key=lambda x: x[1], reverse=True)[0:9]      
    stats['topMajorClaim']=[lis[0] for lis in topMajorClaim]
    stats['topClaim']=[lis[0] for lis in topClaim]                
    stats['topPremis']=[lis[0] for lis in topPremis]               
    stats['major_claim_avg'] =round(mcTokensSum/mcCount) 
    stats['claim_avg'] = round(cTokensSum/claimsCount)
    stats['premis_avg'] = round(premisTokensSum/premisCount)      
    
    sumTrue = sumFalse=0
    for index , row in df.iterrows():
        for p in df['paragraphs'][index]:
            if(p['sufficient']== True):
                sumTrue+=1
            if(p['sufficient']== False):
                 sumFalse+=1   
    stats['sufficient_para']=sumTrue
    stats['insufficient_para']=sumFalse
    
    number_of_sentences=0
    tokens=0
    for txt in df['text']:
        text1=txt.split('\n\n')
        text2=txt.split('\n \n')
        text3=txt.split('\n  \n')
        if(len(text1)==2):
            text=text1[1]
        elif len(text2)==2:
            text=text2[1]
        elif len(text3)==2:
            text=text3[1]  
        txt=nlp(text)    
        sentence=list(txt.sents)      
        number_of_sentences +=len(sentence)
        tokens+=len(txt)
    stats['sentences']=number_of_sentences
    stats['tokens']=tokens
    
    print('Stats Calculated!')
    fig = go.Figure(data=[go.Table(
    header=dict(values=['<b>Name</b>', '<b>Value</b>'],
                line_color='darkslategray',
                fill_color='lightskyblue',
                align='left'),
    cells=dict(values=[['<b>Essays</b>', '<b>Major_Claims</b>','<b>Claims</b>', '<b>Premis</b>','<b>Paragraph</b>','<b>Sufficient Paragraph</b>','<b>Insufficient Paragraph</b>','<b>Essay with Confirmation_bias</b>','<b>Essay without Confirmation_bias</b>','<b>Sentences</b>','<b>Tokens</b>','<b>Major Claims Avg.</b>','<b>Claims Avg.</b>','<b>Premises Avg.</b>','<b>MjaorClaim Specific Words</b>','<b>Claim Specific Words</b>','<b>Premis Specific Words</b>'],
                       [[stats['essay']],
                       [stats['major_claim']],
                       [stats['claim']],
                       [stats['premis']],
                       [stats['paragraph']],
                       [stats['sufficient_para']],
                       [stats['insufficient_para']],
                       [stats['essay_with_conformation']],
                       [stats['essay_without_conformation']],
                       [stats['sentences']],
                       [stats['tokens']],
                       [stats['major_claim_avg']],
                       [stats['claim_avg']],
                       [stats['premis_avg']],
                       [stats['topMajorClaim']],[stats['topClaim']],[stats['topPremis']]]],
               line_color='darkslategray',
               fill_color='lightcyan',
               align='left'))
    ])
    fig.update_layout(width=950, height=1200)
    display(fig)

#file reading starts
trainFiles=filterTrainData()
essays = readEssays(trainFiles)
labels=readLabels()
paragraphs=readParagraphs()
argJSONList=[]
print('Reading annotation files...')
for root, dirs, files in os.walk(essays_path+'/'):
    for file in files:
        if file.endswith('.ann') and isTrainData(trainFiles,file):
            with open(os.path.join(root, file), 'r') as f:
                major_claims=[]
                claims=[]
                premises=[]
                for line in f:
                    major_span=[]
                    claim_span=[]
                    premis_span=[]
                    line = line.strip()
                    columns = line.split()
                    if(columns[1]=='MajorClaim') :
                        major_span.append(int(columns[2]))
                        major_span.append(int(columns[3]))
                        major_text=columns[4:]
                        major_claims.append(major_claim(major_span,major_text))
                    if(columns[1]=='Claim') :
                        claim_span.append(int(columns[2]))
                        claim_span.append(int(columns[3]))
                        claim_text=columns[4:]
                        claims.append(claim(claim_span,claim_text))
                    if(columns[1]=='Premise') :
                        premis_span.append(int(columns[2]))
                        premis_span.append(int(columns[3]))
                        premis_text=columns[4:] 
                        print(premis_text)
                        premises.append(premis(premis_span,premis_text)) 
                fileId=getFileId(file)        
                arg= argument(int(fileId),essays[fileId],major_claims,claims,premises,labels[int(fileId)-1],paragraphs[int(fileId)]) 
                argJSONList.append(createJSONobj(arg)) 
print('Done reading annotation files!')                
with open(unified_json_path, 'w') as outfile:
    json.dump(argJSONList, outfile, indent=2) 
print('Json file created!')     


  
calculateStatistics()