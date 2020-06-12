import glob
import pandas as pd
import unicodedata
import re
import sys
from tqdm import tqdm
from _QA import QAExample
#from transformers import AutoTokenizer, BertModel, AdamW

class sentence:
    def __init__(self,inp,nan):
        self.filename=inp['filename']
        self.page=inp['Page No']
        self.text=inp['Text'].replace(' ','')
        self.index=inp['Index']
        self.par=0 if nan['Parent Index'] else inp['Parent Index']
        self.title=not nan['Is Title']
        self.table=not nan['Is Table']
        self.tagstr=inp['Tag'].replace(' ','') if 'Tag' in inp and not nan['Tag'] else ''
        self.valuestr=inp['Value'].replace(' ','') if 'Value' in inp and not nan['Value'] else ''
        if self.valuestr=='':
            self.tagstr=''
        self.tagstr = unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', self.tagstr))
        self.tag=[] if self.tagstr=='' else self.tagstr.split(';')
        self.value=[] if self.valuestr=='' else self.valuestr.split(';')
        if len(self.tag)!=len(self.value):
            try:
                assert (len(self.value)==1 and len(self.tag)>0) or (len(self.value)>0 and len(self.tag)==1)
                if len(self.value)==1:
                    self.value=self.value*len(self.tag)
                else:
                    self.value=[self.valuestr]
            except:
                #print(inp,file=sys.stderr)
                if len(self.tag)==2:
                    self.tag=['調達年度']+self.tag
                else:
                    self.valuestr=self.valuestr.replace('から',';')
                    self.value=self.valuestr.split(';')
        assert len(self.tag)==len(self.value)
        self.QA=list(zip(self.tag,self.value))
        self.QAdict=dict(self.QA)
        #print(vars(self),file=sys.stderr)
def preprocess(dir,tokenizer,merge_type=0,one_ans=0,k=0,side=1):
    print('Start reading and parsing csv',file=sys.stderr)
    tags=['調達年度','都道府県','入札件名','施設名','需要場所(住所)','調達開始日','調達終了日','公告日','仕様書交付期限','質問票締切日時','資格申請締切日時','入札書締切日時',\
        '開札日時','質問箇所所属/担当者','質問箇所TEL/FAX','資格申請送付先','資格申請送付先部署/担当者名','入札書送付先','入札書送付先部署/担当者名','開札場所']
    sep_token='[SEP]'
    res=[]
    index_list = dict()
    for fn in glob.glob(dir):
        with open(fn,'r') as f:
            df=pd.read_excel(fn)
        cur=[]

        pos = fn.find('.pdf')
        fn_tmp = fn[pos-9 : pos]
        index_list[fn_tmp] = []

        for i in range(len(df)):
            nan=df.iloc[i].isna()
            inp=dict(df.iloc[i])
            inp['filename']=fn

            index_list[fn_tmp].append(inp['Index'])

            try:
                cur.append(sentence(inp,nan))
                for j in cur[-1].tag:
                    try:
                        assert j in tags
                    except:
                        print(j,file=sys.stderr)
                        #exit(0)
            except:
                print(fn,i,file=sys.stderr)
                print(inp,file=sys.stderr)
                #exit(0)
        res.append(cur)
    print('Start creating dataset',file=sys.stderr)
    ret=[]
    for tag in tqdm(tags):
        for a in res:
            bnd=[]
            toklen=[]
            has_ans=[]
            n=len(a)
            i,j=0,0
            while i<n:
                if tag not in a[i].tag:
                    curlen=len(tokenizer.tokenize(a[i].text+sep_token))
                    bnd.append((i,i+1))
                    toklen.append(curlen)
                    has_ans.append(False)
                    i+=1
                else:
                    curlen=0
                    j=i
                    while j<n and tag in a[j].tag:
                        curlen+=len(tokenizer.tokenize(a[j].text+sep_token))+1
                        j+=1
                    bnd.append((i,j))
                    toklen.append(curlen)
                    has_ans.append(True)
                    i=j
            n=len(bnd)
            l,r=0,0
            while l<n:
                r=l
                curlen=0
                hans=False
                while r<n and curlen+toklen[r]<=512-16 and not (one_ans and hans and has_ans[r]):
                    curlen+=toklen[r]
                    hans|=has_ans[r]
                    r+=1
                qa_text=tag
                con_text=''
                ans_text=''
                sid = []
                for i in range(l,r):
                    flag=ans_text=='' and tag in a[bnd[i][0]].tag
                    #print(flag,bnd[i],a[bnd[i][0]].tag,file=sys.stderr)
                    for j in range(bnd[i][0],bnd[i][1]):
                        sid.append(a[j].index)
                        con_text+=sep_token+a[j].text
                        if flag:
                            try:
                                if merge_type==0:
                                    ans_text+=sep_token+a[j].QAdict[tag]
                                elif merge_type==1:
                                    pos=a[j].text.find(a[j].QAdict[tag])
                                    assert pos!=-1
                                    ans_text+=sep_token+a[j].text[:pos+len(a[j].QAdict[tag])]
                            except:
                                print(a[j].text,a[j].QA,file=sys.stderr)
                con_text=con_text[len(sep_token):]
                ans_text=ans_text[len(sep_token):]
                doc_id=a[bnd[l][0]].filename[-18:-9]
                pid=a[bnd[l][0]].page
                ret.append(QAExample(doc_id,sid,pid,con_text,qa_text,ans_text))
                if side==1:
                    l=max(r-k,l+1)
                else:
                    l=l+k 

    print(len(ret),file=sys.stderr)
    return ret, index_list 

#preprocess('release/train/ca_data/*',AutoTokenizer.from_pretrained("bert-base-multilingual-cased"),0,0,0,1)
#preprocess(input_files,tokenizer,merge_type,one_ans,k,side)
#side=0: current_window[l,r) => next_window[l+k,...)
#side=1: current_window[l,r) => next_window[r-k,...)
#merge_type=0: AB[SEP]CD with answer BD will be B[SEP]D
#merge_type=1: AB[SEP]CD with answer BD will be B[SEP]CD (failed)
#one_ans=0: each data may have more than one contiguous answer, and let the first one be the answer.
#one_ans=1: each data will have only one contiguous answer.