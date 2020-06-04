import glob
import pandas as pd
import unicodedata
import re
import sys
from transformers import AutoTokenizer

class sentence:
    def __init__(self,inp,nan):
        self.page=inp['Page No']
        self.text=inp['Text']
        self.index=inp['Index']
        self.par=0 if nan['Parent Index'] else inp['Parent Index']
        self.title=not nan['Is Title']
        self.table=not nan['Is Table']
        self.tagstr=inp['Tag'] if 'Tag' in inp and not nan['Tag'] else ''
        self.valuestr=inp['Value'] if 'Value' in inp and not nan['Value'] else ''
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
                    self.valuestr=self.valuestr.replace(';',' ')
                    self.value=self.valuestr.split(';')
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
def preprocess(dir,tokenizer,k=0,side=1):
    print('Start reading and parsing csv',file=sys.stderr)
    tags=['調達年度','都道府県','入札件名','施設名','需要場所(住所)','調達開始日','調達終了日','公告日','仕様書交付期限','質問票締切日時','資格申請締切日時','入札書締切日時',\
        '開札日時','質問箇所所属/担当者','質問箇所TEL/FAX','資格申請送付先','資格申請送付先部署/担当者名','入札書送付先','入札書送付先部署/担当者名','開札場所']
    res=[]
    for fn in glob.glob(dir):
        with open(fn,'r') as f:
            df=pd.read_excel(fn)
        cur=[]
        for i in range(len(df)):
            nan=df.iloc[i].isna()
            inp=dict(df.iloc[i])
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
    for tag in tags:
        for a in res:
            bnd=[]
            toklen=[]
            n=len(a)
            i,j=0,0
            while i<n:
                if tag not in a[i].tag:
                    curlen=len(tokenizer.tokenize(a[i].text))+1
                    bnd.append((i,i+1))
                    toklen.append(curlen)
                    i+=1
                else:
                    curlen=0
                    j=i
                    while j<n and tag in a[j].tag:
                        curlen+=len(tokenizer.tokenize(a[j].text))+1
                        j+=1
                    bnd.append((i,j))
                    toklen.append(curlen)
                    i=j
            n=len(bnd)
            l,r=0,0
            while l<n:
                r=l
                curlen=0
                while r<n and curlen+toklen[r]<=512-15:
                    curlen+=toklen[r]
                    r+=1
                qa_text=tag
                con_text=''
                ans_text=''
                for i in range(l,r):
                    flag=ans_text=='' and tag in a[bnd[i][0]].tag
                    #print(flag,bnd[i],a[bnd[i][0]].tag,file=sys.stderr)
                    for j in range(bnd[i][0],bnd[i][1]):
                        con_text+='[SEP]'+a[j].text
                        if flag:
                            try:
                                ans_text+='[SEP]'+a[j].QAdict[tag]
                            except:
                                print(tag,j,a[j].QAdict,file=sys.stderr)
                                print(a[j].tag,a[j].value,a[j].QA,file=sys.stderr)
                con_text=con_text[5:]
                ans_text=ans_text[5:]
                ret.append(QAExample(con_text,qa_text,ans_text))
                if side==1:
                    l=r-k
                else:
                    l=l+k
    print(len(ret),file=sys.stderr)
    return ret

#preprocess('./train/ca_data/*',AutoTokenizer.from_pretrained("bert-base-multilingual-cased"),0,1)
#preprocess(input_files,tokenizer,k,side)
#side=0: current_window[l,r) => next_window[l+k,...)
#side=1: current_window[l,r) => next_window[r-k,...)

