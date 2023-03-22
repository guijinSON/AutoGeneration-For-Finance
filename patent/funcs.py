import pandas as pd
import pickle
from tqdm import tqdm

class patent_analysis():
    def __init__(self, 
                 patentStats_pth="patentStats.xlsx",
                 citationStats_pth="citation_stats.csv",
                 price_pth = "s&p500_sector_performance.xlsx",
                 sec2uspc_pth="sec2uspc.pkl"):
        self.psdf = pd.read_excel(patentStats_pth)
        self.csdf = pd.read_csv(citationStats_pth, dtype=str)
        self.price = pd.read_excel(price_pth)
        self.csdf['count'] = self.csdf['count'].astype(int)
        
        with open(sec2uspc_pth, 'rb') as f:
            self.sec2uspc = pickle.load(f)
        
    def get_all_stats_multi(self,sec):
        
        if sec not in ["ENRS","HLTH","INFT"]:
            raise Exception("ENRS, HLTH, INFT 중 하나의 값을 입력해야 합니다.")
            
        uspcls = self.sec2uspc[sec]
        o = [self.get_all_stats_single(uspc) for uspc in tqdm(uspcls)]
        o = sum(o)/len(o)
        o['value_change_ratio'] = [float(self.price[self.price['sector']=='INFT'][i].values) for i in range(2008,2018)]
        return o
        
        
    def get_all_stats_single(self,uspc):
        acc, con = self.get_basic_stats(uspc)
        inward = self.inward_citation_count(uspc)['count'].values
        recursive = self.recursive_citation_count(uspc)['count'].values
        nonrecursive = self.nonrecursive_citation_count(uspc)['count'].values
        diversity = self.citation_diversity_count(uspc)['count'].values
        
        
        o = pd.DataFrame({'year':list(range(2008,2018)),
                          'acc' :acc['count'].values,
                          'con' :con['count'].values,
                          'inwardC':inward,
                          'recursiveC':recursive,
                          'nonrecursiveC':nonrecursive,
                          'diversityC':diversity
                          })
        return o

    def get_basic_stats(self,uspc):
        uspc = self.uspc2string(uspc)
        batch = self.psdf[self.psdf['uspc_class']==uspc]
        
        accbatch = batch[['year','acc']]
        accbatch.columns = ['year','count']
        accbatch = self.fill_year(accbatch)
        
        conbatch = batch[['year','con']]
        conbatch.columns = ['year','count']
        conbatch = self.fill_year(conbatch) 
        return accbatch, conbatch
    
    def inward_citation_count(self,uspc):
        uspc = self.uspc2string(uspc)
        batch = self.csdf[self.csdf['src_uspc']==uspc]
        batch = batch.groupby('year')['count'].sum().reset_index()
        batch = self.fill_year(batch)
        return batch
    
    def recursive_citation_count(self,uspc):
        uspc = self.uspc2string(uspc)
        batch = self.csdf[(self.csdf['src_uspc']==uspc) & (self.csdf['citedby_uspc']==uspc)]
        batch = batch.groupby('year')['count'].sum().reset_index()
        batch = self.fill_year(batch)
        return batch
    
    def nonrecursive_citation_count(self,uspc):
        uspc = self.uspc2string(uspc)
        i = self.inward_citation_count(uspc)
        r = self.recursive_citation_count(uspc)
        batch = pd.DataFrame({'year':i['year'], 'count': i['count']-r['count']})
        batch = self.fill_year(batch)
        return batch 
    
    def citation_diversity_count(self,uspc):
        uspc = self.uspc2string(uspc)
        batch = self.csdf[self.csdf['src_uspc']==uspc]
        diversity_series = [len(set(i[1]['citedby_uspc'].values)) for i in batch.groupby('year')]
        year = [i[0] for i in batch.groupby('year')]
        batch = pd.DataFrame({'year':year, 'count': diversity_series})
        batch = self.fill_year(batch)
        return batch
    
    def uspc2string(self,uspc):
        uspc = ''.join(['0' for i in range(3-len(str(uspc)))]) + str(uspc)
        return uspc
    
    def fill_year(self,batch):
        year = [str(y) for y in batch['year'].values]
        count = list(batch['count'].values)
        missing_year = [str(i) for i in list(range(2008,2018)) if str(i) not in year]

        if len(missing_year) != 0:
            for m in missing_year:
                year.append(str(m))
                count.append(0)
        o = pd.DataFrame({'year':year,'count':count})
        o = o.sort_values(['year'])
        return o
