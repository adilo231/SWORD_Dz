from  pymongo import MongoClient
from neo4j import GraphDatabase,basic_auth
import os
from tqdm import tqdm
import json
import random


class DocDBHandler():
    def __init__(self):
        f = open('authentification.json')
        data = json.load(f)
        myclient=None
        try:
            myclient=MongoClient(data['mongo']['uri'])
        except:
            print("couldn't connect to Mongo")
        if myclient!=None:
            self.myclient = myclient

    def GetDocuments(self):
        return self.col.find()
    
    def setDB(self,db):
        self.mydb = self.myclient[db]

    def setCollection(self,col):
        try:
            self.col = self.mydb[col]
        except : print('can\'t set database')

    def add_doc(self,data):
        try:
            self.col.insert_one(data)
        except : print('can\'t insert data')

    def add_tweet(self,data,colection, collection_user=None,with_user=False):
                if with_user == True:
                    user=data['user']
                    data['user']=user['id']                      
                    self.setCollection(collection_user)
                    if self.col.count_documents({ "id": user['id'] })==0:         
                        loader.add_doc(user)

                loader.setCollection(colection)
                loader.add_doc(data)




class GraphDBHandler():
    def __init__(self):
        self.session=None
        self.driver=None
        f = open('authentification.json')
        data = json.load(f)
        auth=data['neo4j']
        self.driver = GraphDatabase.driver(uri = auth['uri'], auth=basic_auth(auth['username'], auth['password']))
        self.session=self.driver.session()
        print(self.session)
        # except:
        #      print ("Authentication to Graph DB failed")
      
           

    def UserExist(self,lable,id):
        query= f"match (node:{lable}{{ id:{id} }}) return  node"
        return len(list(self.session.run(query)))>0
    def LinkExist(self,lable1,lable2,RelationType,id1,id2):
        query= f"match (u1:{lable1}{{ id:{id1} }})-[r:{RelationType}]->(u2:{lable2}{{id:{id2} }}) return r"
        return len(list(self.session.run(query)))>0
    def AddUserNode(self,lable,id):
        if not self.UserExist(lable,id):
            query= f"create (u:{lable}{{id:{id} }})"
            self.session.run(query)
        
    def Addlink(self,lable1,lable2,RelationType,id1,id2):
        if not self.LinkExist(lable1,lable2,RelationType,id1,id2):
            query= f"match (u1:{lable1}{{ id:{id1} }}) ,(u2:{lable2}{{id:{id2} }}) create (u1)-[r:{RelationType}]->(u2) return r"
            self.session.run(query)
    
    def getConnection(self):
        return self.session
      

        

if __name__ == '__main__':

    user, password="neo4j", "admin"
    handler = GraphDBHandler(user, password)

    
    connectionString="mongodb://localhost:27017/"
    loader = DocDBHandler(connectionString,"PhemeDataset","ebola_essien")

    Source='all-rnr-annotated-threads'
    arr = os.listdir(Source)
    DatasetName=['ebola-essien']
    # for i in range(0,len(arr)):
    #     DatasetName.append(arr[i].split('-all')[0])
    # print(DatasetName)

    for dataset in tqdm(DatasetName,desc=" Dataset", position=0):
        
    
        types=['rumours','non-rumours']
        for type in types:
            path=f'{Source}/{dataset}-all-rnr-threads/{type}'
            for folder in tqdm(os.listdir(path),desc=f"{dataset}", position=1, leave=False):
                if not os.path.isfile(folder):
                    with open(f'{path}/{folder}/source-tweets/{folder}.json') as f:
                        
                        data = json.load(f)
                        data['annotation']=type
                        
                        loader.add_tweet(data,dataset.replace('-','_'),'users',with_user=True)

                        
                    for tweet in (os.listdir(f'{path}/{folder}/reactions')):
                        with open(f'{path}/{folder}/reactions/{tweet}') as f:
                            
                            if tweet != '.DS_Store':
                                data = json.load(f)
                                data['annotation']=type
                                loader.add_tweet(data,dataset.replace('-','_'),'users',with_user=True)
    
    # for dataset in tqdm(loader.db.list_collection_names(),desc=" Dataset", position=0):
    #     loader.setCollection(dataset)
    #     for element in tqdm(loader.col.find(),desc=f"{dataset}", position=1):
        
    #         handler.AddUserNode('tweet',element['id'])
    #         handler.AddUserNode('userTwitter',element['user'])
    #         handler.Addlink('userTwitter','tweet','Tweeted',element['user'],element['id'])
    #         if (element['in_reply_to_status_id']!=None):
    #             handler.AddUserNode('tweet',element['in_reply_to_status_id'])
    #             handler.Addlink('tweet','tweet','Reweeted',element['id'],element['in_reply_to_status_id'])
    #             handler.Addlink('userTwitter','userTwitter','follow',element['user'],element['in_reply_to_user_id'])