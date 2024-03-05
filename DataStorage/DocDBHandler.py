from  pymongo import MongoClient
import json


class DocDBHandler():
    def __init__(self,database_name=None, collection_name=None):
        f = open('authentification.json')
        auth_data = json.load(f)
        self.myclient=None
        try:
           self. myclient=MongoClient(auth_data['mongo']['uri'])
        except Exception as e:
            print("couldn't connect to Mongo")
        if self.myclient!=None:
            self.myclient = self.myclient
        if database_name != None :
            self.setDB(database_name)
            if collection_name != None :
                self.setCollection(collection_name)


    def setDB(self,db):
        try:
            self.mydb = self.myclient[db]
        except Exception as e:
            print('can\'t set database')

    def setCollection(self,col):
        try:
            self.col = self.mydb[col]
        except Exception as e:
            print('can\'t set collection')

    def GetDocuments(self,database=None,collection=None):
        try:
            if database != None and collection != None :
                self.setDB(database)
                self.setCollection(collection)
            cursor = self.col.find()
            return [doc for doc in cursor]
        except Exception as e:
            print("can't find collection")

    def add_doc(self,data):
        try:
            self.col.insert_one(data)
            print("doc is inserted")
        except Exception as e:
            print('can\'t insert data')

    def add_tweet(self,data,database=None,collection=None, collection_user=None,with_user=False):
                if with_user == True:
                    user=data['user']
                    data['user']=user['id']                      
                    self.setCollection(collection_user)
                    if self.col.count_documents({ "id": user['id'] })==0:         
                        self.add_doc(user)
                if database != None and collection != None :
                    self.setDB(database)
                    self.setCollection(collection)
                self.add_doc(data)


