from neo4j import GraphDatabase,basic_auth

## to be as a class for generic files 

class FileUploader():
    
    def __init__(self, filename="facebook.txt",uri = "bolt://localhost:7687",username="neo4j",password="1151999aymana"):
        self.filename = filename
        self.uri = uri
        self.username = username
        self.password = password

    def getConnection(self):
        driver = GraphDatabase.driver(uri =self.uri, auth=basic_auth(self.username, self.password))
        session=driver.session()
        print("Seccessfully connected to  "+self.uri)
        return session

    #-------------------------------------------------------------------------------------------------------------------
    def create_user(self,id,session,listNode):
        ids=str(id)
        if id in listNode: pass
        else:
            query= "create (u:user{id_user:"+ids+"})"
            session.run(query)
            listNode.append(id)
        return listNode
    
    #-------------------------------------------------------------------------------------------------------------------
    def create_relation(self,id1,id2,session,listEdge):
        ids1=str(id1)
        ids2=str(id2)
        if (id1+" "+id2) in listEdge: pass
        else :
            query= "match (u1:user{id_user:"+ids1+"}) ,(u2:user{id_user:"+ids2+"}) create (u1)-[r:friend]->(u2) return r"
            session.run(query)
            listEdge.append(id1+" "+id2)
        return listEdge

    #-------------------------------------------------------------------------------------------------------------------
    def uploadFile(self):
        l=[]
        with open(self.filename) as file:
            for line in file:
                words=[]
                for word in line.split():
                    words.append(word)

                l.append(words)
        #print("successfully registered file in matrice :",len(l)," edge")
        session=self.getConnection()
        listEdge = []
        listNode = []
        for i in l:
            listNode=self.create_user(i[0],session,listNode)
            listNode=self.create_user(i[1],session,listNode)
            listEdge=self.create_relation(i[0], i[1],session,listEdge)
        print("successfully created in database")
