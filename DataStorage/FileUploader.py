from neo4j import GraphDatabase,basic_auth
import networkx as nx

## to be as a class for generic files 

class FileUploader():
    
    def __init__(self, filename="facebook.txt",uri = "bolt://localhost:7687",username="neo4j",password="1151999aymana"):
        self.filename = filename
        self.uri = uri
        self.username = username
        self.password = password
        self.session=None
    def getConnection(self):
        driver = GraphDatabase.driver(uri =self.uri, auth=basic_auth(self.username, self.password))
        session=driver.session()
        self.session=session
        print("Seccessfully connected to  "+self.uri)
        return session

    #-------------------------------------------------------------------------------------------------------------------
    def create_user(self,id,session,listNode,graphModel):
        label=""
        if graphModel=="FB":
            pass
        if graphModel=="ABS":
            label="_small_random"
        if graphModel=="ABM":
            label="_medium_random"
        if graphModel=="ABL":
            label="_large_random"
            
        ids=str(id)
        if id in listNode: pass
        else:
            query= "create (u:user"+label+"{id_user:"+ids+"})"
            session.run(query)
            listNode.append(id)
        return listNode
    
    #-------------------------------------------------------------------------------------------------------------------
    def create_relation(self,id1,id2,session,listEdge,graphModel):
        label=""
        labelR=""
        if graphModel=="FB":
            pass
        if graphModel=="ABS":
            label="_small_random"
            labelR="_in_"+graphModel
        if graphModel=="ABM":
            label="_medium_random"
            labelR="_in_"+graphModel
        if graphModel=="ABL":
            label="_large_random"
            labelR="_in_"+graphModel
            
        ids1=str(id1)
        ids2=str(id2)
        if (ids1+" "+ids2) in listEdge: pass
        else :
            query= "match (u1:user"+label+"{id_user:"+ids1+"}) ,(u2:user"+label+"{id_user:"+ids2+"}) create (u1)-[r:friend"+labelR+"]->(u2) return r"
            session.run(query)
            listEdge.append(ids1+" "+ids2)
        return listEdge

    #-------------------------------------------------------------------------------------------------------------------
    def uploadFile(self,graphModel):
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
            listNode=self.create_user(i[0],session,listNode,graphModel)
            listNode=self.create_user(i[1],session,listNode,graphModel)
            listEdge=self.create_relation(i[0], i[1],session,listEdge,graphModel)
        print("successfully created in database")
               
    def uploadGraphToDB(self,graphModel):
        l=[]
        g=nx.Graph()
        if graphModel=="ABS":
            g = nx.barabasi_albert_graph(100, 4)
        if graphModel=="ABM":
            g = nx.barabasi_albert_graph(1000, 7)
        if graphModel=="ABL":
            g = nx.barabasi_albert_graph(5000, 12)    
            
        l=g.edges()
        session=self.getConnection()
        listEdge = []
        listNode = []
        for i in l:
            listNode=self.create_user(int(i[0]),session,listNode,graphModel)
            listNode=self.create_user(int(i[1]),session,listNode,graphModel)
            listEdge=self.create_relation(int(i[0]), int(i[1]),session,listEdge,graphModel)
        print("successfully created the "+graphModel+" graph in the database")
        self.add_graph_metrics(g,graphModel,session)

    #--------------------------------------------------------------------------------------------------------------------
    def add_graph_metrics(self,graph,graphModel):
        label=""
        if graphModel=="FB":
            pass
        if graphModel=="ABS":
            label="_small_random"
        if graphModel=="ABM":
            label="_medium_random"
        if graphModel=="ABL":
            label="_large_random"
        session=self.getConnection()
        degres=nx.degree(graph)
        deg_cent=nx.degree_centrality(graph)
        clos_cent=nx.closeness_centrality(graph)
        betw_cent=nx.betweenness_centrality(graph)
        page_rank=nx.pagerank(graph,alpha=0.8)
        clustering=nx.clustering(graph)
        
        for i in range(len(graph.nodes())):
            query ="MATCH (u:user"+label+"{id_user:"+str(i)+"}) SET u.degree= "+str(degres[i]) +",u.degree_centrality= "+str(deg_cent[i]) 
            query+=", u.closness_centrality= "+ str(clos_cent[i])+", u.between_centrality= "+str(betw_cent[i]) 
            query+=", u.page_rank= "+ str(page_rank[i])+", u.clustering= "+str(clustering[i]) 

            try:
                session.run(query)
            except:
                print("connection error in add_graph_metrics")
        print("Metrics added successfully")

    