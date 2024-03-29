from neo4j import GraphDatabase,basic_auth
import networkx as nx
import pandas as pd
from DataStorage.DBHandlers import GraphDBHandler 
## to be as a class for generic files 

class FileUploader():
    
    def __init__(self, filename="facebook.txt"):
        self.filename = filename
        Handler=GraphDBHandler()
        self.session=Handler.getConnection()
   

    #-------------------------------------------------------------------------------------------------------------------
    def create_user(self,id,listNode,graphModel):
        label=""
        if graphModel=="FB":
             label="_facebook"
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
            self.session.run(query)
            listNode.append(id)
        return listNode
    
    #-------------------------------------------------------------------------------------------------------------------
    def create_relation(self,id1,id2,listEdge,graphModel):
        label=""
        labelR=""
        if graphModel=="FB":
            label="_facebook"
            labelR="_in_"+graphModel
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
            self.session.run(query)
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
        g=nx.Graph()
        #query= "CREATE INDEX  FOR (n:user) ON (n.id_user)"
        #session.run(query)
        print("started load facebook data base")
        listEdge = []
        listNode = []
        for i in l:
            listNode=self.create_user(i[0],listNode,graphModel)
            listNode=self.create_user(i[1],listNode,graphModel)
            listEdge=self.create_relation(i[0], i[1],listEdge,graphModel)
            u1=int(i[0])
            u2=int(i[1])
            g.add_nodes_from([u1,u2])
            g.add_edge(u1,u2)
        print("successfully created in database")
        self.add_graph_metrics(g,graphModel)
        print("successfully for add metrics")
        
              
    def uploadGraphToDB(self,graphModel):
        l=[]
        g=nx.Graph()
        if graphModel=="ABS":
            g = nx.barabasi_albert_graph(100, 4)
            query= "CREATE INDEX  FOR (n:user_small_random) ON (n.id_user)"
        if graphModel=="ABM":
            g = nx.barabasi_albert_graph(1000, 7)
            query= "CREATE INDEX  FOR (n:user_medium_random) ON (n.id_user)"
        if graphModel=="ABL":
            g = nx.barabasi_albert_graph(5000, 12)
            query= "CREATE INDEX  FOR (n:user_large_random) ON (n.id_user)"
         
        #self.session.run(query)       
        l=g.edges()
       
        listEdge = []
        listNode = []
        for i in l:
            listNode=self.create_user(int(i[0]),listNode,graphModel)
            listNode=self.create_user(int(i[1]),listNode,graphModel)
            listEdge=self.create_relation(int(i[0]), int(i[1]),listEdge,graphModel)
        print("successfully created the "+graphModel+" graph in the database")
        self.add_graph_metrics(g,graphModel)

    #--------------------------------------------------------------------------------------------------------------------
    def add_graph_metrics(self,graph,graphModel):
        label=""
        if graphModel=="FB":
            label="_facebook"
        if graphModel=="ABS":
            label="_small_random"
        if graphModel=="ABM":
            label="_medium_random"
        if graphModel=="ABL":
            label="_large_random"
      
        degres=nx.degree(graph)
        deg_cent=nx.degree_centrality(graph)
        clos_cent=nx.closeness_centrality(graph)
        betw_cent=nx.betweenness_centrality(graph)
        page_rank=nx.pagerank(graph,alpha=0.8)
        clustering=nx.clustering(graph)
        communities = nx.community.greedy_modularity_communities(graph)
        groups = {}
        for i, com in enumerate(communities):
            for node in com:
                groups[node] = i
        
        for i in range(len(graph.nodes())):
            query ="MATCH (u:user"+label+"{id_user:"+str(i)+"}) SET u.degree= "+str(degres[i]) +",u.degree_centrality= "+str(deg_cent[i]) 
            query+=", u.closness_centrality= "+ str(clos_cent[i])+", u.between_centrality= "+str(betw_cent[i]) +",u.page_rank= "+str(page_rank[i])
            query+=", u.clustering= "+str(clustering[i])
            query+=",u.group="+str(groups[i])

            try:
                self.session.run(query)
            except:
                print("connection error in add_graph_metrics")
        print("Metrics added successfully")

    