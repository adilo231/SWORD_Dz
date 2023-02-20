from neo4j import GraphDatabase,basic_auth

## to be as a class for generic files 

driver = GraphDatabase.driver(uri = "bolt://localhost:7687", auth=basic_auth("neo4j", "nicekiller"))
session=driver.session()
print(session)

#-------------------------------------------------------------------------------------------------------------------
def create_user(id,session,listNode):
    ids=str(id)
    if id in listNode: pass
    else:
        query= "create (u:user{id_user:"+ids+"})"
        session.run(query)
        listNode.append(id)
    return listNode
   
#-------------------------------------------------------------------------------------------------------------------
def create_relation(id1,id2,session,listEdge):
    ids1=str(id1)
    ids2=str(id2)
    if (id1+" "+id2) in listEdge: pass
    else :
        query= "match (u1:user{id_user:"+ids1+"}) ,(u2:user{id_user:"+ids2+"}) create (u1)-[r:friend]->(u2) return r"
        session.run(query)
        listEdge.append(id1+" "+id2)
    return listEdge
    
#-------------------------------------------------------------------------------------------------------------------
l=[]
with open("facebook.txt") as file:
    for line in file:
        words=[]
        for word in line.split():
            words.append(word)
            
        l.append(words)
print("successfully registered file in matrice :",len(l)," edge")

listEdge = []
listNode = []
for i in l:
    listNode=create_user(i[0],session,listNode)
    listNode=create_user(i[1],session,listNode)
    listEdge=create_relation(i[0], i[1],session,listEdge)
print("successfully created in database")
