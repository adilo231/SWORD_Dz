

import numpy as np
from networkx.readwrite import json_graph
import networkx as nx
import matplotlib.pyplot as plt
import random
import time
#from matplotlib.ticker import NullFormatter  
import multiprocessing 
from multiprocessing import Manager
import math
from networkx import edge_betweenness_centrality
from networkx.algorithms.approximation import min_weighted_vertex_cover
from scipy.special import gamma, factorial

def Neighbour_finder(g,new_active):
    
    targets = []
    for node in new_active:
        targets += g.neighbors(node)
    return(targets)
def jackard(g, n1, n2):
    nb_intersection = 0
    for i in range(len(g.nodes[n1]['neighbors'])):
        for j in range(len(g.nodes[n2]['neighbors'])):
            if g.nodes[n1]['neighbors'][i] == g.nodes[n2]['neighbors'][j]:
                nb_intersection+=1
    nb_union = len(g.nodes[n1]['neighbors']) + len(g.nodes[n2]['neighbors']) - nb_intersection
    if nb_union != 0:
        return nb_intersection/nb_union 
    else:
        return 0
def remove_links(g,e0,e1):
            g.nodes[e0]['degree']-=1
            g.nodes[e1]['degree']-=1
            if e1 in g.nodes[e0]['neighbors']:
             g.nodes[e0]['neighbors'].remove(e1)
            if e0 in g.nodes[e1]['neighbors']:
             g.nodes[e1]['neighbors'].remove(e0)
def update_links(g, stubs, percentage_adding):
    #dispearing of edges
    list_to_remove = []
    print("debut",len(g.edges))
    for e in g.edges:
        r = random.random()
        #print('r:', type(e))
        if r/2/len(g.nodes) > np.random.random_sample():
            #print("before",len(g.nodes[e[1]]['neighbors']))
            remove_links(g,e[0],e[1])
            
            #print("after",len(g.nodes[e[1]]['neighbors']))
            list_to_remove.append(e)
            stubs.append(e[0])
            stubs.append(e[1])

    g.remove_edges_from(list_to_remove)  
    #print("fin",len(g.edges))

    #add links 
    #adding form inactive stubs
    #for i in range(int(len(g.nodes)* percentage_adding)):
    for i in range(int(len(stubs)/ percentage_adding)):
        indice1= random.randint(0, len(stubs)-1)
        indice2= random.randint(0, len(stubs)-1)
        if stubs[indice1] != stubs[indice2]:
            if jackard(g, stubs[indice1], stubs[indice2]) > np.random.random_sample():
                g.add_edge(stubs[indice1], stubs[indice2])
                g.nodes[stubs[indice1]]['degree']+=1
                g.nodes[stubs[indice2]]['degree']+=1
                g.nodes[stubs[indice1]]['neighbors'].append(stubs[indice2])
                g.nodes[stubs[indice2]]['neighbors'].append(stubs[indice1])
    #print("fin2",len(g.edges))
    #new addtions
    for i in range(int(len(g.nodes)* percentage_adding)):
        node1= random.randint(0, len(g.nodes)-1)
        node2= random.randint(0, len(g.nodes)-1)
        if node1 != node2:
            if jackard(g, node1, node2) > np.random.random_sample() and  not g.has_edge(node1, node2):
                g.add_edge(node1, node2)
                g.nodes[node1]['degree']+=1
                g.nodes[node2]['degree']+=1
                g.nodes[node1]['neighbors'].append(node2)
                g.nodes[node2]['neighbors'].append(node1)

    #print("fin3",len(g.edges))
def HISBmodel (Graph,Seed_Set,Opinion_Set,Statistical,paramater,K,Tdet,method):
    
    #Opinion:normal/denying/supporting
    #State:non_infected/infected/spreaders 
    #Statistical:{'NonInfected':NbrOFnodes,'Infected':**,'Spreaders':**,OpinionDenying':**,'OpinionSupporting':**,'RumorPopularity':**}
    ######### parameters to update the network ####"
    stubs = []
    percentage_adding = 0.1
    ########################################
    bl=0
    ListInfectedNodes=Seed_Set[:]
    Opinion_Set=Opinion_Set[:]
    time=0.125
    Probability=0.3
    i=0
    #Initialis Parameters----------------------------
    #-------------------------
    Nbr_Spreaders=len(ListInfectedNodes)
    Nbr_nonInfected=len(Graph.nodes)
    Nbr_Infected=0
    OpinionDenying=0
    OpinionSupporting=0
    RumorPopularity=0
    InitParameters(Graph,paramater)
    
    ''' if(L_protector!=None):
       for each  in L_protector:
        Graph.nodes[each]['jug']=1'''
   
    for each  in ListInfectedNodes:
        Graph.nodes[each]['Infetime']=0.125 
        Graph.nodes[each]['state']='spreaders'
        Graph.nodes[each]['AccpR']+=1
        
        RumorPopularity+=Graph.nodes[each]['degree']
        Nbr_Infected+=1
        Nbr_nonInfected-=1
        if (Opinion_Set[i]=='denying'):

            Graph.nodes[each]['opinion']='denying'
            Graph.nodes[each]['Accp_NegR']+=1
            OpinionDenying+=1
        else:
          Graph.nodes[each]['opinion']='supporting'
          OpinionSupporting+=1
        i+=1
        
    
    #------------------------------
    Statistical.append({'NonInfected':Nbr_nonInfected,'Infected':Nbr_Infected,'Spreaders':Nbr_Spreaders,'OpinionDenying':OpinionDenying,'OpinionSupporting':OpinionSupporting,'RumorPopularity':RumorPopularity,'graph':0})
    #----------------------
    #if the list is empty we stop the propagation
    
    while ListInfectedNodes:
      #print("time:", time)
    #   sp=[]
   
    #   search_spreaders(Graph,sp)  
   
    #   nb,DNode,cen,Bet,betaD,beta2D=mvc_neighbor(sp,Graph)
    #   print("ratio: ", len(nb)*Nbr_Infected/len(Graph.nodes)/len(Graph.nodes))
      if time%1 == 0.125:
         update_links(Graph, stubs, percentage_adding) 
      RumorPopularity = 0
      Nbr_Spreaders = 0
      L=len(ListInfectedNodes)
      #print(L)
      for X in reversed(range(0,L)):
        
        id = ListInfectedNodes[X]
        #relative time of rumor spreading
        RelativeTime = time - Graph.nodes[id]['Infetime'] 
        if (np.exp(-RelativeTime * Graph.nodes[id]['beta']) < 0.15) :
          ListInfectedNodes.pop(X)
          Graph.nodes[id]['state'] = "infected"
          
              

        else:
            #atrraction of nodes
            ActualAttraction = np.exp(-RelativeTime * Graph.nodes[id]['beta']) * np.abs(np.sin((RelativeTime * Graph.nodes[id]['omega'] )+ Graph.nodes[id]['delta']))
            
            RumorPopularity += ActualAttraction * Graph.nodes[id]['degree']
            #rumor spreading
            
            c=np.random.random_sample()
            
            if (c<=ActualAttraction):
                Nbr_Spreaders+=1
               
                #Calculating if any nodes of those neighbours can be activated, if yes add them to new_ones.
                success = np.random.uniform(0,1,len(Graph.nodes[id]['neighbors'])) < Probability #choic alpha nodes
                # success == [ True  True  True False  True .... True False False  True False]
                new_ones = list(np.extract(success, sorted(Graph.nodes[id]['neighbors'])))
                
                Graph.nodes[id]['SendR']+=len(new_ones)
                
                #Sending Rumor
                for each in new_ones:
                    #Acceptance of the Rumor Probability 
                    ProbToAccRumor = Graph.nodes[id]['degree']/ (Graph.nodes[id]['degree'] + Graph.nodes[each]['degree'])*0.3
                    if (Graph.nodes[each]['blocked'] =='false' and Graph.nodes[each]['Blockingtime'] == 0):
                        if(np.random.random_sample()<=ProbToAccRumor):
                        
                            Graph.nodes[each]['AccpR']+=1
    
                            if (Graph.nodes[each]['Infetime']==0 ):
                                Nbr_Infected+=1
                                Nbr_nonInfected-=1
                                Graph.nodes[each]['Infetime'] =time
                                Graph.nodes[each]['opinion'] =Graph.nodes[id]['opinion']
                                Graph.nodes[id]['state']='spreaders'
                                ListInfectedNodes.append(each)
                                if (Graph.nodes[each]['opinion']=="denying"):
                                    #negativ opinion
                                    Graph.nodes[each]['Accp_NegR']+=1
                                    OpinionDenying+=1
                                else:
                                     OpinionSupporting+=1
                            elif (Graph.nodes[id]['opinion']=="denying"):
                                Graph.nodes[each]['Accp_NegR']+=1
                    
                   # else:
                        #print('le noeud',each,'is blocked')
                        #updateOpinion(id)'''
                if (Graph.nodes[id]['opinion']=="denying"):
                    OpinionDenying-=1
                else:
                    OpinionSupporting-=1
                Graph.nodes[id]['opinion']= updateOpinion(jug=Graph.nodes[id]['jug'],Accpet_NegR=Graph.nodes[id]['Accp_NegR'],Nbr_OF_R=Graph.nodes[id]['AccpR'],Role=Graph.nodes[id]['Protector'])
                if (Graph.nodes[id]['opinion']=="supporting"):
                    
                    OpinionSupporting+=1
                else:
                    
                    OpinionDenying+=1       
      
      #save each step to send it to viewing later
      Statistical.append({'NonInfected':Nbr_nonInfected,'Infected':Nbr_Infected,'Spreaders':Nbr_Spreaders,'OpinionDenying':OpinionDenying,'OpinionSupporting':OpinionSupporting,'RumorPopularity':RumorPopularity,'graph':0})
      if time >=Tdet*0.125 and bl<K  and method != 'NP' :
          #print(method," At time:", time, "blocked nodes Nbr:", bl)
          Nodes=len(Graph.nodes)
          
          p=K-bl
          if (method=='BNLS'):
              Random_Blocking_nodes(Graph, p)
              bl=len(blocked(Graph))
          if (method=='BNLSM'):
              Degree_MAX_Blocking_nodes(Graph,p)
          if (method=='BNSWijayanto'):
              #if time%1 ==0.125:
                wijayantoStrategy(Graph,p,K)
                bl=len(blocked(Graph))
          if (method=='BNLSCen'):
              Centrality_Blocking_nodes(Graph,p)
              bl=len(blocked(Graph))
          if (method=='BNLSBeta'):
              Beta_Blocking_nodes(Graph,p)
              bl=len(blocked(Graph))
          if (method=='BNLSBetaD'):
              BetaD_Blocking_nodes(Graph,p)
              bl=len(blocked(Graph))
          if (method=='BLS'):
              Blocking_links(Graph,p)
              bl=K
          if (method=='DRIMUX'):
              DRIMUX(Graph,p,time)
              bl=len(blocked_tempo(Graph))
          if (method=='BNLSBetaD_time'):
              BetaD_Blocking_nodes_temporary(Graph,p)
              bl=len(blocked_tempo(Graph))
          if (method=='BNLSBetaD_time_mvc'):
              BetaD_Blocking_nodes_temporary(Graph,p)
              bl=len(blocked_tempo(Graph))
          if (method=='BNLSBeta_time_tolerance_mcv'):
              mvc_BetaD_Blocking_nodes_temprary_tolerance0(Graph,p)
              bl=len(blocked_tempo(Graph))
          if (method=='AGPUX'):
              mvc_BetaD_Blocking_nodes_temprary_tolerance_adaptative(Graph,p)
              bl=len(blocked_tempo(Graph))
          if (method=='BNLSBetaD_mvc'):
              mvc_BetaD_Blocking_nodes(Graph,p)
              bl=len(blocked(Graph))
          if (method=='newHybridApproach'):
            
              nb_supporting = Statistical[len(Statistical)-1]['OpinionSupporting']
              nb_denying = Statistical[len(Statistical)-1]['OpinionDenying']
              totale_op = nb_supporting + nb_denying
              mvc_BetaD_Blocking_nodes_temprary_tolerance1(Graph,Nbr_Infected, p, nb_supporting/totale_op+1,K)
              mvc_BetaD_TRuth_comp(Graph,Nbr_Infected,p, nb_denying/totale_op,K)
              Liste_protector=Protector(Graph)
              taille=len(Liste_protector)
              for i in range(taille):
                 ListInfectedNodes.append(Liste_protector[i])
                 Graph.nodes[Liste_protector[i]]['Infetime'] =time
                 Graph.nodes[Liste_protector[i]]['opinion']=="denying"
                 Graph.nodes[Liste_protector[i]]['state']='spreaders'
                 Graph.nodes[Liste_protector[i]]['AccpR']+=1
                 Graph.nodes[Liste_protector[i]]['Accp_NegR']+=1
              bl=len(Liste_protector)+len(blocked_tempo(Graph))+1
          if (method=='newHybridApproach0'):
            
              nb_supporting = Statistical[len(Statistical)-1]['OpinionSupporting']
              nb_denying = Statistical[len(Statistical)-1]['OpinionDenying']
              totale_op = nb_supporting + nb_denying
              mvc_BetaD_Blocking_nodes_temprary_tolerance0(Graph, int(p*nb_supporting/totale_op))
              mvc_BetaD_TRuth_comp0(Graph, int(p*nb_denying/totale_op))
              Liste_protector=Protector(Graph)
              taille=len(Liste_protector)
              for i in range(taille):
                 ListInfectedNodes.append(Liste_protector[i])
                 Graph.nodes[Liste_protector[i]]['Infetime'] =time
                 Graph.nodes[Liste_protector[i]]['opinion']=="denying"
                 Graph.nodes[Liste_protector[i]]['state']='spreaders'
                 Graph.nodes[Liste_protector[i]]['AccpR']+=1
                 Graph.nodes[Liste_protector[i]]['Accp_NegR']+=1
              bl=len(Liste_protector)+len(blocked_tempo(Graph))+1
          

              
          elif method=='TCS':
              Random_TRuth_comp(Graph, p)
              Liste_protector=Protector(Graph)
              taille=len(Liste_protector)
             # print(taille)
              for i in range(taille):
                  Graph.nodes[Liste_protector[i]]['Infetime'] =time
                  Graph.nodes[Liste_protector[i]]['opinion']=="denying"
                  Graph.nodes[Liste_protector[i]]['state']='spreaders'
                  Graph.nodes[Liste_protector[i]]['AccpR']+=1
                  Graph.nodes[Liste_protector[i]]['Accp_NegR']+=1
                  ListInfectedNodes.append(Liste_protector[i])
              bl=len(Liste_protector)
          elif method=='TCSM':
             MaxDegree_TRuth_comp(Graph, p)
             Liste_protector=Protector(Graph)
             taille=len(Liste_protector)
             for i in range(taille):
                 ListInfectedNodes.append(Liste_protector[i])
                 Graph.nodes[Liste_protector[i]]['Infetime'] =time
                 Graph.nodes[Liste_protector[i]]['opinion']=="denying"
                 Graph.nodes[Liste_protector[i]]['state']='spreaders'
                 Graph.nodes[Liste_protector[i]]['AccpR']+=1
                 Graph.nodes[Liste_protector[i]]['Accp_NegR']+=1
             bl=len(Liste_protector)
          elif method=='TCSCen':
             Centrality_TRuth_comp(Graph, p)
             Liste_protector=Protector(Graph)
             taille=len(Liste_protector)
             for i in range(taille):
                 ListInfectedNodes.append(Liste_protector[i])
                 Graph.nodes[Liste_protector[i]]['Infetime'] =time
                 Graph.nodes[Liste_protector[i]]['opinion']=="denying"

                 Graph.nodes[Liste_protector[i]]['state']='spreaders'
                 Graph.nodes[Liste_protector[i]]['AccpR']+=1
                 Graph.nodes[Liste_protector[i]]['Accp_NegR']+=1
             bl=len(Liste_protector)
          elif method=='TCSBeta':
             Beta_TRuth_comp(Graph, p)
             Liste_protector=Protector(Graph)
             taille=len(Liste_protector)
             for i in range(taille):
                 ListInfectedNodes.append(Liste_protector[i])
                 Graph.nodes[Liste_protector[i]]['Infetime'] =time
                 Graph.nodes[Liste_protector[i]]['opinion']=="denying"
                 Graph.nodes[Liste_protector[i]]['state']='spreaders'
                 Graph.nodes[Liste_protector[i]]['AccpR']+=1
                 Graph.nodes[Liste_protector[i]]['Accp_NegR']+=1
             bl=len(Liste_protector)
          elif method=='TCSBetaD':
             BetaD_TRuth_comp(Graph, p)
             Liste_protector=Protector(Graph)
             taille=len(Liste_protector)
             for i in range(taille):
                 ListInfectedNodes.append(Liste_protector[i])
                 Graph.nodes[Liste_protector[i]]['Infetime'] =time
                 Graph.nodes[Liste_protector[i]]['opinion']=="denying"
                 Graph.nodes[Liste_protector[i]]['state']='spreaders'
                 Graph.nodes[Liste_protector[i]]['AccpR']+=1
                 Graph.nodes[Liste_protector[i]]['Accp_NegR']+=1
             bl=len(Liste_protector)
          elif method=='myalgo2':
             mvc_BetaD_Blocking_nodes(Graph,int(p/2))
             mvc_BetaD_TRuth_comp0(Graph, int(p/2))
             Liste_protector=Protector(Graph)
             taille=len(Liste_protector)
             for i in range(taille):
                 ListInfectedNodes.append(Liste_protector[i])
                 Graph.nodes[Liste_protector[i]]['Infetime'] =time
                 Graph.nodes[Liste_protector[i]]['opinion']=="denying"
                 Graph.nodes[Liste_protector[i]]['state']='spreaders'
                 Graph.nodes[Liste_protector[i]]['AccpR']+=1
                 Graph.nodes[Liste_protector[i]]['Accp_NegR']+=1
             bl=len(Liste_protector)+len(blocked(Graph))+1
          elif method=='hosniSpringer':
             ranking_list=[]
             #ranking_list.extend(rankingForHybridStrategy(Graph, time))
             #ranking_list.extend(rankingForTcStrategy(Graph, time))
             rankingForHybridStrategy(Graph, p, time)
             Liste_protector=Protector(Graph)
             taille=len(Liste_protector)
             for i in range(taille):
                 ListInfectedNodes.append(Liste_protector[i])
                 Graph.nodes[Liste_protector[i]]['Infetime'] =time
                 Graph.nodes[Liste_protector[i]]['opinion']=="denying"
                 Graph.nodes[Liste_protector[i]]['state']='spreaders'
                 Graph.nodes[Liste_protector[i]]['AccpR']+=1
                 Graph.nodes[Liste_protector[i]]['Accp_NegR']+=1
             bl=len(blocked(Graph))+len(Liste_protector)
          elif method=='hosniSpringer2':
             ranking_list=[]
             #ranking_list.extend(rankingForHybridStrategy(Graph, time))
             #ranking_list.extend(rankingForTcStrategy(Graph, time))
             rankingForTcStrategy(Graph, p, time)

             Liste_protector=Protector(Graph)
             taille=len(Liste_protector)
             for i in range(taille):
                 ListInfectedNodes.append(Liste_protector[i])
                 Graph.nodes[Liste_protector[i]]['Infetime'] =time
                 Graph.nodes[Liste_protector[i]]['opinion']=="denying"
                 Graph.nodes[Liste_protector[i]]['state']='spreaders'
                 Graph.nodes[Liste_protector[i]]['AccpR']+=1
                 Graph.nodes[Liste_protector[i]]['Accp_NegR']+=1
             bl=len(Liste_protector)
          #print(method," At time:", time, "blocked nodes Nbr:", bl)
            
              
      time += 0.25;   
      #update blockingtime of each node:
      for i in range(len(Graph.nodes)):
            if Graph.nodes[i]['Blockingtime'] > 0:
                Graph.nodes[i]['Blockingtime']-=0.25
def InitParameters(Graph,parameters):
    #Individual back ground knowledge:Beta
    #Forgetting and remembering factore:Omega
    #Hesitating factore:Deleta
    #Subjective judjement:Jug
  
    for node in Graph.nodes:
       Graph.nodes[node]['omega']=Inclusive(parameters[0]['omega_min'],parameters[0]['omega_max'])
       Graph.nodes[node]['beta']=Inclusive(parameters[0]['beta_min'],parameters[0]['beta_max'])
       Graph.nodes[node]['delta']=Inclusive(parameters[0]['delta_min'],parameters[0]['delta_max'])
       Graph.nodes[node]['jug']=Inclusive(parameters[0]['Jug_min'],parameters[0]['Jug_max'])
def Inclusive(min,max):
   
   b= ((np.random.random_sample()*(max - min )) + min)
    
   return b
def updateOpinion(jug,Accpet_NegR,Nbr_OF_R,Role): 
    if(Role=='True'):
       return 'denying' 
   
    opinion=jug
    if Accpet_NegR != 0:
        opinion*=(Accpet_NegR / Nbr_OF_R)
   
    
    if(np.random.random_sample()<= opinion):
        return 'denying'
    else:
        return 'supporting'

def graphe_TO_json(g):
    
    data =  json_graph.node_link_data(g,{"link": "links", "source": "source", "target": "target","weight":"weight"})
    data['nodes'] = [ {"id": i,
                       "state":"non_infected",
                       "Protector":"false",
                       "opinion":"normal",
                       "beta":0,
                       "omega":0,
                       "delta":0,
                       "jug":0,
                       "Infetime":0,
                       "AccpR":0,
                       "SendR":0,
                       "Accp_NegR":0,
                       "value":0,
                       "blocked":'false',
                       "p0_drimux":random.random(),
                       "BlockedTemporary":'false',
                       "Blockingtime":0,
                       "degree":g.degree[i],
                       "neighbors":[n for n in g.neighbors(i)]} for i in range(len(data['nodes'])) ]
    data['links'] = [ {"source":u,
                       "target":v,
                       "weight":(g.degree[u]+g.degree[v])/2} for u,v in g.edges ]
    return data

def geneList_Infectede(Listinfected,Listopinion,N,percentage):
    #10% of Popularity is infected 
    Nbr_OF_ndodesI=int(N*percentage/100)
    L=list(range(N))
    List=random.sample(L, Nbr_OF_ndodesI)
    opinion=np.random.uniform(0,1,Nbr_OF_ndodesI)
    for each in range(Nbr_OF_ndodesI):
        Listinfected.append(List[each])
        if opinion[each]<=0.2:
           Listopinion.append('denying')
        else:
            Listopinion.append('supporting')
           
def parameters(parameter,stepBeta=1,Beta=0.2,stepOmega=5.2,Omega=math.pi/3,stepDelta=0.65,Delta=math.pi/24,stepJug=0.6,Jug=0.1):
    Beta_max=Beta+stepBeta
    Omega_max=Omega +stepOmega
    Delta_max=Delta +stepDelta
    Jug_max=Jug+stepJug
    parameter.append({'beta_min':round(Beta,2),'beta_max':round(Beta_max,2),'omega_min':round(Omega,2),'omega_max':round(Omega_max,2),'delta_min':round(Delta,2),'delta_max':round(Delta_max,2),'Jug_min':round(Jug,2),'Jug_max':round(Jug_max,2)})

def Start(i,Graph,parameter,Stat,percentage,K,Tdet,method):
    #print("The ", i+1, "th simulation")
    for each in range(len(Graph.nodes)):
        Graph.nodes[each]['opinion']="normal"
        Graph.nodes[each]['Infetime']=0 
        Graph.nodes[each]['state']='non_infected'
        Graph.nodes[each]['Protector']='false'
        Graph.nodes[each]['blocked']='false'
        
    Statistical=[]
    ListInfected=[]
    Listopinion=[]
    #X% of Popularity is infected 
    geneList_Infectede(ListInfected,Listopinion,len(Graph.nodes),percentage)
   
    HISBmodel(Graph,ListInfected,Listopinion,Statistical,parameter,K,Tdet,method)  
    Stat.append(Statistical)    
    
    
def globalStat(S,Stat_Global,parameter,method):
    max=0
    Stat=[]
    for each in S:
        
        L=len(each)
        Stat.append(each)
        if(L>max):
            max=L
    for i in range(len(Stat)):
        L=len(Stat[i])
        Nbr_nonInfected=Stat[i][L-1]['NonInfected']
        Nbr_Infected=Stat[i][L-1]['Infected']
        Nbr_Spreaders=Stat[i][L-1]['Spreaders']
        OpinionDenying=Stat[i][L-1]['OpinionDenying']
        OpinionSupporting=Stat[i][L-1]['OpinionSupporting']
        RumorPopularity=Stat[i][L-1]['RumorPopularity']
        for j in range(L,max):
            Stat[i].append({'NonInfected':Nbr_nonInfected,'Infected':Nbr_Infected,'Spreaders':Nbr_Spreaders,'OpinionDenying':OpinionDenying,'OpinionSupporting':OpinionSupporting,'RumorPopularity':RumorPopularity,'graph':0})       

    y1=[]
    y2=[]
    y3=[]
    y4=[]
    y5=[]   
    Len=len(Stat)
  
    for i in range(max):
        
        Infected=0
        Spreaders=0
        RumorPopularity=0
        OpinionDenying=0
        OpinionSupporting=0
        for each in Stat:           
            Infected+=(each[i]['Infected'])
            Spreaders+=(each[i]['Spreaders'])
            RumorPopularity+=(each[i]['RumorPopularity'])
            OpinionDenying+=(each[i]['OpinionDenying'])
            OpinionSupporting+=(each[i]['OpinionSupporting'])
        y1.append(Infected/Len)
        y2.append(Spreaders/Len)
        y3.append(RumorPopularity/Len)
        y4.append(OpinionDenying/Len)
        y5.append(OpinionSupporting/Len)
     

    Stat_Global.append({'Infected':y1,'Spreaders':y2,'RumorPopularity':y3,'OpinionDenying':y4,'OpinionSupporting':y5,'parameter':parameter,'max':max,'method':method})       
    #Number of nodes

def Display(Stat_Global,xx,title_fig,nb):
   #print(Stat_Global)
    #Title=['BNLSBetaD','BNLSBetaD_time','BNLSBetaD_time_mvc','BNLSBetaD_mvc','BNLSBeta_time_tolerance_mcv','TCSBetaD','myalgo2','NP']
    Title=['NP','BNSWijayanto','TCSCen','BNLSCen','BLS','AGPUX']
    max=0
    Stat=[]
    Infected=[]
    para=[]
    for each in Stat_Global:
        L=each['max']
        #print(L,each['Infected'])
        
        para.append(each['method'])
        metho=str(each['method'])
        if metho.startswith('TCS'):
            Infected.append(each['OpinionSupporting'][L-1]/Nodes)
        else:
            Infected.append(each['Infected'][L-1]/Nodes)
                
        
        Stat.append(each)
        if(L>max):
            max=L
    for each in Stat:
        L=each['max']
        if (L<max):
            Nbr_Infected=each['Infected'][L-1]
            Nbr_Spreaders=each['Spreaders'][L-1]
            OpinionDenying=each['OpinionDenying'][L-1]
            OpinionSupporting=each['OpinionSupporting'][L-1]
            RumorPopularity=each['RumorPopularity'][L-1]
            for j in range(L,max):
                each['Infected'].append(Nbr_Infected)
                each['Spreaders'].append(Nbr_Spreaders)
                each['OpinionDenying'].append(OpinionDenying)
                each['OpinionSupporting'].append(OpinionSupporting)
                each['RumorPopularity'].append(RumorPopularity)
    
    pro=int(max/50)
    
    for each in Stat:
            for j in reversed(range(max)):

                d=j%pro
                if(d!=0):
                    each['Infected'].pop(j)
                    each['Spreaders'].pop(j)
                    each['OpinionDenying'].pop(j)
                    each['OpinionSupporting'].pop(j)
                    each['RumorPopularity'].pop(j)

    for each in Stat:
            for j in reversed(range(10)):
                    each['Infected'].pop(40+j)
                    each['Spreaders'].pop(40+j)
                    each['OpinionDenying'].pop(40+j)
                    each['OpinionSupporting'].pop(40+j)
                    each['RumorPopularity'].pop(20+j)
    x = range(0,len(Stat[0]['Infected']))
    x=np.array(x)*pro
    

    # plot 
    
    type=['x','*','p','8','h','H','.','+','4','1','2','3','4','5']
    
    #Infected
    plt.figure(num=xx)
    plt.subplot()
    #k="{}:{},{}]" 
    k="{}" 
    for infected,j in zip( Stat,range(len(Stat))):
      quotients = [number /Nodes  for number in infected["Infected"]]
      plt.plot(x,quotients,marker=type[j],markersize=7,linewidth=1,label=k.format(Title[j]))
    plt.legend(fontsize=12) 

    plt.xlabel('Temps',fontsize=10)
    plt.ylabel('Nombre des individues')
    plt.grid(True)
    plt.savefig('fig/infected.pdf',dpi=50)
    # RumorPopularity
    xx+=1
    plt.figure(num=xx)
    plt.subplot()
    #k="{}:{},{}]" 
    k="{}" 
    for infected,j in zip( Stat,range(len(Stat))):
      quotients = [number /Nodes  for number in infected["RumorPopularity"]]
      plt.plot(x, quotients,marker=type[j],markersize=6,linewidth=1,label=k.format(Title[j]))
    plt.legend(fontsize=12) 
    plt.xlabel('Temps')
    plt.ylabel('Nombre des individues')
    plt.grid(True)
    plt.title("popularity")
    plt.savefig('fig/RumorPopularity.pdf',dpi=20)
    
    #Spreaders
    xx+=1
    plt.figure(num=xx)
    plt.subplot()
    #k="{}:{},{}]" 
    k="{}" 
    for infected ,j in zip( Stat,range(len(Stat))):
      quotients = [number /Nodes  for number in infected["Spreaders"]]
      plt.plot(x, quotients,marker=type[j],markersize=6,linewidth=1,label=k.format(Title[j]))
    
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.title("Spreaders")
    plt.xlabel('Temps')
    plt.ylabel('Nombre des individues')
    plt.savefig('fig/Spreaders.pdf',dpi=20)
   
    
   # # Opinion
   #  xx+=1
   #  plt.figure(num=xx)
   #  plt.subplot()
   #  #k="{}:{},{}]" 
   #  k="{}" 
   #  for infected,j in zip( Stat,range(len(Stat))):
   #    quotients = [number /Nodes  for number in infected["OpinionDenying"]]
      
   #    plt.plot(x, quotients,marker=type[j],markersize=6,linewidth=2,label=k.format(Title[j]))
   #  plt.legend(fontsize=12) 
   #  plt.grid(True)
   #  plt.xlabel('Temps')
   #  plt.ylabel('Nombre des individues')
   #  plt.savefig('fig/OpinionDenying.pdf',dpi=20)
    

    # Opinion
    xx+=1
    plt.figure(num=xx)
    plt.subplot()
    #k="{}:{},{}]" 
    k="{}" 
    for infected,j in zip( Stat,range(len(Stat))):
      quotients = [number /Nodes  for number in infected["OpinionSupporting"]]
      plt.plot(x, quotients,marker=type[j],markersize=6,linewidth=2,label=k.format(Title[j]))
   
    plt.legend(fontsize=12) 
    plt.grid(True)
    plt.xlabel('Temps')
    plt.ylabel('Nombre des individues')
    plt.title("Supporting")
    plt.savefig('fig/OpinionSupporting.pdf',dpi=20)
    
    # Format the minor tick labels of the y-axis into empty strings with
    # `NullFormatter`, to avoid cumbering the axis with too many labels.

    xx+=1
    plt.figure(num=xx) 
    plt.subplot()      
    plt.plot(para,Infected,'bo')
    plt.grid(True)
    plt.xlabel(Title)
    plt.title("infected")
    plt.ylabel('Nombre des individues')
    plt.savefig('fig/nodes.pdf',dpi=20) 
    
def Simulation(index,graph,Stat_Global,percentage):
     Beta=0.2
     with Manager() as manager:
        Stat=manager.list()  
        parameter=[]
        parameters(parameter,Beta=Beta+index/10)
        start_time = time.time()  
        processes=[multiprocessing.Process(target=Start,args=(i,index,graph,parameter,Stat,percentage))for i in range(10)] 
        [process.start() for process in processes] 
        [process.join() for process in processes]
        end_time = time.time()
        #print("Parallel xx time=", end_time - start_time)
        globalStat(Stat,Stat_Global,parameter)
#gene graph

def Random_networks ( N=300 ,P=0.3):
    # Erdős-Rényi graph
    # number of nodes
    # expected number of partners
    
    g = nx.gnp_random_graph(N, P)  
    return graphe_TO_json(g)
def Small_World_networks(N=300,K=10,P=0.3):
    
    #Watts_strogatz graph
    #N=number of nodes
    #K=Each node is joined with its k nearest neighbors in a ring topology(anneau).
    #P=The probability of rewiring each edge(Probabilite de remplace les arretes)
    g= nx.watts_strogatz_graph(N,K,P)
    return graphe_TO_json(g)
def Scale_free_networks (N=300,M=10):
    #Barabasi_albert graph
    #N= Number of nodes
    #M= Number of edges to attach from a new node to existing nodes
    g=nx.barabasi_albert_graph(N,M)
    return graphe_TO_json(g)
def facebook_graph():
    FielName="facebook.txt"
    Graphtype=nx.Graph()
    g= nx.read_edgelist(FielName,create_using=Graphtype,nodetype=int)
    
    return graphe_TO_json(g) 
def search_spreaders(G,sp):
    
    l=len(G.nodes)
    for i in range (l):
        if ( G.nodes[i]['state']=='spreaders'):
          sp.append(i)
                
def neighbor(Spreaders,g):
    neighb=[]
    MaxD=[]
    Cente=[]
    beta=[]
    betaD=[]
    beta2D=[]
    Cent=((nx.degree_centrality(g)))
    
    for i in Spreaders:
        n=g.neighbors(i)
        
        for j in n:
          
          if g.nodes[j]['state'] =='non_infected':
              if j not in neighb :
                  neighb.append(j)
                  Cente.append(Cent[j])
                  MaxD.append(g.nodes[j]['degree'])
                  beta.append(g.nodes[j]['beta'])
                  betaD.append(g.nodes[j]['degree']/g.nodes[j]['beta'])
                  beta2D.append(g.nodes[j]['degree']*g.nodes[j]['beta'])
    return neighb,MaxD,Cente,beta,betaD,beta2D
def mvc_neighbor(Spreaders,g):
    neighb=[]
    MaxD=[]
    Cente=[]
    beta=[]
    betaD=[]
    beta2D=[]
    Cent=((nx.degree_centrality(g)))
    mvc=min_weighted_vertex_cover(g)
    for i in Spreaders:
        n=g.neighbors(i)
        
        for j in n:
          
          if g.nodes[j]['state'] =='non_infected'  and j in mvc:
              if j not in neighb and g.nodes[j]['Blockingtime'] == 0 and g.nodes[j]['blocked'] == 'false':
                  neighb.append(j)
                  Cente.append(Cent[j])
                  MaxD.append(g.nodes[j]['degree'])
                  beta.append(g.nodes[j]['beta'])
                  betaD.append(g.nodes[j]['degree']/g.nodes[j]['beta'])
                  beta2D.append(g.nodes[j]['degree']*g.nodes[j]['beta'])
           
              
   
    return neighb,MaxD,Cente,beta,betaD,beta2D
def simulation_strategy(x,K,k_ratio,Tdet,percentage,method,G):
   
        with Manager() as manager:
            Stat_Global=manager.list() 
            v=0
            for met in method : #the method of RIM
                #print(met)
                g=G[v]
                with Manager() as manager:
                    Stat=manager.list()  
                    parameter=[]
                    parameters(parameter)
                    start_time = time.time()  
                    processes=[multiprocessing.Process(target=Start,args=(i,g,parameter,Stat,percentage,K,Tdet,met))for i in range(NumOFsumi)] 
                    
                    [process.start() for process in processes] 
                    [process.join() for process in processes]
                    end_time = time.time() 
                    #print("Parallel xx time=", end_time - start_time)
                    globalStat(Stat,Stat_Global,parameter,met)
                    print("tDetection:",Tdet, ", for k=",k_ratio, ", the method:", met, ", the result:", Stat_Global[len(Stat_Global)-1]['OpinionSupporting'][len(Stat_Global[len(Stat_Global)-1]['OpinionSupporting'])-1]/Nodes)
                v+=1     
            #Display(Stat_Global,x,'NBLS',Nodes)
            
def Iterative():
    start_time = time.time()  
    StatI=[]

    for i in range(6):
        parameter=[]
        parameters(parameter,Omega=0.2+i/10)
        Stat=[]
        start_time1 = time.time() 
        for j in range(50):
            Start(i,j,g,parameter,Stat,percentage)
        end_time1 = time.time()
        #print("Serial xx time=", end_time1 - start_time1)
        globalStat(Stat,StatI,parameter)
    end_time = time.time()
    #print("Serial time=", end_time - start_time)
    Display(StatI)
def Random_Blocking_nodes(Graphe,k):
    sp=[]
    search_spreaders(Graphe,sp)
    nb,d,cen,Bet,betaD,beta2D=neighbor(sp,Graphe)
    size=len(nb)
    if k>size:
      k=size-1
    for i in range(k):
        s=random.randint(0, size-1)
        Graphe.nodes[nb[s]]['blocked']='True'
        nb.pop(s)
        size-=1
       
def Degree_MAX(G,K,nb):
    L=[]
    
  
    for i in range(len(nb)):
        L.append(G.nodes[i]['degree'])

    return L
def Degree_MAX_Blocking_nodes(G,k):
    
    sp=[]
   
    search_spreaders(G,sp)
   
    nb,DNode,cen,Bet,betaD,beta2D=neighbor(sp,G)
    

    for i in range(k):
            
            ID = DNode.index(max(DNode))
            G.nodes[nb[ID]]['blocked']='True'
            DNode.pop(ID)
            nb.pop(ID)
def wijayantoStrategy(G,k,k0):

    nb,DNode,cen,Bet,betaD,beta2D=mvc_neighbor(list(G.nodes),G)
    k_i = int(k0/15) #neworks have in average 15 snapshot according to wijayanto
    if int(k0/15) > k:
        k_i = k
    #print("k_i", k_i)
    for i in range(k_i):
            if len(DNode) == 0:
                break
            ID = DNode.index(max(DNode))
            G.nodes[nb[ID]]['blocked']='True'
            DNode.pop(ID)
            nb.pop(ID)
            
def Centrality_Blocking_nodes(G,k):
    
    #sp=[]
   
    #search_spreaders(G,sp)
    nb,DNode,cen,Bet,betaD,beta2D=nodes(list(G.nodes),G)
    for i in range(k):
            
            ID = cen.index(max(cen))
            G.nodes[nb[ID]]['blocked']='True'
            cen.pop(ID)
            nb.pop(ID)         
def nodes(nodes, g):

    MaxD=[]
    Cente=[]
    beta=[]
    betaD=[]
    beta2D=[]
    nb=[]
    Cent=((nx.degree_centrality(g)))
    for j in nodes:
     nb.append(j)
     Cente.append(Cent[j])
     MaxD.append(g.nodes[j]['degree'])
     beta.append(g.nodes[j]['beta'])
     betaD.append(g.nodes[j]['degree']/g.nodes[j]['beta'])
     beta2D.append(g.nodes[j]['degree']*g.nodes[j]['beta'])
    return nb,MaxD,Cente,beta,betaD,beta2D
def Beta_Blocking_nodes(G,k):
    
    sp=[]
   
    search_spreaders(G,sp)
   
    nb,DNode,cen,Bet,betaD,beta2D=neighbor(sp,G)
    

    for i in range(k):
            
            ID = Bet.index(min(Bet))
            G.nodes[nb[ID]]['blocked']='True'
            Bet.pop(ID)
            nb.pop(ID)         
def BetaD_Blocking_nodes(G,k):
    
    sp=[]
   
    search_spreaders(G,sp)
   
    nb,DNode,cen,Bet,betaD,beta2D=neighbor(sp,G)
    

    for i in range(k):
            
            ID = betaD.index(max(betaD))
            G.nodes[nb[ID]]['blocked']='True'
            betaD.pop(ID)
            nb.pop(ID)
def Blocking_links(G, k):
    #print("CalculateAverageDegree(G)", CalculateAverageDegree(G))
    #budget = k * CalculateAverageDegree(G)
    budget = int(k * len(G.edges)/ len(G.nodes))
    #print('budget:',budget)
    ListEdges = edge_betweenness_centrality(G, k=10, normalized=True, weight=None, seed=None)
    rankedListEdges = sorted(ListEdges.items(), key=lambda x: x[1], reverse=True)
    counter = 0
    for edge in rankedListEdges:
        if counter > budget:
            break
        G.remove_edge(edge[0][0], edge[0][1])
        remove_links(G,edge[0][0],edge[0][1])
        counter+=1
def CalculateAverageDegree(G):
    s=0
    for i in range(len(G.nodes)):
       s+=G.nodes[i]['degree'] 
    return s/len(G.nodes)/2

def BetaD_Blocking_nodes_temporary(G,k):
    
    sp=[]
   
    search_spreaders(G,sp)
   
    nb,DNode,cen,Bet,betaD,beta2D=neighbor(sp,G)
    

    for i in range(k):
            
            ID = betaD.index(max(betaD))
            G.nodes[nb[ID]]['Blockingtime']=5
            G.nodes[nb[ID]]['BlockedTemporary']='true'
            betaD.pop(ID)
            nb.pop(ID)
def DRIMUX(G, k, t):
    k_i = int(k/2)
    if k_i > k:
        k_i = k
    
    beta1 = random.random()
    beta2 = 1 - beta1
    ranked_listDRIMUX = {}
    for u in range(len(G.nodes)):
        #print("node",u)
        if G.nodes[u]['BlockedTemporary']=='false':   
            u_influence = 0
            for v in G.nodes[u]['neighbors']:    
                #calculate Pind
                RelativeTime = t - G.nodes[u]['Infetime'] 
                Psend = G.nodes[u]['p0_drimux']/ np.log(10+ RelativeTime)
                freedom = 4
                Pacc = 1/ G.nodes[u]['degree']
                Pind = Psend * Pacc
                Pglb = (pow(2, 1-freedom/2) * pow(RelativeTime, freedom-1) * np.exp(-RelativeTime*RelativeTime/2)) / gamma(freedom/2)

                Puv = 1/ (1 + np.exp(-(beta1*Pglb + beta2*Pind)))
                firstTerm = Puv
                secondTerm = 0
                for o in G.nodes[v]['neighbors']:
                    RelativeTime2 = t - G.nodes[o]['Infetime'] 
                    Psend = G.nodes[o]['p0_drimux']/ np.log(10+ RelativeTime2)
                    freedom = 4
                    Pacc = 1/ G.nodes[o]['degree']
                    Pind = Psend * Pacc
                    Pglb = (pow(2, 1-freedom/2) * pow(RelativeTime2, freedom-1) * np.exp(-RelativeTime2*RelativeTime2/2)) / gamma(freedom/2)
                    Pov = 1/ (1 + np.exp(-(beta1*Pglb + beta2*Pind)))
                    
                    secondTerm *= np.exp(-Pov)
                u_influence+= firstTerm * secondTerm

            
            
            ranked_listDRIMUX[u]= u_influence
        
    ranked_listDRIMUX = sorted(ranked_listDRIMUX.items(), key=lambda x: x[1], reverse=True)
    counter = 0
    for key in ranked_listDRIMUX:
        if counter > k_i:
            break
        G.nodes[key[0]]['Blockingtime']= G.nodes[key[0]]['degree']*49/1044
        G.nodes[key[0]]['BlockedTemporary']='true'
        counter+=1
def mvc_BetaD_Blocking_nodes_temprary_tolerance_adaptative(G,k):
    sp=[]
   
    search_spreaders(G,sp)
   
    nb,DNode,cen,Bet,betaD,beta2D=mvc_neighbor(sp,G)
    k_i=0
    if len(nb) > k:
        k_i = k
    else:
        k_i = len(nb)

    
    

    for i in range(k_i):
            ID = betaD.index(max(betaD))
            G.nodes[nb[ID]]['Blockingtime']=G.nodes[nb[ID]]['degree']*49/1044 # *(50-1)==time of the whole process/ (maxDeg-minDeg)==(1045-1)
            G.nodes[nb[ID]]['BlockedTemporary']='true'
            betaD.pop(ID)
            nb.pop(ID)   
def mvc_BetaD_Blocking_nodes_temprary_tolerance1(G,Nbr_Infected,k,ratio,k0):
    sp=[]
   
    search_spreaders(G,sp)
   
    nb,DNode,cen,Bet,betaD,beta2D=mvc_neighbor(sp,G)
    k_i = int(k0*ratio * len(nb)*Nbr_Infected/len(G.nodes)/len(G.nodes))
    if k_i > int(k*ratio):
        k_i = int(k*ratio)
    
    

    for i in range(k_i):
            if len(betaD) == 0:
                break
            ID = betaD.index(max(betaD))
            G.nodes[nb[ID]]['Blockingtime']=G.nodes[nb[ID]]['degree']*49/1044 # *(50-1)==time of the whole process/ (maxDeg-minDeg)==(1045-1)
            G.nodes[nb[ID]]['BlockedTemporary']='true'
            betaD.pop(ID)
            nb.pop(ID)   
def mvc_BetaD_Blocking_nodes_temprary_tolerance0(G,k):
    sp=[]
   
    search_spreaders(G,sp)
   
    nb,DNode,cen,Bet,betaD,beta2D=mvc_neighbor(sp,G)
    

    for i in range(k):
            if len(betaD) == 0:
                break
            ID = betaD.index(max(betaD))
            G.nodes[nb[ID]]['Blockingtime']=G.nodes[nb[ID]]['degree']*49/1044 # *(50-1)==time of the whole process/ (maxDeg-minDeg)==(1045-1)
            G.nodes[nb[ID]]['BlockedTemporary']='true'
            betaD.pop(ID)
            nb.pop(ID)   
def mvc_BetaD_Blocking_nodes(G,k):
    
    sp=[]
   
    search_spreaders(G,sp)
   
    nb,DNode,cen,Bet,betaD,beta2D=mvc_neighbor(sp,G)
    

    for i in range(k):
            
            ID = betaD.index(max(betaD))
            G.nodes[nb[ID]]['blocked']='True'
            betaD.pop(ID)
            nb.pop(ID)

def merge_two_dicts(x, y):
    z = x.copy()   # start with keys and values of x
    z.update(y)    # modifies z with keys and values of y/
    return z

def rankingForHybridStrategy(Graph, p, time):
    ranked_listBNLS = {}
    ranked_listTCS = {}
    for node in range(len(Graph.nodes)):
        #print(node)
        Gv_positive_node_nextStep = 0
        Fv_positive_node_currentStep = 0
        Gv_negative_node_nextStep = 0
        Fv_negative_node_currentStep = 0
        #calculate Gv_positive_node_nextStep
        RelativeTime = time - Graph.nodes[node]['Infetime'] 
        for neighbor in Graph.nodes[node]['neighbors']:
            Puv = np.exp(-(RelativeTime+0.25+0.25) * Graph.nodes[node]['beta']) * np.abs(np.sin(((RelativeTime+0.25+0.25) * Graph.nodes[node]['omega'] )+ Graph.nodes[node]['delta'])) * Graph.nodes[node]['degree']/ (Graph.nodes[node]['degree'] + Graph.nodes[neighbor]['degree'])*0.3
            
            Ro = 0
            if Graph.nodes[neighbor]['AccpR'] != 0:
                Ro = Graph.nodes[neighbor]['jug']* ( Graph.nodes[neighbor]['Accp_NegR'] / Graph.nodes[neighbor]['AccpR'])
            else:
                Ro = Graph.nodes[neighbor]['jug']
            PIv_negative = (Ro*Ro - 2*Ro +1)/(Ro*Ro + Ro +1)
            PIv_positive = 1 - PIv_negative
            firstTerm = PIv_positive * Puv
            firstTermTCS = PIv_negative * Puv
            secondTerm = 0
            secondTermTCS = 0
            for Nei_neighbor in Graph.nodes[node]['neighbors']:
                secondTerm+= PIv_positive * ( np.exp(-RelativeTime+0.25+0.25 * Graph.nodes[node]['beta']) * np.abs(np.sin((RelativeTime+0.25+0.25 * Graph.nodes[node]['omega'] )+ Graph.nodes[node]['delta'])) - np.exp(-RelativeTime+0.25 * Graph.nodes[node]['beta']) * np.abs(np.sin((RelativeTime+0.25 * Graph.nodes[node]['omega'] )+ Graph.nodes[node]['delta'])) )* Graph.nodes[node]['degree']/ (Graph.nodes[node]['degree'] + Graph.nodes[Nei_neighbor]['degree'])*0.3
                secondTermTCS+= PIv_negative * ( np.exp(-RelativeTime+0.25+0.25 * Graph.nodes[node]['beta']) * np.abs(np.sin((RelativeTime+0.25+0.25 * Graph.nodes[node]['omega'] )+ Graph.nodes[node]['delta'])) - np.exp(-RelativeTime+0.25 * Graph.nodes[node]['beta']) * np.abs(np.sin((RelativeTime+0.25 * Graph.nodes[node]['omega'] )+ Graph.nodes[node]['delta'])) )* Graph.nodes[node]['degree']/ (Graph.nodes[node]['degree'] + Graph.nodes[Nei_neighbor]['degree'])*0.3
            secondTerm = np.exp(-secondTerm)
            secondTermTCS = np.exp(-secondTermTCS)
            Gv_positive_node_nextStep+= (firstTerm * secondTerm)
            Gv_negative_node_nextStep+= (firstTermTCS * secondTermTCS)
            #calculate Fv_positive_node_currentStep and negative
            Pvu = np.exp(-(RelativeTime) * Graph.nodes[neighbor]['beta']) * np.abs(np.sin(((RelativeTime) * Graph.nodes[neighbor]['omega'] )+ Graph.nodes[neighbor]['delta'])) * Graph.nodes[neighbor]['degree']/ (Graph.nodes[neighbor]['degree'] + Graph.nodes[node]['degree'])*0.3
            ro = 0
            if Graph.nodes[node]['AccpR'] != 0:
                ro = Graph.nodes[node]['jug']* ( Graph.nodes[node]['Accp_NegR'] / Graph.nodes[node]['AccpR'])
            else:
                ro = Graph.nodes[node]['jug']
            PIv_negative = (ro*ro - 2*ro +1)/(ro*ro + ro +1)
            PIv_positive = 1 - PIv_negative
            firstTerm = PIv_positive * Pvu
            firstTermTCS = PIv_negative * Pvu
            secondTerm = 0
            secondTermTCS = 0
            for Nei_neighbor in Graph.nodes[node]['neighbors']:
                if neighbor != Nei_neighbor:   
                    secondTerm+= PIv_positive *  np.exp(-RelativeTime * Graph.nodes[neighbor]['beta']) * np.abs(np.sin((RelativeTime * Graph.nodes[neighbor]['omega'] )+ Graph.nodes[neighbor]['delta'])) * Graph.nodes[neighbor]['degree']/ (Graph.nodes[neighbor]['degree'] + Graph.nodes[Nei_neighbor]['degree'])*0.3
                    secondTermTCS+= PIv_positive *  np.exp(-RelativeTime * Graph.nodes[neighbor]['beta']) * np.abs(np.sin((RelativeTime * Graph.nodes[neighbor]['omega'] )+ Graph.nodes[neighbor]['delta'])) * Graph.nodes[neighbor]['degree']/ (Graph.nodes[neighbor]['degree'] + Graph.nodes[Nei_neighbor]['degree'])*0.3
            secondTerm = np.exp(-secondTerm)
            secondTermTCS = np.exp(-secondTermTCS)
            Fv_positive_node_currentStep+= (firstTerm * secondTerm)
            Fv_negative_node_currentStep+= (firstTermTCS * secondTermTCS)
        
        ranked_listBNLS[node]= Gv_positive_node_nextStep*Fv_positive_node_currentStep
        ranked_listTCS[node]= Gv_negative_node_nextStep*Fv_negative_node_currentStep
        
    # ranked_listBNLS = sorted(ranked_listBNLS.items(), key=lambda x: x[1], reverse=True)
    # ranked_listTCS = sorted(ranked_listTCS.items(), key=lambda x: x[1], reverse=True)
    l = merge_two_dicts(ranked_listBNLS, ranked_listTCS)
    rankedList = sorted(l.items(), key=lambda x: x[1], reverse=True)
    counter = 0
    for key in rankedList:
        if counter > p:
            break
        if ranked_listBNLS[key[0]] == key[1]:
            Graph.nodes[key[0]]['blocked'] = 'True'
        else:
            Graph.nodes[key[0]]['Protector']='True'
            Graph.nodes[key[0]]['state']='infected'
        counter+=1
    #print(ranked_listTCS)

def rankingForTcStrategy(Graph,p, time):
    ranked_listTCS = {}
    for node in range(len(Graph.nodes)):
        #print(node)
        Gv_negative_node_nextStep = 0
        Fv_negative_node_currentStep = 0
        #calculate Gv_positive_node_nextStep
        RelativeTime = time - Graph.nodes[node]['Infetime'] 
        for neighbor in Graph.nodes[node]['neighbors']:
            Puv = np.exp(-(RelativeTime+0.25+0.25) * Graph.nodes[node]['beta']) * np.abs(np.sin(((RelativeTime+0.25+0.25) * Graph.nodes[node]['omega'] )+ Graph.nodes[node]['delta'])) * Graph.nodes[node]['degree']/ (Graph.nodes[node]['degree'] + Graph.nodes[neighbor]['degree'])*0.3
            PIv_negative = 1
            if Graph.nodes[neighbor]['AccpR'] != 0:
                PIv_negative = Graph.nodes[neighbor]['jug']* ( Graph.nodes[neighbor]['Accp_NegR'] / Graph.nodes[neighbor]['AccpR'])
            PIv_positive = 1 - PIv_negative
            firstTermTCS = PIv_negative * Puv
            secondTermTCS = 0
            for Nei_neighbor in Graph.nodes[node]['neighbors']:
                secondTermTCS+= PIv_negative * ( np.exp(-RelativeTime+0.25+0.25 * Graph.nodes[node]['beta']) * np.abs(np.sin((RelativeTime+0.25+0.25 * Graph.nodes[node]['omega'] )+ Graph.nodes[node]['delta'])) - np.exp(-RelativeTime+0.25 * Graph.nodes[node]['beta']) * np.abs(np.sin((RelativeTime+0.25 * Graph.nodes[node]['omega'] )+ Graph.nodes[node]['delta'])) )* Graph.nodes[node]['degree']/ (Graph.nodes[node]['degree'] + Graph.nodes[Nei_neighbor]['degree'])*0.3
            secondTermTCS = np.exp(-secondTermTCS)
            Gv_negative_node_nextStep+= (firstTermTCS * secondTermTCS)
            #calculate Fv_positive_node_currentStep and negative
            Pvu = np.exp(-(RelativeTime) * Graph.nodes[neighbor]['beta']) * np.abs(np.sin(((RelativeTime) * Graph.nodes[neighbor]['omega'] )+ Graph.nodes[neighbor]['delta'])) * Graph.nodes[neighbor]['degree']/ (Graph.nodes[neighbor]['degree'] + Graph.nodes[node]['degree'])*0.3
            PIv_negative = 1
            if Graph.nodes[node]['AccpR'] != 0:
                PIv_negative = Graph.nodes[node]['jug']* ( Graph.nodes[node]['Accp_NegR'] / Graph.nodes[node]['AccpR'])
            PIv_positive = 1 - PIv_negative
            firstTermTCS = PIv_negative * Pvu
            secondTermTCS = 0
            for Nei_neighbor in Graph.nodes[node]['neighbors']:
                if neighbor != Nei_neighbor:   
                    secondTermTCS+= PIv_positive *  np.exp(-RelativeTime * Graph.nodes[neighbor]['beta']) * np.abs(np.sin((RelativeTime * Graph.nodes[neighbor]['omega'] )+ Graph.nodes[neighbor]['delta'])) * Graph.nodes[neighbor]['degree']/ (Graph.nodes[neighbor]['degree'] + Graph.nodes[Nei_neighbor]['degree'])*0.3
            secondTermTCS = np.exp(-secondTermTCS)
            Fv_negative_node_currentStep+= (firstTermTCS * secondTermTCS)
        
        ranked_listTCS[node]= Gv_negative_node_nextStep*Fv_negative_node_currentStep
        
    ranked_listTCS = sorted(ranked_listTCS.items(), key=lambda x: x[1], reverse=True)
    counter = 0
    for key in ranked_listTCS:
        if counter > p:
            break
        Graph.nodes[key[0]]['Protector']='True'
        Graph.nodes[key[0]]['state']='infected'
        counter+=1
    print(ranked_listTCS)

def tcsHISB(Graph, time):
    ranked_listTCS = {}
    for node in range(len(Graph.nodes)):
        #print(node)
        Fv_t = 0
        #calculate Gv_positive_node_nextStep
        RelativeTime = time - Graph.nodes[node]['Infetime'] 
        for neighbor in Graph.nodes[node]['neighbors']:
            Puv = np.exp(-(RelativeTime+0.25) * Graph.nodes[node]['beta']) * np.abs(np.sin(((RelativeTime+0.25) * Graph.nodes[node]['omega'] )+ Graph.nodes[node]['delta'])) * Graph.nodes[node]['degree']/ (Graph.nodes[node]['degree'] + Graph.nodes[neighbor]['degree'])*0.3
           
            firstTermTCS = Puv
            secondTermTCS = 0
            for Nei_neighbor in Graph.nodes[node]['neighbors']:
                secondTermTCS+= ( np.exp(-RelativeTime+0.25 * Graph.nodes[node]['beta']) * np.abs(np.sin((RelativeTime+0.25 * Graph.nodes[node]['omega'] )+ Graph.nodes[node]['delta'])) - np.exp(-RelativeTime * Graph.nodes[node]['beta']) * np.abs(np.sin((RelativeTime * Graph.nodes[node]['omega'] )+ Graph.nodes[node]['delta'])) )* Graph.nodes[node]['degree']/ (Graph.nodes[node]['degree'] + Graph.nodes[Nei_neighbor]['degree'])*0.3
            secondTermTCS = np.exp(-secondTermTCS)
            Fv_t+= (firstTermTCS * secondTermTCS)
        
        
        ranked_listTCS[node]= Fv_t
        
    ranked_listTCS = sorted(ranked_listTCS.items(), key=lambda x: x[1], reverse=True)
    counter = 0
    for key in ranked_listTCS:
        if counter > 806:
            break
        Graph.nodes[key[0]]['Protector']='True'
        Graph.nodes[key[0]]['state']='infected'
        counter+=1
    #print(ranked_listTCS)



def mvc_BetaD_Blocking_nodes_temprary(G,k):
    
    sp=[]
   
    search_spreaders(G,sp)
   
    nb,DNode,cen,Bet,betaD,beta2D=mvc_neighbor(sp,G)
    

    for i in range(k):
            
            ID = betaD.index(max(betaD))
            G.nodes[nb[ID]]['Blockingtime']=5
            G.nodes[nb[ID]]['BlockedTemporary']='true'
            betaD.pop(ID)
            nb.pop(ID)   

def Random_TRuth_comp(Graphe,k):
    sp=[]
    search_spreaders(Graphe,sp)
    nb,d,cen,Bet,betaD,beta2D=neighbor(sp,Graphe)
    size=len(nb)
    if k > size :
       k=size-1
    for i in range(k):
        s=random.randint(0, size-1)
        Graphe.nodes[nb[s]]['Protector']='True'
        Graphe.nodes[nb[s]]['state']='infected'
        nb.pop(s)
        size-=1
def MaxDegree_TRuth_comp(Graphe,K):
    sp=[]
    search_spreaders(Graphe,sp)
    nb,d,cen,Bet,betaD,beta2D=neighbor(sp,Graphe)
    size=len(nb)
    k=K
    if k > size :
       k=size-1
    for i in range(k):
        s = d.index(max(d))
        Graphe.nodes[nb[s]]['Protector']='True'
        Graphe.nodes[nb[s]]['state']='infected'
        nb.pop(s)
        d.pop(s)
def Centrality_TRuth_comp(Graphe,K):
    #sp=[]
    #search_spreaders(Graphe,sp)
   
    nb,d,cen,Bet,betaD,beta2D=nodes(list(Graphe.nodes),Graphe)
   
    size=len(nb)
    k=K
    # if k > size :
    #    k=size-1
    for i in range(k):
        s = cen.index(max(cen))
        Graphe.nodes[nb[s]]['Protector']='True'
        Graphe.nodes[nb[s]]['state']='infected'
        nb.pop(s)
        cen.pop(s)
def Beta_TRuth_comp(Graphe,K):
    sp=[]
    search_spreaders(Graphe,sp)
    nb,d,bet,cen,betaD,beta2D=neighbor(sp,Graphe)
    size=len(nb)
    k=K
    if k > size :
       k=size-1
    for i in range(k):
        s = cen.index(min(cen))
        Graphe.nodes[nb[s]]['Protector']='True'
        Graphe.nodes[nb[s]]['state']='infected'
        nb.pop(s)
        cen.pop(s)
def BetaD_TRuth_comp(Graphe,K):
    sp=[]
    search_spreaders(Graphe,sp)
    nb,d,bet,cen,betaD,beta2D=neighbor(sp,Graphe)
    size=len(nb)
    k=K
    if k > size :
       k=size-1
    for i in range(k):
        s = beta2D.index(max(beta2D))
        Graphe.nodes[nb[s]]['Protector']='True'
        Graphe.nodes[nb[s]]['state']='infected'
        nb.pop(s)
        beta2D.pop(s)

def mvc_BetaD_TRuth_comp0(Graphe,K):
    sp=[]
    search_spreaders(Graphe,sp)
    nb,d,bet,cen,betaD,beta2D=mvc_neighbor(sp,Graphe)
    size=len(nb)
    k=K
    if k > size :
       k=size-1
    for i in range(k):
        s = beta2D.index(max(beta2D))
        Graphe.nodes[nb[s]]['Protector']='True'
        Graphe.nodes[nb[s]]['state']='infected'
        nb.pop(s)
        beta2D.pop(s)
def mvc_BetaD_TRuth_comp(Graphe,Nbr_Infected,k,ratio,k0):
    sp=[]
    search_spreaders(Graphe,sp)
    nb,d,bet,cen,betaD,beta2D=mvc_neighbor(sp,Graphe)
    k_i = int(k0*ratio * len(nb)*Nbr_Infected/len(Graphe.nodes)/len(Graphe.nodes))
    if k_i > int(k*ratio):
        k_i = int(k*ratio)
    size=len(nb)

    if k_i > int(size*ratio) :
       k_i=int(size*ratio)
    for i in range(k_i):
        s = beta2D.index(max(beta2D))
        Graphe.nodes[nb[s]]['Protector']='True'
        Graphe.nodes[nb[s]]['state']='infected'
        nb.pop(s)
        beta2D.pop(s)


def blocked(G):
    
    L=[]
    for i in range (len(G.nodes)):
        if(G.nodes[i]['blocked']=='True'):
            L.append(i)           
    return L
def blocked_tempo(G):
    
    L=[]
    for i in range (len(G.nodes)):
        if(G.nodes[i]['BlockedTemporary'] == 'true'):
            L.append(i)           
    return L
def Protector(G):  
    
    L=[]
    for i in range (len(G.nodes)):
        if(G.nodes[i]['Protector']=='True'):
            L.append(i)           
    return L


if __name__ == '__main__':
       # use net.Graph() for undirected graph

# How to read from a file. Note: if your egde weights are int, 
# change float to int.
   
    #Graph's Parametres 
    P=0.3
    K=100
    M=20
    nbb=0
   
    #g=json_graph.node_link_graph(Scale_free_networks(500,M))
    #g=json_graph.node_link_graph(Small_World_networks(2000,K,P))
    #g=json_graph.node_link_graph(Random_networks(Nodes,P))
    g=json_graph.node_link_graph(facebook_graph())
    
    #print(g.nodes[12]['neighbors'])
    G=[]
    
    #m=['BNLSBetaD','BNLSBetaD_time','BNLSBetaD_time_mvc','BNLSBetaD_mvc','BNLSBeta_time_tolerance_mcv','TCSBetaD','myalgo2','NP']
    m=['NP','BNSWijayanto','TCSCen','BNLSCen','BLS','AGPUX']
    #m=['DRIMUX']
    for i in range(len(m)):
        G.append(g)

    
    
    Nodes=len(g.nodes)
    static="Nodes :{},Edegs:{}."
    percentage=0.1 #1% of popularity" is infected 
    NumOFsumi=1
    beta=0.20
    omega=0 
    juge=0.1
    delta=0
    K=int(Nodes*0.05)
    Tdet=1
    
   
  

    for Tdetection in [1, 5, 10, 15]:
        for k_ratio in [0.05, 0.10, 0.15, 0.20]:
            K=int(Nodes*k_ratio)
            simulation_strategy(1,  K, k_ratio, Tdetection,percentage, m,G)
    # simulation_strategy(1,  K, Tdet,percentage, m,G)
    # plt.show()
    #rankingForHybridStrategy(g, 0.125)
    # min = 0
    # max = 0
    # for n in g.nodes:
    #     if n['degree'] > max:
    #         max = n['degree']
    #     elif n['degree'] < min:
    #         min = n['degree']
    # print(max)
    # print(min)
