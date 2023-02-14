import numpy as np
import networkx as nx
import pandas as pd
import random 
import matplotlib.pyplot as plt
import matplotlib
plt.style.use('ggplot')








class HSIBmodel():
    def __init__(self,Graph,Seed_Set,opinion_set,baisAccepte=0.3, setptime=0.125, Probability=0.3,Tdet=np.Infinity,method='none'):
        self.setptime=setptime
        self.Graph=Graph
        self.time=0.125
        self.Probability=0.3
        self.ListInfectedNodes=Seed_Set
        RumorPopularity=self.SetParameters(opinion_set)
        self.baisAccepte=baisAccepte
        self.Statistical=pd.DataFrame(data={'NonInfected': self.Nbr_nonInfected,
                                       'Infected': self.Nbr_Infected,
                                       'Spreaders': self.Nbr_Infected,
                                       'OpinionDenying': self.OpinionDenying,
                                       'OpinionSupporting': self.OpinionSupporting,
                                       'RumorPopularity': RumorPopularity
                                       },index=[0])

    def DisplyResults(self):
        fig, axe= plt.subplots(2,3)
        col=self.Statistical.columns
        for i, ax in enumerate(axe.flat):
            ax.plot(self.Statistical.index,self.Statistical[col[i]])
            ax.set_title(f'The evolution of {col[i]}')
            ax.set_ylabel(f'Number of {col[i]}')
            ax.set_xlabel(f'Time')
    
        #     plt.subplot(2,3,idx)
        #     self.Statistical[idx].plot()
        #     axe[:,idx].set_ylabel('YLabel', loc='top')
        plt.show()
    
    def SetParameters(self,opinion_set):
        self.Nbr_Infected=len(self.ListInfectedNodes)
        
        self.Nbr_nonInfected=self.Graph.number_of_nodes()-self.Nbr_Infected
        self.OpinionDenying=0
        self.OpinionSupporting=0
        RumorPopularity=0
        for i,each  in enumerate(self.ListInfectedNodes):
            self.Graph.nodes[each]['Infetime']=0.125 
            self.Graph.nodes[each]['state']='spreaders'
            self.Graph.nodes[each]['AccpR']+=1
            RumorPopularity+=self.Graph.degree(each)
            
            if (opinion_set[i]=='D'):

                self.Graph.nodes[each]['opinion']='D'
                self.Graph.nodes[each]['Accp_NegR']+=1
                self.OpinionDenying+=1
            else:
                self.Graph.nodes[each]['opinion']='S'
                self.OpinionSupporting+=1
          
        return RumorPopularity

    def updateOpinion(self,id,jugf,NegR,R): 
        opinion=jugf
        assert R>0
        if NegR != 0:
            opinion*=float(NegR / R)
        if (self.Graph.nodes[id]['opinion']=="S"):
            self.OpinionSupporting-=1
        else:
            self.OpinionDenying-=1 

        if(np.random.random_sample()<= opinion):
            self.Graph.nodes[id]['opinion']="D"
        else:
            self.Graph.nodes[id]['opinion']="S"
        
        if (self.Graph.nodes[id]['opinion']=="S"):
            self.OpinionSupporting+=1
        else:
            self.OpinionDenying+=1 

    def simulation(self):
        time=0.125
        
        while self.ListInfectedNodes:  
            RumorPopularity = 0
            Nbr_Spreaders = 0
            for index, id in reversed(list(enumerate(self.ListInfectedNodes))):
                RelativeTime = time - self.Graph.nodes[id]['Infetime']
                
                if (np.exp(-RelativeTime * self.Graph.nodes[id]['beta']) < 0.10) :
                    self.ListInfectedNodes.remove(id)
                    self.Graph.nodes[id]['state'] = "infected"

                else:
                    #Node Attaction 
                    ActualAttraction = np.exp(-RelativeTime * self.Graph.nodes[id]['beta']) * np.abs(np.sin((RelativeTime * self.Graph.nodes[id]['omega'] )+ self.Graph.nodes[id]['delta']))
                    RumorPopularity += ActualAttraction * self.Graph.degree(id)
                    #Update node opinion
                    self.updateOpinion( id,
                                        self.Graph.nodes[id]['jug'],
                                        self.Graph.nodes[id]['Accp_NegR'],
                                        self.Graph.nodes[id]['AccpR'])

                    #rumor spreading
                    c=np.random.rand()

                    

                    if (c<=ActualAttraction):
                        Nbr_Spreaders+=1
                        self.Graph.nodes[id]['state']='spreaders'
                        neighbours=list(self.Graph.neighbors(id))
                        #Calculating if any nodes of those neighbours can be activated, if yes add them to new_ones.
                        success = np.random.uniform(0,1,len(neighbours)) < self.Probability #choic alpha nodes
                        # success == [ True  True  True False  True .... True False False  True False]
                        new_ones = list(np.extract(success, neighbours))
                        self.Graph.nodes[id]['SendR']+=len(new_ones)
                        
                        
                        #Sending Rumor
                        for each in new_ones:
                            #Acceptance of the Rumor Probability 
                            ProbToAccRumor = self.Graph.degree(id)/ (self.Graph.degree(id) + self.Graph.degree(each))*self.baisAccepte
                            if(np.random.rand()<=ProbToAccRumor):
                                
                                    self.Graph.nodes[each]['AccpR']+=1
            
                                    if (self.Graph.nodes[each]['Infetime']==0 ):
                                        self.Nbr_Infected+=1
                                        self.Nbr_nonInfected-=1
                                        self.Graph.nodes[each]['Infetime'] =time
                                        self.Graph.nodes[each]['opinion'] =self.Graph.nodes[id]['opinion']
                                        self.ListInfectedNodes.append(each)
                                        if (self.Graph.nodes[each]['opinion']=="D"):
                                            #negativ opinion
                                            self.Graph.nodes[each]['Accp_NegR']+=1
                                            self.OpinionDenying+=1
                                        else:
                                            self.OpinionSupporting+=1
                                    elif (self.Graph.nodes[id]['opinion']=="D"):
                                        self.Graph.nodes[each]['Accp_NegR']+=1
                        
                
                        
                              
        
        #save each step to send it to viewing later
            new =pd.DataFrame(data={'NonInfected': self.Nbr_nonInfected ,
                                        'Infected': self.Nbr_Infected,
                                        'Spreaders': Nbr_Spreaders,
                                        'OpinionDenying': self.OpinionDenying,
                                        'OpinionSupporting': self.OpinionSupporting,
                                        'RumorPopularity': RumorPopularity
                                        },index=[time])
            self.Statistical =pd.concat([self.Statistical, new])      
            time += self.setptime;   


 
# Crete Random graphe
def CreateGraph(parameters,N=100,M=3):
    g=nx.barabasi_albert_graph(N,M)
    InitParameters(g,parameters)
    return g

# Init the model paramters 
def getRandomValues(n,min,max):
    return (np.random.rand(n)*(max - min)) + min

def InitParameters(g,parameters):
    n=g.number_of_nodes()

    # Set omega
    
    values = dict( enumerate(getRandomValues(n,parameters['omega_min'],parameters['omega_max'])) )
    nx.set_node_attributes(g, values, 'omega')
    # Set beta
    values = dict( enumerate(getRandomValues(n,parameters['beta_min'],parameters['beta_max'])) )
    nx.set_node_attributes(g, values, 'beta')
    # Set delta
    values = dict( enumerate(getRandomValues(n,parameters['delta_min'],parameters['delta_max'])) )
    nx.set_node_attributes(g, values, 'delta')

    # Set jug
    values = dict( enumerate(getRandomValues(n,parameters['jug_min'],parameters['jug_max'])) )
    nx.set_node_attributes(g, values, 'jug')

    # Set other Attributes
    attributes =[ "Infetime","AccpR","SendR","Accp_NegR"]
    zeros = dict( enumerate(np.zeros(n)) )
    for atrrib in attributes:
        nx.set_node_attributes(g, zeros, atrrib)

    nx.set_node_attributes(g, 'non_infected', "state")
 
    # S, D, Q, T: supporting, Denying, Questioning, Neutral
    nx.set_node_attributes(g, 'S', "opinion")

if __name__ == '__main__':

   
    #Graph's Parametres 
    n=200
    P=0.3
    K=100
    M=20
    nbb=0
   
    parameters={'omega_min':np.pi/24,
                'omega_max':np.pi*2,
                "delta_min":np.pi/24,
                "delta_max":np.pi/2,
                "jug_min":0.1,
                "jug_max":0.9,
                "beta_max":1.2,
                "beta_min":0.2}

    g= CreateGraph(parameters,n)
    seed= int(0.01*n)
    l= ['D','S']
    seedNode=random.sample(range(0,n),seed)
    seedOpinion=random.choices(l,k=seed)
    print(seedNode,seedOpinion,len(seedNode),len(seedOpinion))
    print(g.nodes[0]['jug'])
    sim=HSIBmodel(g,seedNode,seedOpinion)
    sim.simulation()
    sim.DisplyResults()
   
    


   

    
    

    
   

    

    
   
  

