import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import networkx as nx 



plt.style.use('ggplot')



class HSIBmodel():
    def __init__(self, Graph, Seed_Set=None, opinion_set=None,seedsSize=0.05, baisAccepte=0.3, setptime=0.125, Probability=0.2, Tdet=0.125,k=0, method='none',verbose=False):
        """This is a class for the HISBmodel, which is a rumor propagation model based on human and social behavior.

        Parameters:
        ----------
        Graph : networkx graph
            Graph of the network to run simulation on.
        Seed_Set : list, optional
            Set seeds of the infected nodes. Default is None.
        opinion_set : list, optional
            The opinion of the set seeds of the infected nodes. Default is None.
        seedsSize : float, optional
            The size of the set seeds if the set seeds are not given it will be generated automatically. Default is 0.05.
        baisAccepte : float, optional
            A probability parameter to calibrate the model in the acceptance probability. Default is 0.3.
        setptime : float, optional
            Step time of the simulation. Default is 0.125.
        Probability : float, optional
            A probability parameter to calibrate the model in the acceptance probability. Default is 0.3.
        Tdet : float, optional
            Detection time of a rumor. Default is 0.125.
        k : int, optional
            Number of nodes to be employed in the rumor influence minimization strategy. Default is 0.
        method : str, optional
            The selected rumor influence minimization strategy. Default is 'none'.
        verbose : bool, optional
            If True, print information about the tweets being extracted. Default is False.

        Returns:
        -------
        None.
        """
        
        if method != 'none':
            self.method = method
            self.k = k
            self.Tdet=Tdet
            pass

        # Initialize variables
        self.blocked_nodes = 0
        self.used_nodes_in_TCS=0
        self.time = 0.125
        self.Probability = 0.3
        self.setptime = setptime
        self.Graph = Graph
        self.baisAccepte = baisAccepte
        self.verbose=verbose

        # Generate seed set automatically if not provided
        if Seed_Set==None:
                Seed_Set,opinion_set=self.GenerateSeedsSet(seedsSize)
        self.ListInfectedNodes = Seed_Set
        RumorPopularity = self.SetParameters(opinion_set)
        
        # Create a statistical data frame
        self.Statistical = pd.DataFrame(data={'Non_Infected': self.Nbr_nonInfected,
                                              'Infected': self.Nbr_Infected,
                                              'Spreaders': self.Nbr_Infected,
                                              'Opinion_Denying': self.OpinionDenying,
                                              'Opinion_Supporting': self.OpinionSupporting,
                                              'RumorPopularity': RumorPopularity
                                              }, index=[0])
    def SetParameters(self, opinion_set):
        """Set the parameters for infected nodes
        
        Parameters:
        ----------
        opinion_set : list
            The opinion of the set seeds of the infected nodes.

        Returns:
        -------
        RumorPopularity : int
            Total degree of the infected nodes.
        """
        self.Nbr_Infected = len(self.ListInfectedNodes)

        self.Nbr_nonInfected = self.Graph.number_of_nodes()-self.Nbr_Infected
        self.OpinionDenying = 0
        self.OpinionSupporting = 0
        RumorPopularity = 0
        # Set the parameters for each node in the seed set
        for i, each in enumerate(self.ListInfectedNodes):
            self.Graph.nodes[each]['Infetime'] = 0.125
            self.Graph.nodes[each]['state'] = 'spreaders'
            self.Graph.nodes[each]['AccpR'] += 1
            RumorPopularity += self.Graph.degree(each)
            # Set the opinion of the node based on the opinion_set
            if (opinion_set[i] == 'D'):

                self.Graph.nodes[each]['opinion'] = 'D'
                self.Graph.nodes[each]['Accp_NegR'] += 1
                self.OpinionDenying += 1
            else:
                self.Graph.nodes[each]['opinion'] = 'S'
                self.OpinionSupporting += 1

        return RumorPopularity
    def GenerateSeedsSet(self,size=0.05):
        """
            Generate a random seed set if one is not provided.

            Parameters:
            -----------
            size: float
                The proportion of the total number of nodes to include in the seed set.

            Returns:
            --------
            seedNode: list of int
                The list of node IDs in the seed set.
            seedOpinion: list of str
                The opinions of the nodes in the seed set, either 'D' for denying or 'S' for supporting.
        """
        
        seed = int(size*self.Graph.number_of_nodes())
        l = ['D', 'S']
        seedNode = random.sample(range(0, self.Graph.number_of_nodes()), seed)
        seedOpinion = random.choices(l, k=seed)
        return seedNode,seedOpinion
    
    
    def DisplyResults(self):
        fig, axe = plt.subplots(2, 3)
        col = self.Statistical.columns
        for i, ax in enumerate(axe.flat):
            ax.plot(self.Statistical.index, self.Statistical[col[i]])
            ax.set_title(f'The evolution of {col[i]}')
            ax.set_ylabel(f'Number of {col[i]}')
            ax.set_xlabel(f'Time')

        #     plt.subplot(2,3,idx)
        #     self.Statistical[idx].plot()
        #     axe[:,idx].set_ylabel('YLabel', loc='top')
        plt.show()
    def UpdateOpinion(self, id, jugf, NegR, R):
        """Updates the opinion of a node based on the received rumors.

            Args:
                id (int): The ID of the node to update.
                jugf (float): The subjective judgment parameter of the node.
                NegR (int): The number of negative rumors received by the node.
                R (int): The total number of rumors received by the node.

            Returns:
                None.

            
        """
        opinion = jugf
        assert R > 0  # Ensure R is greater than zero to avoid division by zero later.
        if NegR != 0:
            opinion *= float(NegR / R)  # Update the opinion based on the ratio of negative rumors.
        if (self.Graph.nodes[id]['opinion'] == "S"):
            self.OpinionSupporting -= 1  # Decrement the supporting opinion count if the node previously supported.
        else:
            self.OpinionDenying -= 1  # Decrement the denying opinion count if the node previously denied.

        # Update the opinion of the node with a probability proportional to the updated opinion.
        if(np.random.random_sample() <= opinion):
            self.Graph.nodes[id]['opinion'] = "D"
        else:
            self.Graph.nodes[id]['opinion'] = "S"

        # Update the supporting/denying opinion count based on the updated opinion.
        if (self.Graph.nodes[id]['opinion'] == "S"):
            self.OpinionSupporting += 1
        else:
            self.OpinionDenying += 1
        

    def neighbor(self):
        neighb=[]
        MaxD=[]
        Cente=[]
        beta=[]
        #Cent=((nx.degree_centrality(g)))
        
        Cent=[]
        for i in range(self.Graph.number_of_nodes()):
            Cent.append(self.Graph.nodes[i]['degree_centrality'])
               
        for i in self.ListInfectedNodes:
            n=self.Graph.neighbors(i)
            
            for j in n:
                if j not in self.ListInfectedNodes :
                    if j not in neighb :
                        neighb.append(j)
                        Cente.append(Cent[j])
                        MaxD.append(self.Graph.degree[j])
                        beta.append(self.Graph.nodes[j]['beta'])
                      
        return neighb,MaxD,Cente,beta       
    def judgements(self,nodes):
        jug =[]
        for i in nodes:
            jug.append(self.Graph.nodes[i]['jug'])  
        return jug
          
    def Beta_Blocking_nodes(self,k):
        # select neighbors from the list of Spreaders    
        neighbors,_,_,Beta=self.neighbor()
        
        for i in range(k):
            node_to_block = Beta.index(min(Beta))
            self.Graph.nodes[neighbors[node_to_block]]['blocked']='True'
            Beta.pop(node_to_block)
            self.blocked_nodes+=1
            neighbors.pop(node_to_block) 
    def Random_Blocking_nodes(self,k):
       
        neighbors,_,_,_=self.neighbor()
        size=len(neighbors)
        if k>size:
          k=size-1
        for i in range(k):
            node_to_block=random.randint(0, size-1)
            self.Graph.nodes[neighbors[node_to_block]]['blocked']='true'
            self.blocked_nodes+=1
            neighbors.pop(node_to_block)
            
    def Degree_MAX_Blocking_nodes(self,k):
        neighbors,nodes_degree,_,_=self.neighbor()
        for i in range(k):   
                node_to_block = nodes_degree.index(max(nodes_degree))
                nodes_degree.pop(node_to_block)
                self.Graph.nodes[neighbors[node_to_block]]['blocked']='True'
                self.blocked_nodes+=1
                neighbors.pop(node_to_block)  
    def Centrality_Blocking_nodes(self,k):      
        neighbors,_,centrality,_=self.neighbor()
        for i in range(k):  
            node_to_block = centrality.index(max(centrality))
            centrality.pop(node_to_block)
            self.Graph.nodes[neighbors[node_to_block]]['blocked']='True'
            self.blocked_nodes+=1
            neighbors.pop(node_to_block) 
    
    def blocking_methods(self,k,nodes_degree,centrality,Beta,method):
       
        if method == 'RBN' : #Random_Blocking_nodes
            node_to_block=random.randint(0,k)
           
            
        if method == 'BBN' : #Beta_Blocking_nodes
            node_to_block = Beta.index(min(Beta))
            Beta.pop(node_to_block)
           
            
        if method == 'DMBN' : #Degree_MAX_Blocking_nodes
            node_to_block = nodes_degree.index(max(nodes_degree))
            nodes_degree.pop(node_to_block)
        
        if method == 'BCN' : #Centrality_Blocking_nodes
            node_to_block = centrality.index(max(centrality))
            centrality.pop(node_to_block)

        return node_to_block
            
    def Block_nodes(self,k,method):
        neighbors,nodes_degree,centrality,Beta=self.neighbor()
        size=len(neighbors)
        if k>size:
          k=size-1
       
        for i in range(k):   
                node_to_block = self.blocking_methods(k-i,nodes_degree,centrality,Beta,method)
                print(neighbors)
                print(node_to_block)
                self.Graph.nodes[neighbors[node_to_block]]['blocked']='True'
                self.blocked_nodes+=1
                neighbors.pop(node_to_block)
                
                  
    
    def TCS_methods(self,k,degree,centrality,Beta,judgement,method):
        if method == 'RTCS' :
            node_to_use=random.randint(0, k)
        if method == 'MDTCS' :
            node_to_use = degree.index(max(degree))
            degree.pop(node_to_use)
        if method == 'MRIBHBTCS':
            node_to_use = judgement.index(min(judgement))
            judgement.pop(node_to_use)
        return node_to_use
    
    def Truth_campaign_strategy(self,k,method):
        neighbors,degree,centrality,Beta=self.neighbor()
        judgement=self.judgements(neighbors)
        size=len(neighbors)
        if k > size :
            k=size-1
        for i in range(k):
            node_to_use=self.TCS_methods(k,degree,centrality,Beta,judgement,method)
            self.Graph.nodes[neighbors[node_to_use]]['jug']=1
            self.Graph.nodes[neighbors[node_to_use]]['state']='infected'
            self.used_nodes_in_TCS+=1
            neighbors.pop(node_to_use)
          

    def Random_Truth_Comp(self,k):
        neighbors,_,_,_=self.neighbor()
        size=len(neighbors)
        if k > size :
            k=size-1
        for i in range(k):
            node_to_use=random.randint(0, size-1)
            self.Graph.nodes[neighbors[node_to_use]]['jug']=1
            self.Graph.nodes[neighbors[node_to_use]]['state']='infected'
            self.used_nodes_in_TCS+=1
            neighbors.pop(node_to_use)
            
    def MaxDegree_Truth_Comp(self,K):
        neighbors,degree,_,_=self.neighbor()
        size=len(neighbors)
        k=K
        if k > size :
           k=size-1
        for i in range(k):
            node_to_use = degree.index(max(degree))
            self.Graph.nodes[neighbors[node_to_use]]['jug']=1
            self.Graph.nodes[neighbors[node_to_use]]['state']='infected'
            neighbors.pop(node_to_use)
            degree.pop(node_to_use)
    def MRIBHB_Truth_Comp(self,k):
        neighbors,_,_,_=self.neighbor()
        size=len(neighbors)
        if k > size :
           k=size-1
        judgement=self.judgements(neighbors)
        for i in range(k):
            node_to_use = judgement.index(min(judgement))
            self.Graph.nodes[neighbors[node_to_use]]['jug']=1
            self.Graph.nodes[neighbors[node_to_use]]['state']='infected'
            neighbors.pop(node_to_use)
            judgement.pop(node_to_use)



    def applyRIM(self):
        m = self.method.split("_")
        if self.time>=self.Tdet :
            if m[0]=='T':
                self.Truth_campaign_strategy(self.k,m[1])    

            if m[0]=='B':
                self.Block_nodes(self.k,m[1])

                
            
        
    def runModel(self, i=0, typeOfSim=1, Stat=0):
        """
            Simulates the rumor spreading process in the network using the model parameters specified in the object instance.

            Parameters:
                i (int): Index of the simulation, used for tracking and bookkeeping. Default is 0.
                typeOfSim (int): Type of simulation to run. 0 for per-node statistics, 1 for global statistics, and 2 for basic statistics. Default is 1.
                Stat (list): List to store the statistical results of the simulation. Default is 0.

            Returns:
                None.
        """
        if self.verbose:
            print(f'Simulation number {i} is on run')
        time = 0.125

        while self.ListInfectedNodes:
            # Initialize counters for tracking the rumor spreading process
            RumorPopularity = 0
            Nbr_Spreaders = 0
            for index, id in reversed(list(enumerate(self.ListInfectedNodes))):
                # Calculate the relative time of the node since infection 
                RelativeTime = time - self.Graph.nodes[id]['Infetime']
                # Remove the node from the infection list if its attraction to the rumor is too low
                if (np.exp(-RelativeTime * self.Graph.nodes[id]['beta']) < 0.10):
                    self.ListInfectedNodes.remove(id)
                    self.Graph.nodes[id]['state'] = "infected"

                else:
                     # Calculate the node's attraction to the rumor
                    ActualAttraction = np.exp(-RelativeTime * self.Graph.nodes[id]['beta']) * np.abs(
                        np.sin((RelativeTime * self.Graph.nodes[id]['omega']) + self.Graph.nodes[id]['delta']))
                    RumorPopularity += ActualAttraction * self.Graph.degree(id)
                     # Update the node's opinion based on the model parameters
                    self.UpdateOpinion(id,
                                       self.Graph.nodes[id]['jug'],
                                       self.Graph.nodes[id]['Accp_NegR'],
                                       self.Graph.nodes[id]['AccpR'])

                    # Check if the node will spread the rumor to its neighbors
                    c = np.random.rand()
                    if (c <= ActualAttraction*0.5):
                        Nbr_Spreaders += 1
                        self.Graph.nodes[id]['state'] = 'spreaders'
                        neighbours = list(self.Graph.neighbors(id)) 
                        # Calculating if any nodes of those neighbours can be activated, if yes add them to new_ones.
                        success = np.random.uniform(
                            0, 1, len(neighbours)) < self.Probability  # choic alpha nodes
                        # success == [ True  True  True False  True .... True False False  True False]
                        new_ones = list(np.extract(success, neighbours))
                        self.Graph.nodes[id]['SendR'] += len(new_ones)

                        # Sending Rumor
                        for each in new_ones:
                            # Acceptance of the Rumor Probability
                            ProbToAccRumor = self.Graph.degree(id) / (self.Graph.degree(id) + self.Graph.degree(each))*self.baisAccepte
                            if (self.Graph.nodes[each]['blocked'] =='false'):
                                if(np.random.rand() <= ProbToAccRumor ):

                                    self.Graph.nodes[each]['AccpR'] += 1
                                    self.Graph.nodes[id]['Nb_Accpted_Rm'] += 1
                                    
                                    if (self.Graph.nodes[each]['Infetime'] == 0):
                                        self.Nbr_Infected += 1
                                        self.Nbr_nonInfected -= 1
                                        self.Graph.nodes[each]['Infetime'] = time
                                        self.Graph.nodes[each]['opinion'] = self.Graph.nodes[id]['opinion']
                                        self.ListInfectedNodes.append(each)
                                        if (self.Graph.nodes[each]['opinion'] == "D"):
                                            # negativ opinion
                                            self.Graph.nodes[each]['Accp_NegR'] += 1
                                            self.OpinionDenying += 1
                                        else:
                                            self.OpinionSupporting += 1
                                    elif (self.Graph.nodes[id]['opinion'] == "D"):
                                        self.Graph.nodes[each]['Accp_NegR'] += 1
                            else:
                                pass
                               #print('le noeud',each,'is blocked')
                self.k-=self.blocked_nodes
                self.k-=self.used_nodes_in_TCS
                if self.k>0 and self.method!= 'None':
                    self.applyRIM()
        # save each step to send it to viewing later
            
            new = pd.DataFrame(data={'Non_Infected': self.Nbr_nonInfected,
                                     'Infected': self.Nbr_Infected,
                                     'Spreaders': Nbr_Spreaders,
                                     'Opinion_Denying': self.OpinionDenying,
                                     'Opinion_Supporting': self.OpinionSupporting,
                                     'RumorPopularity': RumorPopularity,
                                     'method':self.method                                     }, index=[time])
            self.Statistical = pd.concat([self.Statistical, new])
            time += self.setptime
        if self.verbose:
            print(f'Simulation number {i} has finnied')

    
        if Stat != 0:
            if typeOfSim == 0:
                Stat_Global=pd.DataFrame()
                for i in range(self.Graph.number_of_nodes()):
                     new =pd.DataFrame(data={'AccpR': self.Graph.nodes[i]['AccpR'],
                                                'SendR': self.Graph.nodes[i]['SendR'],
                                                'Accp_NegR': self.Graph.nodes[i]['Accp_NegR'],
                                                'Nb_Accpted_Rm': self.Graph.nodes[i]['Nb_Accpted_Rm'],

                                                },index=[i])
                     Stat_Global =pd.concat([Stat_Global, new])
                print("stat----------------------------: ",len(Stat_Global))
                Stat.append(Stat_Global)
                
            elif typeOfSim == 1:
                Stat.append(self.Statistical) 
                  
            elif typeOfSim == 2:          
                Stat.append([self.Nbr_Infected,self.OpinionDenying,self.OpinionSupporting,self.method])
                

                
               