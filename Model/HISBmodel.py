import numpy as np
import networkx as nx
import pandas as pd
import seaborn as sns
import random
import matplotlib.pyplot as plt
import matplotlib
import multiprocessing
import time
import DataStorage.GraphGenerator as gg


plt.style.use('ggplot')



class HSIBmodel():
    def __init__(self, Graph, Seed_Set=None, opinion_set=None,seedsSize=0.05, baisAccepte=0.3, setptime=0.125, Probability=0.3, Tdet=np.Infinity, method='none'):
        
        self.setptime = setptime
        self.Graph = Graph
        self.time = 0.125
        self.Probability = 0.3

        if Seed_Set==None:
                Seed_Set,opinion_set=self.GenerateSeedsSet(seedsSize)
                print(Seed_Set,opinion_set)
        self.ListInfectedNodes = Seed_Set
        RumorPopularity = self.SetParameters(opinion_set)
        self.baisAccepte = baisAccepte
        self.Statistical = pd.DataFrame(data={'Non_Infected': self.Nbr_nonInfected,
                                              'Infected': self.Nbr_Infected,
                                              'Spreaders': self.Nbr_Infected,
                                              'Opinion_Denying': self.OpinionDenying,
                                              'Opinion_Supporting': self.OpinionSupporting,
                                              'RumorPopularity': RumorPopularity
                                              }, index=[0])
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
    def SetParameters(self, opinion_set):
        self.Nbr_Infected = len(self.ListInfectedNodes)

        self.Nbr_nonInfected = self.Graph.number_of_nodes()-self.Nbr_Infected
        self.OpinionDenying = 0
        self.OpinionSupporting = 0
        RumorPopularity = 0
        for i, each in enumerate(self.ListInfectedNodes):
            self.Graph.nodes[each]['Infetime'] = 0.125
            self.Graph.nodes[each]['state'] = 'spreaders'
            self.Graph.nodes[each]['AccpR'] += 1
            RumorPopularity += self.Graph.degree(each)

            if (opinion_set[i] == 'D'):

                self.Graph.nodes[each]['opinion'] = 'D'
                self.Graph.nodes[each]['Accp_NegR'] += 1
                self.OpinionDenying += 1
            else:
                self.Graph.nodes[each]['opinion'] = 'S'
                self.OpinionSupporting += 1

        return RumorPopularity
    def GenerateSeedsSet(self,size=0.05):
        seed = int(size*self.Graph.number_of_nodes())
        l = ['D', 'S']
        seedNode = random.sample(range(0, self.Graph.number_of_nodes()), seed)
        seedOpinion = random.choices(l, k=seed)
        return seedNode,seedOpinion
    def UpdateOpinion(self, id, jugf, NegR, R):
        opinion = jugf
        assert R > 0
        if NegR != 0:
            opinion *= float(NegR / R)
        if (self.Graph.nodes[id]['opinion'] == "S"):
            self.OpinionSupporting -= 1
        else:
            self.OpinionDenying -= 1

        if(np.random.random_sample() <= opinion):
            self.Graph.nodes[id]['opinion'] = "D"
        else:
            self.Graph.nodes[id]['opinion'] = "S"

        if (self.Graph.nodes[id]['opinion'] == "S"):
            self.OpinionSupporting += 1
        else:
            self.OpinionDenying += 1

    def runModel(self, i=0, typeOfSim=1, send_end=0):
        print(f'Simulation number {i} is on run')
        time = 0.125

        while self.ListInfectedNodes:
            RumorPopularity = 0
            Nbr_Spreaders = 0
            for index, id in reversed(list(enumerate(self.ListInfectedNodes))):
                RelativeTime = time - self.Graph.nodes[id]['Infetime']

                if (np.exp(-RelativeTime * self.Graph.nodes[id]['beta']) < 0.10):
                    self.ListInfectedNodes.remove(id)
                    self.Graph.nodes[id]['state'] = "infected"

                else:
                    # Node Attaction
                    ActualAttraction = np.exp(-RelativeTime * self.Graph.nodes[id]['beta']) * np.abs(
                        np.sin((RelativeTime * self.Graph.nodes[id]['omega']) + self.Graph.nodes[id]['delta']))
                    RumorPopularity += ActualAttraction * self.Graph.degree(id)
                    # Update node opinion
                    self.UpdateOpinion(id,
                                       self.Graph.nodes[id]['jug'],
                                       self.Graph.nodes[id]['Accp_NegR'],
                                       self.Graph.nodes[id]['AccpR'])

                    # rumor spreading
                    c = np.random.rand()

                    if (c <= ActualAttraction):
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
                            ProbToAccRumor = self.Graph.degree(
                                id) / (self.Graph.degree(id) + self.Graph.degree(each))*self.baisAccepte
                            if(np.random.rand() <= ProbToAccRumor):

                                self.Graph.nodes[each]['AccpR'] += 1

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

            
        # save each step to send it to viewing later
            new = pd.DataFrame(data={'Non_Infected': self.Nbr_nonInfected,
                                     'Infected': self.Nbr_Infected,
                                     'Spreaders': Nbr_Spreaders,
                                     'Opinion_Denying': self.OpinionDenying,
                                     'Opinion_Supporting': self.OpinionSupporting,
                                     'RumorPopularity': RumorPopularity
                                     }, index=[time])
            self.Statistical = pd.concat([self.Statistical, new])
            time += self.setptime
        print(f'Simulation number {i} has finnied')
        if send_end != 0:
            if typeOfSim == 1:
                send_end.send(self.Statistical)
            elif typeOfSim == 2:
                send_end.send([self.Nbr_Infected,self.OpinionDenying,self.OpinionSupporting])


def Simulations(NbrSim, g,seedsSize=0.05, seedNode=None, seedOpinion=None, typeOfSim=1):
    jobs = []
    pipe_list = []
    sim = HSIBmodel(g, Seed_Set=seedNode, opinion_set=seedOpinion,seedsSize=seedsSize)
    for i in range(NbrSim):
        recv_end, send_end = multiprocessing.Pipe(False)
        p = multiprocessing.Process(
            target=sim.runModel, args=(i, typeOfSim, send_end))
        jobs.append(p)
        pipe_list.append(recv_end)

    for proc in jobs:
        proc.start()
    for proc in jobs:
        proc.join()
    # result_list = [x.recv() for x in pipe_list]
    # print (result_list)
    return pipe_list


# Crete Random graphe
def CreateGraph(parameters, N=100, M=3):
    g = nx.barabasi_albert_graph(N, M)
    InitParameters(g, parameters)
    return g

# Init the model paramters


def GetRandomValues(n, min, max):
    return (np.random.rand(n)*(max - min)) + min


def InitParameters(g, parameters):
    n = g.number_of_nodes()

    # Set omega

    values = dict(enumerate(GetRandomValues(
        n, parameters['omega_min'], parameters['omega_max'])))
    nx.set_node_attributes(g, values, 'omega')
    # Set beta
    values = dict(enumerate(GetRandomValues(
        n, parameters['beta_min'], parameters['beta_max'])))
    nx.set_node_attributes(g, values, 'beta')
    # Set delta
    values = dict(enumerate(GetRandomValues(
        n, parameters['delta_min'], parameters['delta_max'])))
    nx.set_node_attributes(g, values, 'delta')

    # Set jug
    values = dict(enumerate(GetRandomValues(
        n, parameters['jug_min'], parameters['jug_max'])))
    nx.set_node_attributes(g, values, 'jug')

    # Set other Attributes
    attributes = ["Infetime", "AccpR", "SendR", "Accp_NegR"]
    zeros = dict(enumerate(np.zeros(n)))
    for atrrib in attributes:
        nx.set_node_attributes(g, zeros, atrrib)

    nx.set_node_attributes(g, 'non_infected', "state")

    # S, D, Q, T: supporting, Denying, Questioning, Neutral
    nx.set_node_attributes(g, 'S', "opinion")

def DisplyResults(results,resultType=1):
    if resultType==1:
        fig, axe = plt.subplots(2, 3)
        for df_rep in results:
            df = df_rep.recv()
            col = df.columns
            for i, ax in enumerate(axe.flat):
                ax.plot(df[col[i]])
                ax.set_title(f'The evolution of {col[i]}')
                ax.set_ylabel(f'Number of {col[i]}')
                ax.set_xlabel(f'Time')
    elif resultType==2:
        pass
    plt.show()

def CreateDataFrame(results,df ,sim=1):
    start=0
    if df.empty:
        l=results[0].recv()
        df= pd.DataFrame(data={'Infected':l[0],
                            'Suporting':l[1],
                            'Denying':l[2],
                            'sim':sim},index=[0])
        start=1
    
    
    for i in range(start,len(results)):
        l=results[i].recv()
        l.append(sim)
        df.loc[df.shape[0]]=l
    return df 


if __name__ == '__main__':

    # Graph's Parametres
    n = 100
    P = 0.3
    K = 100
    M = 20
    nbb = 0
    NbrSim = 50

    # parameters = {'omega_min': np.pi/24,
    #               'omega_max': np.pi*2,
    #               "delta_min": np.pi/24,
    #               "delta_max": np.pi/2,
    #               "jug_min": 0.1,
    #               "jug_max": 0.4,
    #               "beta_max": 1.2,
    #               "beta_min": 0.05}
    # print('graphe generation')
    # g = CreateGraph(parameters, n)
    # seed = int(0.05*n)
    # l = ['D', 'S']
    # seedNode = random.sample(range(0, n), seed)
    # seedOpinion = random.choices(l, k=seed)
    # print('simulation')

    # run simple simulation and display
    # print(seedNode,seedOpinion,len(seedNode),len(seedOpinion))
    # print(g.nodes[0]['jug'])
    # sim=HSIBmodel(g,seedNode,seedOpinion)
    # sim.runModel()
    # sim.DisplyResults()

    # # Run multiple and paralle simulations than display
    # start_time = time.time()
    # dfs = Simulations(3, g, seedNode, seedOpinion, 1)
    # end_time = time.time()
    # print('Parallel time: ', end_time-start_time)
    # DisplyResults(dfs)

    # Run multiple and paralle simulations get final results
    # start_time = time.time()
    # parameters = {'omega_min': np.pi/24,
    #               'omega_max': np.pi*2,
    #               "delta_min": np.pi/24,
    #               "delta_max": np.pi/2,
    #               "jug_min": 0,
    #               "jug_max": 0,
    #               "beta_max": 1.2,
    #               "beta_min": 0.8}
    # SimulationResults= pd.DataFrame()
    # for beta in np.arange(0.1,1,0.1):
    #     parameters['beta_min']=beta
    #     parameters['beta_max']=beta+0.1
    #     g = CreateGraph(parameters, n)
    #     results = Simulations(10, g, typeOfSim= 2)
    #     SimulationResults = CreateDataFrame(results,SimulationResults,sim=beta)
    # end_time = time.time()
    # print('Parallel time: ', end_time-start_time)
    
    # fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    # for name, ax in zip(['Infected','Suporting','Denying'], axes):
    #     sns.boxplot(data=SimulationResults, x='sim', y=name, ax=ax)
    #     ax.set_ylabel('Number of individuals')
    #     ax.set_title(name)
    # # Remove the automatic x-axis label from all but the bottom subplot
    # if ax != axes[-1]:
    #     ax.set_xlabel('')
    # plt.show()
   

    # Get the DataFrame results from simulation
    # for x in pipe_list:

    #  print((x.recv().shape))
    gg.printGraph()