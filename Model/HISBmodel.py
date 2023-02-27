import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt



plt.style.use('ggplot')



class HSIBmodel():
    def __init__(self, Graph, Seed_Set=None, opinion_set=None,seedsSize=0.05, baisAccepte=0.3, setptime=0.125, Probability=0.3, Tdet=np.Infinity,k=0, method='none',verbose=False):
        
        if method != 'none':
            self.method = method
            self.k = k
            self.Tdet=Tdet
            pass


        self.setptime = setptime
        self.Graph = Graph
        self.time = 0.125
        self.Probability = 0.3
        self.verbose=verbose
        if Seed_Set==None:
                Seed_Set,opinion_set=self.GenerateSeedsSet(seedsSize)
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
    def applyRIM(self):

        if self.method == 'BLNS':
            pass
        if self.method == 'RBLNS':
            pass
        
    def runModel(self, i=0, typeOfSim=1, send_end=0):
        if self.verbose:
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
                            ProbToAccRumor = self.Graph.degree(id) / (self.Graph.degree(id) + self.Graph.degree(each))*self.baisAccepte
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

            #self.applyRIM()
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
        if self.verbose:
         print(f'Simulation number {i} has finnied')
        if send_end != 0:
            if typeOfSim == 1:
                send_end.send(self.Statistical)
            elif typeOfSim == 2:
                send_end.send([self.Nbr_Infected,self.OpinionDenying,self.OpinionSupporting])