import pandas as pd
import numpy as np
import os
import Model.HISBmodel as m
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Manager
import multiprocessing


class RumorSimulator():

    def runSimulation(self, g, NbrSim=1, seedsSize=0.05, seedNode=None, seedOpinion=None, typeOfSim=1, simName=1, verbose=False, method='none', blockPeriod=0, Tdet=0.125, k=0, setptime=0.125):
        """
        Runs a simulation of the HSIB model on a given network.

        Parameters:
        -----------
        g : networkx graph
            Graph of the network to run simulation on.

        NbrSim : int, optional
            Number of simulations to run. Default is 1.

        seedsSize : float, optional
            The size of the seed set if seedNode is not provided. Default is 0.05.

        seedNode : list of ints, optional
            List of infected nodes to begin simulation with. If not provided, seed set will be generated randomly.

        seedOpinion : list of strings, optional
            List of opinions of the seed set nodes. Default is None.

        typeOfSim : int, optional
            Type of simulation to run. 1 for the main simulation, 0 for control simulations. Default is 1.

        simName : int, optional
            Name of the simulation. Default is 1.

        verbose : bool, optional
            If True, print information about the simulation. Default is False.

        method : string, optional
            The selected rumor influence minimization strategy. Default is 'non'.

        k : int, optional
            Number of nodes to be employed in the rumor influence minimization strategy. Default is 0.

        setptime : float, optional
            Step time of the simulation. Default is 0.125.

        Returns:
        --------
        result : pandas DataFrame or None
            If typeOfSim is 0, returns a DataFrame with network measures statistics. Otherwise, returns None.
        """
        # Create an instance of the HSIBmodel class with the given parameters
        sim = m.HSIBmodel(g, Seed_Set=seedNode, opinion_set=seedOpinion, seedsSize=seedsSize,
                          verbose=verbose, method=method, blockPeriod=blockPeriod, Tdet=Tdet, k=k, setptime=setptime)

        if verbose:
            print(
                f'simulations started for {method}, noberof k = {k}, DetT= {1},')

        # Use a multiprocessing Manager to store simulation statistics in a shared list
        with Manager() as manager:
            Stat = manager.list()
            #start_time = time.time()
            # Create a list of processes for running the simulation in parallel
            # the number of similation depend the number of cores in your laptob
            num_cores = multiprocessing.cpu_count()
            nub_group = NbrSim/num_cores
            num_lots = int(nub_group)
            num_process_rest = NbrSim - num_lots*num_cores
            for i in range(num_lots):

                processes = [multiprocessing.Process(target=sim.runModel, args=(
                    i, typeOfSim, Stat))for i in range(num_cores)]
                # Start all the processes
                [process.start() for process in processes]
                # Wait for all the process
                # es to finish
                [process.join() for process in processes]

            processes = [multiprocessing.Process(target=sim.runModel, args=(
                i, typeOfSim, Stat))for i in range(num_process_rest)]
            # Start all the processes
            [process.start() for process in processes]
            # Wait for all the process
            # es to finish
            [process.join() for process in processes]

            df = pd.DataFrame()
            # If the type of simulation is 0 (steady-state simulation), create a DataFrame of simulation results and
            # calculate network measures statistics, then return the results
            if typeOfSim == 0:
                df = self.CreateSimulationsDF(Stat, df, typeOfSim, setptime)
                result = self.showNetworkMeasuresStatistics(g, df)
                return result
            else:
                # If the type of simulation is 1 (dynamic simulation), create a DataFrame of simulation results and return it
                df = self.CreateSimulationsDF(Stat, df, typeOfSim, setptime)
                return df

    def DisplyResults(self, results, resultType=1, save=False, imageName=""):
        color = ['black', 'red', 'green', 'blue', 'purple', 'pink',
                 'silver', 'yellow', 'orange', 'cyan', 'maroon', 'lime', 'magenta']
        if resultType == 0:
            # create a list of tuples containing column names and axis labels
            cols_and_labels = [("deg_cent", "Degree Centrality"),
                               ("clos_cent", "Closeness Centrality"), 
                               ("betw_cent", "Betweenness Centrality"),
                               ("page_rank", "Page Rank"),                   
                               ("degree", "Degree")]

            # create a list of tuples containing column names and y-axis labels
            y_cols_and_labels = [("AccpR", "# of Accepted rumor"),                     
                                 ("SendR", "# of sent Rumor "),                     
                                 ("Accp_NegR", "# Of Accepted neg Rumor"),                     
                                 ("Nb_Accpted_Rm", "Nodes Send Impact")]

            fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(16, 10))

            for i, (col, xlabel) in enumerate(cols_and_labels):
                for j, (y_col, ylabel) in enumerate(y_cols_and_labels):
                    ax = axs[j, i]
                    ax.scatter(results[col], results[y_col], linewidths=0, alpha=0.4, color=color[j])
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel(ylabel)
                    
            fig.tight_layout()


            # affichage du plot
            plt.show()

        if resultType == 1:
            fig, axe = plt.subplots(2, 3)
            fig.set_size_inches(16, 10)
            for i, ax in enumerate(axe.flat):
                for j in range(len(results)):

                    col = results[j].columns
                    ax.plot(results[j].index, results[j]
                            [col[i]],  label=results[j]['method'][0])
                    ax.set_title(f'The evolution of {col[i]}')
                    ax.set_ylabel(f'Number of {col[i]}')
                    ax.set_xlabel(f'Time')

            # Get the handles and labels from the last axes object
            handles, labels = axe[-1][-1].get_legend_handles_labels()

            # Create a single legend for all subplots using the handles and labels
            fig.legend(handles, labels, loc='center left')
            plt.show()
        elif resultType == 2:
            # Concatenate all results into a single dataframe
            all_results = pd.concat(results)

            # Create a figure with a subplot for each measure
            fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
            fig.set_size_inches(16, 10)
            # Loop over each measure and create a boxplot for each simulation method
            for i, name in enumerate(['Infected', 'Suporting', 'Denying']):
                sns.boxplot(data=all_results, x='method', y=name, ax=axes[i])
                axes[i].set_ylabel('Number of individuals')
                axes[i].set_title(name)

            # Remove the automatic x-axis label from all but the bottom subplot
            if axes[0] != axes[-1]:
                for ax in axes[:-1]:
                    ax.set_xlabel('')
            plt.show()

        # save the figure in DataStorage/simulationResults
        if save:
            path = self.saveResult(imageName, resultType)
            fig.savefig(path, dpi=300)

    def CreateSimulationsDF(self, results, df, simName=1, setptime=0.125):
        SizeResults = len(results)
        if(simName == 0):
            data = {'AccpR': np.zeros(len(results[0])),
                    'SendR': np.zeros(len(results[0])),
                    'Accp_NegR': np.zeros(len(results[0])),
                    'Nb_Accpted_Rm': np.zeros(len(results[0])),
                    'beta': np.zeros(len(results[0])),
                    'omega': np.zeros(len(results[0])),
                    'delta': np.zeros(len(results[0])),
                    'jug': np.zeros(len(results[0])),
                }

            for stat in results:
                for key in data:
                    data[key] += stat[key]

            for key in data:
                data[key] /= len(results)

            Stat_Global = pd.DataFrame(data=data)
            return Stat_Global


            # Stat_Global = pd.DataFrame()

            # L = len(results[0])
            # for i in range(L):
            #     AccpR = 0
            #     SendR = 0
            #     Accp_NegR = 0
            #     Nb_Accpted_Rm = 0
            #     beta = 0.0
            #     omega = 0.0
            #     delta = 0.0
            #     jug = 0.0
            #     for stat in results:
            #         AccpR += stat['AccpR'][i]
            #         SendR += stat['SendR'][i]
            #         Accp_NegR += stat['Accp_NegR'][i]
            #         Nb_Accpted_Rm += stat['Nb_Accpted_Rm'][i]
            #         beta += stat['beta'][i]
            #         omega += stat['omega'][i]
            #         delta += stat['delta'][i]
            #         jug += stat['jug'][i]
            #     AccpR = int(AccpR/SizeResults)
            #     SendR = int(SendR/SizeResults)
            #     Accp_NegR = int(Accp_NegR/SizeResults)
            #     Nb_Accpted_Rm = int(Nb_Accpted_Rm/SizeResults)
            #     beta = float(beta/SizeResults)
            #     omega = float(omega/SizeResults)
            #     delta = float(delta/SizeResults)
            #     jug = float(jug/SizeResults)
               
            #     new = pd.DataFrame(data={'AccpR': AccpR,
            #                              'SendR': SendR,
            #                              'Accp_NegR': Accp_NegR,
            #                              'Nb_Accpted_Rm': Nb_Accpted_Rm,
            #                              'beta': beta,
            #                              'omega': omega,
            #                              'delta': delta,
            #                              'jug': jug,
            #                              }, index=[i])
            #     Stat_Global = pd.concat([Stat_Global, new])
            # # print(Stat_Global)
            return Stat_Global

        if(simName == 1):

            Stat_Global = pd.DataFrame()
            max = 0
            Stat = []
            for each in results:

                L = len(each)
                Stat.append(each)
                if(L > max):
                    max = L

            for i in range(len(Stat)):
                L = len(Stat[i])

                a = setptime*(L-1)
                Nbr_nonInfected = Stat[i]['Non_Infected'][a]
                Nbr_Infected = Stat[i]['Infected'][a]
                Nbr_Spreaders = Stat[i]['Spreaders'][a]
                OpinionDenying = Stat[i]['Opinion_Denying'][a]
                OpinionSupporting = Stat[i]['Opinion_Supporting'][a]
                RumorPopularity = Stat[i]['RumorPopularity'][a]
                method = Stat[i]['method'][a]
                for j in range(L, max):
                    b = j*setptime
                    new = pd.DataFrame(data={'Non_Infected': Nbr_nonInfected,
                                             'Infected': Nbr_Infected,
                                             'Spreaders': Nbr_Spreaders,
                                             'Opinion_Denying': OpinionDenying,
                                             'Opinion_Supporting': OpinionSupporting,
                                             'RumorPopularity': RumorPopularity,
                                             'method': method
                                             }, index=[b])
                    Stat[i] = pd.concat([Stat[i], new])
                # self.DisplyResults(Stat[i],1)
            y0 = []
            y1 = []
            y2 = []
            y3 = []
            y4 = []
            y5 = []

            Len = len(Stat)
            for i in range(max):
                a = i*setptime

                No_Infected = 0
                Infected = 0
                Spreaders = 0
                RumorPopularity = 0
                OpinionDenying = 0
                OpinionSupporting = 0
                method = ''
                for each in Stat:

                    No_Infected += (each['Non_Infected'][a])
                    Infected += (each['Infected'][a])
                    Spreaders += (each['Spreaders'][a])
                    RumorPopularity += (each['RumorPopularity'][a])
                    OpinionDenying += (each['Opinion_Denying'][a])
                    OpinionSupporting += (each['Opinion_Supporting'][a])
                    method = each['method'][a]
                # print("----")
                y0.append(No_Infected/Len)
                y1.append(Infected/Len)
                y2.append(Spreaders/Len)
                y3.append(RumorPopularity/Len)
                y4.append(OpinionDenying/Len)
                y5.append(OpinionSupporting/Len)
            # print(y1)
            for j in range(max):

                b = j*setptime
                new = pd.DataFrame(data={'Non_Infected': y0[j],
                                         'Infected': y1[j],
                                         'Spreaders': y2[j],
                                         'Opinion_Denying': y4[j],
                                         'Opinion_Supporting': y5[j],
                                         'RumorPopularity': y3[j],
                                         'method': method
                                         }, index=[b])
                Stat_Global = pd.concat([Stat_Global, new])

            return Stat_Global

        elif(simName == 2):
            start = 0
            if df.empty:
                l = results[0]
                df = pd.DataFrame(data={'Infected': l[0],
                                        'Suporting': l[1],
                                        'Denying': l[2],
                                        'method': l[3],
                                        'sim': simName}, index=[0])
                start = 1

            for i in range(start, len(results)):
                l = results[i]
                l.append(simName)
                df.loc[df.shape[0]] = l

            return df

    def showNetworkMeasuresStatistics(self,Graph, data_global):
        """
        Calculates and adds network measures statistics to the given dataframe.

        Args:
            Graph (networkx.Graph): The network graph object.
            data_global (pandas.DataFrame): The dataframe to add the network measures statistics to.

        Returns:
            pandas.DataFrame: The dataframe with the added network measures statistics.
        """
       
        # extract network measures from the Graph object
        nodes_data = Graph.nodes.data()
        deg_cent = [node_data['degree_centrality'] for _, node_data in nodes_data]
        clos_cent = [node_data['closeness_centrality'] for _, node_data in nodes_data]
        betw_cent = [node_data['between_centrality'] for _, node_data in nodes_data]
        page_rank = [node_data['page_rank'] for _, node_data in nodes_data]
        Degree = [node_data['degree'] for _, node_data in nodes_data]

        # update data_global DataFrame with network measures statistics
        data_global['deg_cent'] = deg_cent
        data_global['clos_cent'] = clos_cent
        data_global['betw_cent'] = betw_cent
        data_global['page_rank'] = page_rank
        data_global['degree'] = Degree

        # Print status message.
        print("Network measures statistics calculated.")

        # Return the modified dataframe.
        return data_global


    def saveResult(self, imageName, type):
        dirPath = "DataStorage/SimulationResults/SimType"+str(type)+"/"
        if imageName != "":
            imageName = os.path.splitext(imageName)[0]+".png"

            while os.path.exists(dirPath+imageName):
                name_without_extension = os.path.splitext(imageName)[0]
                m = name_without_extension.split('_')
                if len(m) > 1:
                    name_without_extension = m[0]+"_"+str(int(m[1])+1)
                else:
                    name_without_extension += "_"+str(1)

                imageName = name_without_extension+".png"
        else:
            imageName = "image.png"
            while os.path.exists(dirPath+imageName):
                name_without_extension = os.path.splitext(imageName)[0]
                m = name_without_extension.split('_')
                if len(m) > 1:
                    name_without_extension = m[0]+"_"+str(int(m[1])+1)
                else:
                    name_without_extension += "_"+str(1)

                imageName = name_without_extension+".png"
        dirPath += imageName
        return dirPath
