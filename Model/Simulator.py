import pandas as pd
import os
import Model.HISBmodel as m
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Manager


class RumorSimulator():

    def runSimulation(self,g, NbrSim=1 ,seedsSize=0.05, seedNode=None, seedOpinion=None, typeOfSim=1,simName=1,verbose=False,method='none',k=0,setptime=0.125):
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
        sim = m.HSIBmodel(g, Seed_Set=seedNode, opinion_set=seedOpinion,seedsSize=seedsSize,verbose=verbose,method=method,k=k,setptime=setptime)
        
        if verbose:
            print(f'simulations started for {method}, noberof k = {k}, DetT= {1},')
        
        # Use a multiprocessing Manager to store simulation statistics in a shared list
        with Manager() as manager:
            Stat=manager.list()
            print('Stat')
            #start_time = time.time()
            # Create a list of processes for running the simulation in parallel  
            processes=[multiprocessing.Process(target=sim.runModel,args=(i,typeOfSim,Stat))for i in range(NbrSim)] 
            # Start all the processes
            [process.start() for process in processes]
            # Wait for all the processes to finish
            [process.join() for process in processes]
            
            df= pd.DataFrame()
            # If the type of simulation is 0 (steady-state simulation), create a DataFrame of simulation results and 
            # calculate network measures statistics, then return the results
            if typeOfSim==0:
                df=self.CreateSimulationsDF(Stat,df ,typeOfSim,setptime)
                result=self.showNetworkMeasuresStatistics(g,df)
                return result
            else:
                # If the type of simulation is 1 (dynamic simulation), create a DataFrame of simulation results and return it
                df=self.CreateSimulationsDF(Stat,df ,typeOfSim,setptime)
                return df
                
    def DisplyResults(self,results,resultType=1,save=False):
        color=['black','red','green','blue','purple','pink','silver','yellow','orange','oliver','cyan','maroon','lime','magenta']
        if resultType==0:
           # création de la grille de sous-graphiques
            fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(12, 12))
            fig.set_size_inches(16, 10)
            # tracé de chaque variable en fonction de chaque variable de centralité
            axs[0, 0].scatter(results["deg_cent"], results["AccpR"],linewidths=0,alpha=1,color=color[0])
            axs[0, 0].set_xlabel("Degree Centrality")
            axs[0, 0].set_ylabel("AccpR")

            axs[0, 1].scatter(results["clos_cent"], results["AccpR"],linewidths=0,alpha=1,color=color[0])
            axs[0, 1].set_xlabel("Closeness Centrality")
            axs[0, 1].set_ylabel("AccpR")

            axs[0, 2].scatter(results["betw_cent"], results["AccpR"],linewidths=0,alpha=1,color=color[0])
            axs[0, 2].set_xlabel("Betweenness Centrality")
            axs[0, 2].set_ylabel("AccpR")

            axs[0, 3].scatter(results["page_rank"], results["AccpR"],linewidths=0,alpha=1,color=color[0])
            axs[0, 3].set_xlabel("Page Rank")
            axs[0, 3].set_ylabel("AccpR")

            axs[0, 4].scatter(results["degree"], results["AccpR"],linewidths=0,alpha=1,color=color[0])
            axs[0, 4].set_xlabel("Degree")
            axs[0, 4].set_ylabel("AccpR")

            axs[1, 0].scatter(results["deg_cent"], results["SendR"],linewidths=0,alpha=1,color=color[1])
            axs[1, 0].set_xlabel("Degree Centrality")
            axs[1, 0].set_ylabel("SendR")

            axs[1, 1].scatter(results["clos_cent"], results["SendR"],linewidths=0,alpha=1,color=color[1])
            axs[1, 1].set_xlabel("Closeness Centrality")
            axs[1, 1].set_ylabel("SendR")

            axs[1, 2].scatter(results["betw_cent"], results["SendR"],linewidths=0,alpha=1,color=color[1])
            axs[1, 2].set_xlabel("Betweenness Centrality")
            axs[1, 2].set_ylabel("SendR")

            axs[1, 3].scatter(results["page_rank"], results["SendR"],linewidths=0,alpha=1,color=color[1])
            axs[1, 3].set_xlabel("Page Rank")
            axs[1, 3].set_ylabel("SendR")

            axs[1, 4].scatter(results["degree"], results["SendR"],linewidths=0,alpha=1,color=color[1])
            axs[1, 4].set_xlabel("Degree")
            axs[1, 4].set_ylabel("SendR")

            axs[2, 0].scatter(results["deg_cent"], results["Accp_NegR"],linewidths=0,alpha=1,color=color[2])
            axs[2, 0].set_xlabel("Degree Centrality")
            axs[2, 0].set_ylabel("Accp_NegR")

            axs[2, 1].scatter(results["clos_cent"], results["Accp_NegR"],linewidths=0,alpha=1,color=color[2])
            axs[2, 1].set_xlabel("Closeness Centrality")
            axs[2, 1].set_ylabel("Accp_NegR")

            axs[2, 2].scatter(results["betw_cent"], results["Accp_NegR"],linewidths=0,alpha=1,color=color[2])
            axs[2, 2].set_xlabel("Betweenness Centrality")
            axs[2, 2].set_ylabel("Accp_NegR")

            axs[2, 3].scatter(results["page_rank"], results["Accp_NegR"],linewidths=0,alpha=1,color=color[2])
            axs[2, 3].set_xlabel("Page Rank")
            axs[2, 3].set_ylabel("Accp_NegR")

            axs[2, 4].scatter(results["degree"], results["Accp_NegR"],linewidths=0,alpha=1,color=color[2])
            axs[2, 4].set_xlabel("Degree")
            axs[2, 4].set_ylabel("Accp_NegR")

            axs[3, 0].scatter(results["deg_cent"], results["Nb_Accpted_Rm"],linewidths=0,alpha=1,color=color[3])
            axs[3, 0].set_xlabel("Degree Centrality")
            axs[3, 0].set_ylabel("Nb_Accpted_Rm")

            axs[3, 1].scatter(results["clos_cent"], results["Nb_Accpted_Rm"],linewidths=0,alpha=1,color=color[3])
            axs[3, 1].set_xlabel("Closeness Centrality")
            axs[3, 1].set_ylabel("Nb_Accpted_Rm")

            axs[3, 2].scatter(results["betw_cent"], results["Nb_Accpted_Rm"],linewidths=0,alpha=1,color=color[3])
            axs[3, 2].set_xlabel("Betweenness Centrality")
            axs[3, 2].set_ylabel("Nb_Accpted_Rm")

            axs[3, 3].scatter(results["page_rank"], results ["Nb_Accpted_Rm"],linewidths=0,alpha=1,color=color[3])
            axs[3, 3].set_xlabel("Page Rank")
            axs[3, 3].set_ylabel("Nb_Accpted_Rm")

            axs[3, 4].scatter(results["degree"], results ["Nb_Accpted_Rm"],linewidths=0,alpha=1,color=color[3])
            axs[3, 4].set_xlabel("Degree")
            axs[3, 4].set_ylabel("Nb_Accpted_Rm")


            # ajustement des espaces entre les subplots
            if save:
                file_list = os.listdir("DataStorage/SimType0")
                number_of_files = len(file_list)
                fig.savefig('SimulationResults/SimType0/image_'+str(number_of_files+1)+'.png', dpi=300 )
            plt.tight_layout()

            # affichage du plot
            plt.show()
           
        if resultType==1:
            fig, axe= plt.subplots(2,3)
            fig.set_size_inches(16, 10)
            for i, ax in enumerate(axe.flat):
                for j in  range(len(results) ):
                    
                    col=results[j].columns
                    ax.plot(results[j].index, results[j][col[i]], color=str(color[j]), label=results[j]['method'][0]+' method ')
                    ax.set_title(f'The evolution of {col[i]}')
                    ax.set_ylabel(f'Number of {col[i]}')
                    ax.set_xlabel(f'Time')
                    ax.legend()
            if save:
                file_list = os.listdir("SimulationResults/SimType1")
                number_of_files = len(file_list)
                fig.savefig('SimulationResults/SimType1/image_'+str(number_of_files+1)+'.png', dpi=300 )
            plt.show()
        elif resultType == 2:
        # Concatenate all results into a single dataframe
            all_results = pd.concat(results)
            print(all_results)
            
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

            if save:
                file_list = os.listdir("DataStorage/SimType2")
                number_of_files = len(file_list)
                fig.savefig('DataStorage/SimType2/image_'+str(number_of_files+1)+'.png', dpi=300 ) 
            plt.show()

    def CreateSimulationsDF(self,results,df ,simName=1,setptime=0.125):
      
       if(simName==0):
           Stat_Global=pd.DataFrame()
           
           L=len(results[0])
           for i in range(L):
               AccpR=0
               SendR=0
               Accp_NegR=0
               Nb_Accpted_Rm=0
               for stat in results:
                   AccpR+=stat['AccpR'][i]
                   SendR+=stat['SendR'][i]
                   Accp_NegR+=stat['Accp_NegR'][i]
                   Nb_Accpted_Rm+=stat['Nb_Accpted_Rm'][i]
               AccpR=int(AccpR/len(results))
               SendR=int(SendR/len(results))
               Accp_NegR=int(Accp_NegR/len(results))
               Nb_Accpted_Rm=int(Nb_Accpted_Rm/len(results))
               new =pd.DataFrame(data={'AccpR': AccpR,
                                                'SendR':SendR,
                                                'Accp_NegR': Accp_NegR,
                                                'Nb_Accpted_Rm': Nb_Accpted_Rm,
                                                
                                                },index=[i])
               Stat_Global =pd.concat([Stat_Global, new])
           #print(Stat_Global)   
           return Stat_Global       
                                            
       if(simName==1):
               
           Stat_Global=pd.DataFrame()
           max=0
           Stat=[]
           for each in results:
                
                L=len(each)
                Stat.append(each)
                if(L>max):
                    max=L
            

           for i in range(len(Stat)):
                L=len(Stat[i])
                
                a=setptime*(L-1)
                Nbr_nonInfected=Stat[i]['Non_Infected'][a]
                Nbr_Infected=Stat[i]['Infected'][a]
                Nbr_Spreaders=Stat[i]['Spreaders'][a]
                OpinionDenying=Stat[i]['Opinion_Denying'][a]
                OpinionSupporting=Stat[i]['Opinion_Supporting'][a]
                RumorPopularity=Stat[i]['RumorPopularity'][a]
                method=Stat[i]['method'][a]
                for j in range(L,max):
                    b=j*setptime
                    new =pd.DataFrame(data={'Non_Infected': Nbr_nonInfected,
                                                'Infected': Nbr_Infected,
                                                'Spreaders': Nbr_Spreaders,
                                                'Opinion_Denying': OpinionDenying,
                                                'Opinion_Supporting': OpinionSupporting,
                                                'RumorPopularity': RumorPopularity,
                                                'method':method
                                                },index=[b])
                    Stat[i] =pd.concat([Stat[i], new])
                #self.DisplyResults(Stat[i],1)
           y0=[]
           y1=[]
           y2=[]
           y3=[]
           y4=[]
           y5=[]
             
           Len=len(Stat)
           for i in range(max):
                a=i*setptime
                
                
                No_Infected=0
                Infected=0
                Spreaders=0
                RumorPopularity=0
                OpinionDenying=0
                OpinionSupporting=0
                method=''
                for each in Stat:
                    
                    No_Infected+=(each['Non_Infected'][a])           
                    Infected+=(each['Infected'][a])
                    Spreaders+=(each['Spreaders'][a])
                    RumorPopularity+=(each['RumorPopularity'][a])
                    OpinionDenying+=(each['Opinion_Denying'][a])
                    OpinionSupporting+=(each['Opinion_Supporting'][a])
                    method=each['method'][a]
                #print("----")
                y0.append(No_Infected/Len)
                y1.append(Infected/Len)
                y2.append(Spreaders/Len)
                y3.append(RumorPopularity/Len)
                y4.append(OpinionDenying/Len)
                y5.append(OpinionSupporting/Len)
            #print(y1)
           for j in range(max):
                
                    b=j*setptime
                    new =pd.DataFrame(data={'Non_Infected': y0[j],
                                                'Infected': y1[j],
                                                'Spreaders': y2[j],
                                                'Opinion_Denying': y4[j],
                                                'Opinion_Supporting': y5[j],
                                                'RumorPopularity': y3[j],
                                                'method':method
                                                },index=[b])
                    Stat_Global =pd.concat([Stat_Global, new])

           print(Stat_Global)   
           return Stat_Global      
       
       elif(simName==2):
        start=0
        if df.empty:
            l=results[0]
            df= pd.DataFrame(data={'Infected':l[0],
                                'Suporting':l[1],
                                'Denying':l[2],
                                'method':l[3],
                                'sim':simName},index=[0])
            start=1
        
        for i in range(start,len(results)):
            l=results[i]
            l.append(simName)
            df.loc[df.shape[0]]=l
        print(df)
       
        return df 
   
    def showNetworkMeasuresStatistics(self,Graph,data_global):
        print ("showing network measures statistics")
        #calculate network measures
        deg_cent=[]
        clos_cent=[]
        betw_cent=[]
        page_rank=[]
        Degree=[]
        for i in range(Graph.number_of_nodes()):
            deg_cent.append(Graph.nodes[i]['degree_centrality'])
            clos_cent.append(Graph.nodes[i]['closeness_centrality'] )
            betw_cent.append(Graph.nodes[i]['between_centrality'] )
            page_rank.append(Graph.nodes[i]['page_rank'])
            Degree.append(Graph.nodes[i]['degree'] ) 

       
        #prepare the data in the apropreate format
        for i in range(len(data_global)):
            data_global.loc[i, 'deg_cent'] = deg_cent[i]
            data_global.loc[i, 'clos_cent'] = clos_cent[i]
            data_global.loc[i, 'betw_cent'] = betw_cent[i]
            data_global.loc[i, 'page_rank'] = page_rank[i]
            data_global.loc[i, 'degree'] = Degree[i]
        
        print(data_global)
        return data_global

 