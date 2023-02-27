import pandas as pd
import Model.HISBmodel as m
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Manager


class RumorSimulator():

    def runSimulation(self,g, NbrSim=1 ,seedsSize=0.05, seedNode=None, seedOpinion=None, typeOfSim=1,simName=1,verbose=False,method='non',k=0):
        
        sim = m.HSIBmodel(g, Seed_Set=seedNode, opinion_set=seedOpinion,seedsSize=seedsSize,verbose=verbose,method=method,k=k)
        if verbose:
            print('simulations started')
        with Manager() as manager:
            Stat=manager.list()
            #start_time = time.time()  
            processes=[multiprocessing.Process(target=sim.runModel,args=(i,typeOfSim,Stat))for i in range(NbrSim)] 
            [process.start() for process in processes]
            [process.join() for process in processes]
            df= pd.DataFrame()
            if typeOfSim==0:
                df=self.CreateSimulationsDF(results= Stat,df= df ,simName= typeOfSim)
                result=self.showNetworkMeasuresStatistics(g,df)
                return result
            else:
                df=self.CreateSimulationsDF(results= Stat,df= df ,simName= typeOfSim)
                return df
                
        

    def DisplyResults(self,results,resultType=1):
        color=['black','red','yellow','green','blue','purple','orange','oliver','cyan','maroon','lime','pink','silver','magenta']
        if resultType==0:
           # création de la grille de sous-graphiques
            fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(12, 12))
            
            # tracé de chaque variable en fonction de chaque variable de centralité
            axs[0, 0].scatter(results["deg_cent"], results["AccpR"])
            axs[0, 0].set_xlabel("Degree Centrality")
            axs[0, 0].set_ylabel("AccpR")

            axs[0, 1].scatter(results["clos_cent"], results["AccpR"])
            axs[0, 1].set_xlabel("Closeness Centrality")
            axs[0, 1].set_ylabel("AccpR")

            axs[0, 2].scatter(results["betw_cent"], results["AccpR"])
            axs[0, 2].set_xlabel("Betweenness Centrality")
            axs[0, 2].set_ylabel("AccpR")

            axs[0, 3].scatter(results["page_rank"], results["AccpR"])
            axs[0, 3].set_xlabel("Page Rank")
            axs[0, 3].set_ylabel("AccpR")

            axs[1, 0].scatter(results["deg_cent"], results["SendR"])
            axs[1, 0].set_xlabel("Degree Centrality")
            axs[1, 0].set_ylabel("SendR")

            axs[1, 1].scatter(results["clos_cent"], results["SendR"])
            axs[1, 1].set_xlabel("Closeness Centrality")
            axs[1, 1].set_ylabel("SendR")

            axs[1, 2].scatter(results["betw_cent"], results["SendR"])
            axs[1, 2].set_xlabel("Betweenness Centrality")
            axs[1, 2].set_ylabel("SendR")

            axs[1, 3].scatter(results["page_rank"], results["SendR"])
            axs[1, 3].set_xlabel("Page Rank")
            axs[1, 3].set_ylabel("SendR")

            axs[2, 0].scatter(results["deg_cent"], results["Accp_NegR"])
            axs[2, 0].set_xlabel("Degree Centrality")
            axs[2, 0].set_ylabel("Accp_NegR")

            axs[2, 1].scatter(results["clos_cent"], results["Accp_NegR"])
            axs[2, 1].set_xlabel("Closeness Centrality")
            axs[2, 1].set_ylabel("Accp_NegR")

            axs[2, 2].scatter(results["betw_cent"], results["Accp_NegR"])
            axs[2, 2].set_xlabel("Betweenness Centrality")
            axs[2, 2].set_ylabel("Accp_NegR")

            axs[2, 3].scatter(results["page_rank"], results["Accp_NegR"])
            axs[2, 3].set_xlabel("Page Rank")
            axs[2, 3].set_ylabel("Accp_NegR")
            axs[3, 0].scatter(results["deg_cent"], results["Nb_Accpted_Rm"])
            axs[3, 0].set_xlabel("Degree Centrality")
            axs[3, 0].set_ylabel("Nb_Accpted_Rm")

            axs[3, 1].scatter(results["clos_cent"], results["Nb_Accpted_Rm"])
            axs[3, 1].set_xlabel("Closeness Centrality")
            axs[3, 1].set_ylabel("Nb_Accpted_Rm")

            axs[3, 2].scatter(results["betw_cent"], results["Nb_Accpted_Rm"])
            axs[3, 2].set_xlabel("Betweenness Centrality")
            axs[3, 2].set_ylabel("Nb_Accpted_Rm")

            axs[3, 3].scatter(results["page_rank"], results ["Nb_Accpted_Rm"])
            axs[3, 3].set_xlabel("Page Rank")
            axs[3, 3].set_ylabel("Nb_Accpted_Rm")


            # ajustement des espaces entre les subplots
            plt.tight_layout()

            # affichage du plot
            plt.show()
           
        if resultType==1:
            fig, axe= plt.subplots(2,3)
            for i, ax in enumerate(axe.flat):
                for j in  range(len(results) ):
                    
                    col=results[j].columns
                    ax.plot(results[j].index, results[j][col[i]], color=str(color[j]), label=results[j]['method'][0]+' method ')
                    ax.set_title(f'The evolution of {col[i]}')
                    ax.set_ylabel(f'Number of {col[i]}')
                    ax.set_xlabel(f'Time')
                    ax.legend()
            plt.show()
        elif resultType == 2:
        # Concatenate all results into a single dataframe
            all_results = pd.concat(results)
            print(all_results)
            
            # Create a figure with a subplot for each measure
            fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
            
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




    def CreateSimulationsDF(self,results,df ,simName=1,timeStemp=0.125):
      
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
                
                a=timeStemp*(L-1)
                Nbr_nonInfected=Stat[i]['Non_Infected'][a]
                Nbr_Infected=Stat[i]['Infected'][a]
                Nbr_Spreaders=Stat[i]['Spreaders'][a]
                OpinionDenying=Stat[i]['Opinion_Denying'][a]
                OpinionSupporting=Stat[i]['Opinion_Supporting'][a]
                RumorPopularity=Stat[i]['RumorPopularity'][a]
                method=Stat[i]['method'][a]
                for j in range(L,max):
                    b=j*timeStemp
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
                a=i*timeStemp
                
                
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
                
                    b=j*timeStemp
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
        deg_cent=nx.degree_centrality(Graph)
        clos_cent=nx.closeness_centrality(Graph)
        betw_cent=nx.betweenness_centrality(Graph)
        page_rank=nx.pagerank(Graph,alpha=0.8)


        #prepare the data in the apropreate format
        for i in range(len(data_global)):
            data_global.loc[i, 'deg_cent'] = float(deg_cent[i])
            data_global.loc[i, 'clos_cent'] = float(clos_cent[i])
            data_global.loc[i, 'betw_cent'] = float(betw_cent[i])
            data_global.loc[i, 'page_rank'] = float(page_rank[i])
        print(data_global)
        return data_global

        #plot the statistics for the three attributes "AccpR", "SendR", "Accp_NegR" and "Nb_Accpted_Rm"
        # for attr in ['AccpR','SendR','Accp_NegR','Nb_Accpted_Rm']:
        #     li=[]
           
        #     attr_description=""
        #     if attr == 'AccpR':
        #         li=list_AccpR
        #         numfig="1"
        #         attr_description="Acceptnece of Rumors"
        #     if attr == 'SendR':
        #         li=list_SendR
        #         numfig="2"
        #         attr_description="Send of Rumors"
        #     if attr == 'Accp_NegR':
        #         li=list_Accp_NegR
        #         numfig="3"
        #         attr_description="Accept Negatif Rumors"
        #     if attr == 'Nb_Accpted_Rm':
        #         li=list_Nb_Accpted_Rm
        #         numfig="4"
        #         attr_description="Number of sent and accepted rumors"
        #     # print("li: ",li)
        #     # print("deg_cent_list : ",deg_cent_list)
        #     plt.rcParams["figure.figsize"] = (15,8) 
        #     fig = plt.figure("Figure "+numfig)

        #     plt.subplot(2,2,1)
        #     plt.xlabel('Degree Centrality')
        #     plt.ylabel(attr_description)
        #     plt.scatter(deg_cent_list,li,c='r')
        #     plt.grid()

        #     plt.subplot(2,2,2)
        #     plt.xlabel('Closeness Centrality')
        #     plt.ylabel(attr_description)
        #     plt.scatter(clos_cent_list,li,c='g')
        #     plt.grid()

        #     plt.subplot(2,2,3)
        #     plt.xlabel('Betweenness Centrality')
        #     plt.ylabel(attr_description)
        #     plt.scatter(betw_cent_list,li,c='b')
        #     plt.grid()

        #     plt.subplot(2,2,4)
        #     plt.xlabel('Page rank')
        #     plt.ylabel(attr_description)
        #     plt.scatter(page_rank_list,li,c='black')
        #     plt.grid()

        #     plt.show()
