import pandas as pd
import Model.HISBmodel as m
import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing


class RumorSimulator():

    def runSimulation(self,g, NbrSim=1 ,seedsSize=0.05, seedNode=None, seedOpinion=None, typeOfSim=1,simName=1,verbose=False):
        jobs = []
        pipe_list = []
        sim = m.HSIBmodel(g, Seed_Set=seedNode, opinion_set=seedOpinion,seedsSize=seedsSize,verbose=verbose)
        if verbose:
            print('simulations started')
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
        if verbose:
            print('simulations started finished')
        df= pd.DataFrame()
        df=self.CreateSimulationsDF(pipe_list, df ,typeOfSim)
        return df


    # Crete Random graphe

    def DisplyResults(self,results,resultType=1):
        if resultType==1:
            fig, axe= plt.subplots(2,3)
            col=results.columns
            for i, ax in enumerate(axe.flat):
                ax.plot(results.index,results[col[i]])
                ax.set_title(f'The evolution of {col[i]}')
                ax.set_ylabel(f'Number of {col[i]}')
                ax.set_xlabel(f'Time')
            plt.show()
        elif resultType==2:
            fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
            for name, ax in zip(['Infected','Suporting','Denying'], axes):
                sns.boxplot(data=results, x='sim', y=name, ax=ax)
                ax.set_ylabel('Number of individuals')
                ax.set_title(name)
            # Remove the automatic x-axis label from all but the bottom subplot
            if ax != axes[-1]:
                ax.set_xlabel('')
        
            plt.show()

    def CreateSimulationsDF(self,results,df ,simName=1):
       if(simName==1):
           for i in range(len(results)):
                results[i]=results[i].recv()
            
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
                
                a=0.125*(L-1)
                Nbr_nonInfected=Stat[i]['Non_Infected'][a]
                Nbr_Infected=Stat[i]['Infected'][a]
                Nbr_Spreaders=Stat[i]['Spreaders'][a]
                OpinionDenying=Stat[i]['Opinion_Denying'][a]
                OpinionSupporting=Stat[i]['Opinion_Supporting'][a]
                RumorPopularity=Stat[i]['RumorPopularity'][a]
                for j in range(L,max):
                    b=j*0.125
                    new =pd.DataFrame(data={'Non_Infected': Nbr_nonInfected,
                                                'Infected': Nbr_Infected,
                                                'Spreaders': Nbr_Spreaders,
                                                'Opinion_Denying': OpinionDenying,
                                                'Opinion_Supporting': OpinionSupporting,
                                                'RumorPopularity': RumorPopularity
                                                },index=[b])
                    Stat[i] =pd.concat([Stat[i], new])
                #DisplyResultsT(Stat[i])
           y0=[]
           y1=[]
           y2=[]
           y3=[]
           y4=[]
           y5=[]
             
           Len=len(Stat)
           for i in range(max):
                a=i*0.125
                
                
                No_Infected=0
                Infected=0
                Spreaders=0
                RumorPopularity=0
                OpinionDenying=0
                OpinionSupporting=0
                for each in Stat:
                    
                    No_Infected+=(each['Non_Infected'][a])           
                    Infected+=(each['Infected'][a])
                    Spreaders+=(each['Spreaders'][a])
                    RumorPopularity+=(each['RumorPopularity'][a])
                    OpinionDenying+=(each['Opinion_Denying'][a])
                    OpinionSupporting+=(each['Opinion_Supporting'][a])
                #print("----")
                y0.append(No_Infected/Len)
                y1.append(Infected/Len)
                y2.append(Spreaders/Len)
                y3.append(RumorPopularity/Len)
                y4.append(OpinionDenying/Len)
                y5.append(OpinionSupporting/Len)
            #print(y1)
           for j in range(max):
                
                    b=j*0.125
                    new =pd.DataFrame(data={'Non_Infected': y0[j],
                                                'Infected': y1[j],
                                                'Spreaders': y2[j],
                                                'Opinion_Denying': y4[j],
                                                'Opinion_Supporting': y5[j],
                                                'RumorPopularity': y3[j]
                                                },index=[b])
                    Stat_Global =pd.concat([Stat_Global, new])

              
           return Stat_Global      
       
       elif(simName==2):
        start=0
        if df.empty:
            l=results[0].recv()
            df= pd.DataFrame(data={'Infected':l[0],
                                'Suporting':l[1],
                                'Denying':l[2],
                                'sim':simName},index=[0])
            start=1
        for i in range(start,len(results)):
            l=results[i].recv()
            l.append(simName)
            df.loc[df.shape[0]]=l
        print(df)
        return df 
