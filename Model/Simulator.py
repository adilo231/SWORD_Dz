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
        df=self.CreateSimulationsDF(pipe_list, df ,simName)
        return df


    # Crete Random graphe

    def DisplyResults(self,results,resultType=1):
        if resultType==1:
            #to compleeete this one
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
        return df 
