import DataStorage.GraphGenerator as gg
# from DataExtraction.TwitterExtractor import  TweetExtractor
import Simulation.Simulator as sim
import numpy as np
import pandas as pd
import time
from tqdm import  tqdm
import DataStorage.loadData2DB as fp

print('started')
Generator=fp.FileUploader()

################ barabasi_albert_graph de taille 100 ############################
# Generator.uploadGraphToDB('ABS')
# print("ABS is finiched")

################ barabasi_albert_graph de taille 1000 ############################
# Generator.uploadGraphToDB('ABM')
# print("ABM is finiched")

################ barabasi_albert_graph de taille 5000 ############################
# Generator.uploadGraphToDB('ABL')
# print("ABL is finiched")

################ facebook_graph de taille 4039 ############################
Generator.uploadFile(graphModel='FB')
print("fb is finiched")