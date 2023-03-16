import DataStorage.GraphGenerator as gg
# from DataExtraction.TwitterExtractor import  TweetExtractor
import Model.Simulator as sim
import numpy as np
import pandas as pd
import time
from tqdm import  tqdm
import DataStorage.loadData2DB as fp

Generator=fp.FileUploader()
#Generator.uploadFile(graphModel='FB')
#Generator.uploadGraphToDB('ABS')
Generator.uploadGraphToDB('ABM')
Generator.uploadGraphToDB('ABL')
