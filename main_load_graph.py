import DataStorage.GraphGenerator as gg
# from DataExtraction.TwitterExtractor import  TweetExtractor
import Model.Simulator as sim
import numpy as np
import pandas as pd
import time
from tqdm import  tqdm
import DataStorage.FileUploader as fp

Generator=fp.FileUploader(uri="bolt://localhost:7687",username="neo4j",password="1151999aymana")
Generator.uploadFile(graphModel='FB')
# Generator.uploadGraphToDB('ABS')
# Generator.uploadGraphToDB('ABM')
# Generator.uploadGraphToDB('ABL')
