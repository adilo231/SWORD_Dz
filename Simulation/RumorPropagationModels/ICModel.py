import matplotlib.pyplot as plt
import random
import networkx as nx
import pandas as pd
import seaborn as sns

class ICModel:
    
    def __init__(self, graph):
        self.graph = graph
    
    def simulate(self, seeds, p):
        
        active_nodes = set(seeds)
        new_active_nodes = set(seeds)
        time_step = 0
        result = [(time_step, len(active_nodes), len(new_active_nodes))]
        
        while new_active_nodes:
            time_step += 1
            next_active_nodes = set()
            for node in new_active_nodes:
                for neighbor in self.graph.neighbors(node):
                    if neighbor not in active_nodes:
                        if random.random() < p:
                            next_active_nodes.add(neighbor)
            active_nodes |= next_active_nodes
            new_active_nodes = next_active_nodes
            result.append((time_step, len(active_nodes), len(new_active_nodes)))
        
        return pd.DataFrame(result, columns=['Time Step', 'Total Activated', 'Newly Activated'])
