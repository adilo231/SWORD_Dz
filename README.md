# Introduction:
The S.W.O.R.D. framework is designed to combat the spread of false information and rumors in online social networks through a multi-faceted approach. The framework consists of five modules that work together to achieve this objective:

    Social Media Harvester (SMH): This module is responsible for data harvesting and collection from various social media platforms. The SMH module uses web scraping techniques to gather data related to the spread of false information and rumors.

    Web Data Analyzer (WDA): This module is responsible for data study and analysis. The WDA module processes the collected data to extract insights, patterns, and trends related to the spread of false information and rumors.

    Online Rumor fighter  (ORF): This module is responsible for rumor influence minimization. The OIM module uses a range of strategies to minimize the impact of false information and rumors, including targeted interventions, online reputation management, and social engineering techniques.

    Rumor Detector (RD): This module is responsible for rumor detection and fact checking. The RD module uses natural language processing (NLP) and machine learning (ML) techniques to identify false information and rumors in the collected data, and then verifies their accuracy through fact checking.

    Data Defender (DD): This module is responsible for data storage and management. The DD module ensures the secure and reliable storage of collected data, as well as the effective management of this data to facilitate its analysis and use in combating false information and rumors.

In summary, the S.W.O.R.D. framework offers a comprehensive approach to combatting false information and rumors in online social networks. By using a range of modules that work together to collect, analyze, and manage data, detect false information and rumors, and minimize their influence, the framework can help to protect online communities from the harmful effects of misinformation.

## Installation

1. Clone or download the code from GitHub:
```
$ git clone https://github.com/myusername/SWORD_Dz.git
```


2. Install virtualenv using pip:
```
$ pip install virtualenv
```




3. Create a new virtual environment:
```
$ virtualenv env
```



4. Activate the virtual environment:
```
$ source env/bin/activate
```



5. Install the required Python packages:
```
$ pip install -r requirements.txt
```



6. Run Docker Compose to mount images of MongoDB and Neo4j:
```
$ docker-compose up
```



## Usage

1. Upload the Facebook graph and synthetic graphs to Neo4j using main_load_graph.py:
```
$ python main_load_graph.py
```



2. Run simulations by running main_boulma.py:
```
$ python main_boulma.py
```




# Our Modules 

## Social Media Harvester Module: ![](https://us-central1-progress-markdown.cloudfunctions.net/progress/80)
The Social Media Harvester module is a crucial component of our project as it is responsible for collecting data from Twitter and storing it in our databases for further analysis. We are using two databases for this purpose: MongoDB and Neo4j.

MongoDB is a popular NoSQL database that is well-suited for storing unstructured data, such as tweets. In this module, we are using the Twitter API to collect tweets based on certain criteria, such as hashtags or keywords. Once we have collected the tweets, we store them in a MongoDB collection.

Neo4j, on the other hand, is a graph database that is ideal for storing and querying relationships between entities, such as users and tweets. In this module, we are using the Twitter API to collect information about users who have posted the tweets we collected earlier. We store this information in Neo4j, along with the relationships between users and the tweets they posted.

Overall, the data extraction module is essential for our project as it provides the raw data we need to analyze and detect fake news and rumors on social media. Our progress on this module is currently at 80%, meaning that we have made significant progress towards completing this component of our project, but there is still some work to be done.

## Data Defender Module: ![](https://us-central1-progress-markdown.cloudfunctions.net/progress/70)
The Data Defender module is a critical component of our project that acts as a middleware between all other modules. It consists of three components: docDB, graphDB, and relationldb handlers, which provide control to all modules to store and retrieve data from the databases.

DocDB refers to the document-based database, MongoDB, where we store the tweets we have collected. The docDB handler provides an interface for all other modules to access and retrieve tweets from the database.

GraphDB refers to the graph-based database, Neo4j, where we store the relationships between users and tweets. The graphDB handler provides an interface for all other modules to access and retrieve the relationship information stored in the database.

The data storage module acts as a central hub that allows all other modules to store and retrieve data from the different databases we are using in our project. Our progress on this module is currently at 70%, meaning that we have made significant progress towards completing this component of our project, but there is still some work to be done.

## Web Data Analyzer Module: ![](https://us-central1-progress-markdown.cloudfunctions.net/progress/60)
The Web Data Analyzer module consists of various tools for text and graph transformation to perform tasks such as rumor detection, link prediction, rumor propagation, and rumor influence minimization. The progress on this module is 60%.

## Rumor Detector Module: ![](https://us-central1-progress-markdown.cloudfunctions.net/progress/60)
The Rumor Detector module is constituted of several deep learning graph neural network-based models that allow the framework to detect and classify text as fake or not fake and rumor or not rumor. This module is not yet ready, and progress is at 60%.

## Simulation Module: ![](https://us-central1-progress-markdown.cloudfunctions.net/progress/80)
The simulation module allows the framework to simulate rumor propagation based on HISB model. The module can simulate on different real or synthetic graphs, with the graphs stored in the databases mentioned earlier. This module also allows us to test rumor influence minimization strategies such as blocking nodes and truth campaign strategies and provides statistical analysis. The progress on this module is 80%.






#Conclusion:
Our project aims to provide a framework to combat the spread of fake news and rumors on online social networks. Our project is currently in development and is divided into five modules, each serving a specific purpose. We plan to continue working on this project and hope to achieve our goal of making online social networks a safer place for everyone.

## Credits

This framework was developed by [adilo231](https://github.com/adilo231).


