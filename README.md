# Introduction:
Our project aims to combat the spread of fake news and rumors on online social networks. The framework is designed to extract data from social media platforms, specifically Twitter, and store it in databases for further analysis. Our project is currently divided into five modules, each serving a unique purpose.

## Installation

1. Clone or download the code from GitHub:
$ git clone https://github.com/myusername/myframework.git

# Our Modules 

## Data Extraction Module: ![](https://us-central1-progress-markdown.cloudfunctions.net/progress/80)
The data extraction module is a crucial component of our project as it is responsible for collecting data from Twitter and storing it in our databases for further analysis. We are using two databases for this purpose: MongoDB and Neo4j.

MongoDB is a popular NoSQL database that is well-suited for storing unstructured data, such as tweets. In this module, we are using the Twitter API to collect tweets based on certain criteria, such as hashtags or keywords. Once we have collected the tweets, we store them in a MongoDB collection.

Neo4j, on the other hand, is a graph database that is ideal for storing and querying relationships between entities, such as users and tweets. In this module, we are using the Twitter API to collect information about users who have posted the tweets we collected earlier. We store this information in Neo4j, along with the relationships between users and the tweets they posted.

Overall, the data extraction module is essential for our project as it provides the raw data we need to analyze and detect fake news and rumors on social media. Our progress on this module is currently at 80%, meaning that we have made significant progress towards completing this component of our project, but there is still some work to be done.

## Data Storage Module: ![](https://us-central1-progress-markdown.cloudfunctions.net/progress/70)
The data storage module is a critical component of our project that acts as a middleware between all other modules. It consists of three components: docDB, graphDB, and relationldb handlers, which provide control to all modules to store and retrieve data from the databases.

DocDB refers to the document-based database, MongoDB, where we store the tweets we have collected. The docDB handler provides an interface for all other modules to access and retrieve tweets from the database.

GraphDB refers to the graph-based database, Neo4j, where we store the relationships between users and tweets. The graphDB handler provides an interface for all other modules to access and retrieve the relationship information stored in the database.

The data storage module acts as a central hub that allows all other modules to store and retrieve data from the different databases we are using in our project. Our progress on this module is currently at 70%, meaning that we have made significant progress towards completing this component of our project, but there is still some work to be done.

## Transformers Module: ![](https://us-central1-progress-markdown.cloudfunctions.net/progress/60)
The transformers module consists of various tools for text and graph transformation to perform tasks such as rumor detection, link prediction, rumor propagation, and rumor influence minimization. The progress on this module is 60%.

## Rumor Detection Module: ![](https://us-central1-progress-markdown.cloudfunctions.net/progress/0)
The rumor detection module is constituted of several deep learning graph neural network-based models that allow the framework to detect and classify text as fake or not fake and rumor or not rumor. This module is not yet ready, and progress is at 0%.

## Simulation Module: ![](https://us-central1-progress-markdown.cloudfunctions.net/progress/80)
The simulation module allows the framework to simulate rumor propagation based on HISB model. The module can simulate on different real or synthetic graphs, with the graphs stored in the databases mentioned earlier. This module also allows us to test rumor influence minimization strategies such as blocking nodes and truth campaign strategies and provides statistical analysis. The progress on this module is 80%.






#Conclusion:
Our project aims to provide a framework to combat the spread of fake news and rumors on online social networks. Our project is currently in development and is divided into five modules, each serving a specific purpose. We plan to continue working on this project and hope to achieve our goal of making online social networks a safer place for everyone.

