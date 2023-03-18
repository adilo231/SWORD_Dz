# Introduction:
Our project aims to combat the spread of fake news and rumors on online social networks. The framework is designed to extract data from social media platforms, specifically Twitter, and store it in databases for further analysis. Our project is currently divided into five modules, each serving a unique purpose.

## Data Extraction Module:
The data extraction module extracts data from Twitter and stores it in two databases: MongoDB and Neo4j. MongoDB is used to store tweets, while Neo4j stores the links between users and tweets. The progress on this module is 80%.

## Data Storage Module:
The data storage module is the middleware between all other modules. It is made up of docDB and graphDB handlers that give control to all modules to store and retrieve data from the databases. The progress on this module is 70%.

## Transformers Module:
The transformers module consists of various tools for text and graph transformation to perform tasks such as rumor detection, link prediction, rumor propagation, and rumor influence minimization. The progress on this module is 10%.

## Rumor Detection Module:
The rumor detection module is constituted of several deep learning graph neural network-based models that allow the framework to detect and classify text as fake or not fake and rumor or not rumor. This module is not yet ready, and progress is at 0%.

## Simulation Module:
The simulation module allows the framework to simulate rumor propagation based on HISB model. The module can simulate on different real or synthetic graphs, with the graphs stored in the databases mentioned earlier. This module also allows us to test rumor influence minimization strategies such as blocking nodes and truth campaign strategies and provides statistical analysis. The progress on this module is 80%.

Progress:
Data Extraction Module - 80% [████████████████████░░░░░░░░░░░]

Data Storage Module    - 70% [█████████████████░░░░░░░░░░░░░░]

Transformers Module    - 60% [████████████░░░░░░░░░░░░░░░░░░░]

Rumor Detection Module - 60% [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]

Simulation Module      - 80% [████████████████████░░░░░░░░░░░░]

Overall Progress       - 48% [██████████░░░░░░░░░░░░░░░░░░░░░░]

Progress:
Data Extraction Module - 80%
Data Storage Module - 70%
Transformers Module - 10%
Rumor Detection Module - 0%
Simulation Module - 80%
Overall Progress - 48%

Conclusion:
Our project aims to provide a framework to combat the spread of fake news and rumors on online social networks. Our project is currently in development and is divided into five modules, each serving a specific purpose. We plan to continue working on this project and hope to achieve our goal of making online social networks a safer place for everyone.

