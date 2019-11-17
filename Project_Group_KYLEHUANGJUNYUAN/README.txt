NTU CZ4032 - Data Analytics and Mining (Group 27)
-------------------------------------------------
Topic: Predict house prices based on features provided using regression techniques & a neural network

Team Members
-------------------------------------------------
1) Kyle Huang Junyuan (U1721717G)
2) Russell Chua Kui Hon (U1720526F)
3) Nelson Ko Mingwei (U1721410B)
4) Ngoh Guang Wei (U1722281K)
5) Khin Yamin Thu (U1721925F)
6) Wilson Neo Yuan Wei (U1721538L)

Setup
-------------------------------------------------
All the codes for this project are written in Python 3. To install, download the correct .exe file from the official python website (https://www.python.org/downloads/release/python-370/). Next, add the python path to environment variables.

Install the relevant libraries:
Command lines : pip install -r requirements.txt

Running
-------------------------------------------------
All 5 regression models (Linear, SVR, Ridge, Random Forest, XGBoost)
Command: python regression_housing_prediction.py
Explanation: 
The program will prompt the user to select the following: 
			- Choice to print correlation matrix heatmap
			- Choice to view correlation of individual features to SalePrice through scatter plot
			- The regression model to be used
			- The preferred type of plot to be generated
The program will then output the error metrics and plots accordingly.

Neural Network
Command: python neural_housing_prediction.py
Explanation:
The program will automatically generate all the error metrics and plots will be generated and saved accordingly. 

