# Project Title
 Covid-19 new cases prediction
 - Found the dataset in Github and decided to do a project is to predict the number of Covid-19 cases that will come in the next 30 days . I used the **long-short term memory networks(LSTM)** model to do the deep learning before make the prediction.
 
 
 ![img](/resources/imagecv.png)
# Models used :
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white) 	![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) 
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)



## Project Description
 Steps involved in this project included:
 1. Data loading 
 - Load the read the dataset using pandas
 2. Data cleaning 
 - By using the Interpolation method, the NaN value inside the covid_new column can be filled.
 <br><br> ![img](/resources/EDA.PNG)
 3. Data preprocessing
 - Normalize the data using MinMax Scaler so that the data can be learned.
 4. Train-test split
 - Train test split is a model validation process that allows me to simulate how the chosen model would perform with my new data.
 5. Model Development(LSTM) & compilation
- The model architecture 
   <br> ![img](/resources/Capture.PNG)
 - Epoch loss graph from tensorboard
   <br> ![img](/resources/Epoch_loss.PNG)
 6. Evaluation & Prediction
 - The actual cases vs predicted cases graph
 <br><br> ![img](/resources/Capture2.PNG)
 - Using mean absolute percentage error(mape) metrics to understand a machine learning model's performance.
 <br><br> ![img](/resources/Evaluate.PNG)
 

 
  
 ### Acknowledgement
- Special thanks to the MOH for providing the dataset
 1. Source of dataset : GitHub - MoH-Malaysia/covid19-public: Official data on the COVID-19 epidemic in Malaysia. Powered by CPRC, CPRC Hospital System, MKAK, and MySejahtera.
 2. Source of Covid-19 image : https://economictimes.indiatimes.com/news/science/mathematical-modelling-gives-more-accurate-picture-of-covid-19-cases-study/articleshow/75121178.cms
