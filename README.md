## Pipeline:
In this repository, we provide a framework for water level prediction. 
1.	We have used pressure, wind, and tide data to predict the water level.
2.	We have used CNNs, LSTMs, and CNN-LSTMs as the prediction model.
3.	For the explanation, we applied the permutation method, where we remove each feature and find the effect of that on the prediction accuracy.

## How to run the code?
•	The experiments were run on Google Colab using an A100GPU 

•	**You need to first run .py files in 1_Concatenation folder. Then run the .py files in 2_Preprocessing and afterward .py files in 3_Prediction

•	**To make it easier, you can download all the files in 3_Prediction folder and run the .py files there. We have put all the data files after preprocessing in 3_Prediction folder.

•	**For the explanation part, you can easily remove the features that you want from the 1_prediction-main.py file in 3_Prediction folder. It is expressed as a comment in this .py file. Only to remove "tide" feature, we you need to run 4_XAI_without_tide.py file. 

•	**If you use this code in your work, please cite our paper "A Comparative Study of Convolutional and Recurrent Neural Networks for Storm Surge Prediction in Tampa Bay”, 2026. 🤗

