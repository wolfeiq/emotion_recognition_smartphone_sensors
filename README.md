Emotion Recognition ML on the Basis of Smartphone Accelerometer Sensors

Hi, 

I tried to find a correation between emotional states (or vibes, as they are referred to in the deployed mobile applications) and the way the user moves the phone while typing (for example, a message). I wanted to represent these emotions then in color (an accompanying color bubble after each message on WhatsApp, for example).

I collected data from myself in different emotional states, ending up with around an hour worth of data for each emotion. In this approach, I identified the emotions as calm/relaxed, frustrated, happy, anxious and sad. The data was collected from both accelerometer and gyroscope sensors at 50Hz on a custom coded application as I was typing on an Android Samsung AG52 device. I visualized the collected data in Excel to remove outliers visible by eye. The difference between gyroscope sensor data for different emotional states was insignificant and I did not preprocess it further, focusing on the accelerometer data. In line with the previous statement, I refrained from using the Kalman filter, as well as the Butterworth or complementary filters. The accelerometer data for five emotions was then handled manually for feature selection. I examined the graphs visually and selected parameters I thought distinguished the graphs from each other the most. I came up with ranges for each selected feature that was acceptable for that feature based on the emotion. If a feature range overlapped significantly between different emotions, I got rid of that feature. This was done all done manually. This is also how I further removed outliers from the intial data set. No synthetic data was introduced. Lastly, I trained a very simple Random Forest model with an 85% accuracy and deployed it in my applications. 

The raw data for the gyroscope and accelerometer sensors for myself can be accessed here (it takes long to load because of the graphs): https://docs.google.com/spreadsheets/d/1zumlTXjLASAonMdBG7jH85BS_jsIeyqu/edit?usp=share_link&ouid=106124660547835903984&rtpof=true&sd=true

The feature selection data is uploaded in this repository, as well as the Random Forest training file. 

In the second approach, I collected data from 10 participants for three emotions (relaxed, frustrated, happy). I similarly dismissed the gyroscope data. The Random Firest modell showed a similar accuracy of around 83% for this accelerometer data set. I trained four different CNN models to experiment which one ould perform the best. I picked out a simple baseline one, a deep CNN, an attention-based and a residual models. The baseline and attention-based models were optimized for their hyperparameters with the Optuna framework. At the end, all things equal, the baseline CNN performed best after Optuna optimization. The file is accessible here. The data that was fed in was minimally preprocessed and not denoised in any way. However, the CNNs were trained on the 3 input data, i.e. 3 emotions and not 5. 


My projects using the RF model currently in action: 

phatedapp.com
portablehackerhouse.com

The data is not stored on servers, unless the Premium option is purchased in the Phated app to fine-tune the model. It is processed ephemerally.
