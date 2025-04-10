
# 🎯 Emotion Recognition ML on the Basis of Smartphone Accelerometer Sensors

✨ *Feel Your Words in Color* ✨

Hi 👋, 

I tried to find a correlation between emotional states (or *vibes*, as they are referred to in the deployed mobile applications) and the way the user moves the phone while typing (for example, a message) 📱. I wanted to represent these emotions then in color 🎨 (an accompanying color bubble after each message on WhatsApp, for example 💬).

I collected data from myself in different emotional states, ending up with around an hour worth of data for each emotion ⏱️. In this approach, I identified the emotions as calm/relaxed 🧘, frustrated 😤, happy 😄, anxious 😰 and sad 😢. The data was collected from both accelerometer and gyroscope sensors at 50Hz 📊 on a custom coded application as I was typing on an Android Samsung AG52 device 🤖. I visualized the collected data in Excel to remove outliers visible by eye 👀. The difference between gyroscope sensor data for different emotional states was insignificant ❌ and I did not preprocess it further, focusing on the accelerometer data ✅. In line with the previous statement, I refrained from using the Kalman filter, as well as the Butterworth or complementary filters ⚙️. The accelerometer data for five emotions was then handled manually for feature selection 🛠️. I examined the graphs visually and selected parameters I thought distinguished the graphs from each other the most 📈. I came up with ranges for each selected feature that was acceptable for that feature based on the emotion 🧠. If a feature range overlapped significantly between different emotions, I got rid of that feature ❌. This was done all done manually. This is also how I further removed outliers from the initial data set (accel_feature_all_axes_new.xlsx). No synthetic data was introduced 🚫. Lastly, I trained a very simple Random Forest model 🌳 with an 85% accuracy and deployed it in my applications 🚀. 

📎 [The raw data for the gyroscope and accelerometer sensors for myself can be accessed here on Google Drive](https://docs.google.com/spreadsheets/d/1zumlTXjLASAonMdBG7jH85BS_jsIeyqu/edit?usp=share_link&ouid=106124660547835903984&rtpof=true&sd=true) *(it takes long to load because of the graphs)*.

📎 [Or you can get the data sets of my own data on Hugging Face](https://huggingface.co/datasets/wolfeiq/accelerometer_raw_emotion_recognition_smartphone_sensors/tree/main)

📎 [Here is the Random Forest Model on Hugging Face](https://huggingface.co/wolfeiq/random_forest_emotion_recognition/tree/main)

📁 The feature selection data is uploaded in this repository, as well as the Random Forest training file. 

In the second approach, I collected data from 10 participants 👥 for three emotions (relaxed, frustrated, happy). I similarly dismissed the gyroscope data 🌀. The Random Forest model showed a similar accuracy of around 83% 📉 for this accelerometer data set. I trained four different CNN models 🧠 to experiment which one could perform the best. I picked out a simple baseline one, a deep CNN, an attention-based and a residual model 🔬. The baseline and attention-based models were optimized for their hyperparameters with the Optuna framework 🔧. At the end, all things equal, the baseline CNN performed best after Optuna optimization 🏆. The file is accessible here. The data that was fed in was minimally preprocessed and not denoised in any way ✂️. However, the CNNs were trained on the 3 input data, i.e. 3 emotions and not 5. The optimized CNN model has an accuracy of 95%. Neither the data nor the model will be published due to privacy concerns. 

However, what I can say is this, the idea works pretty well for each specific user, as in creating a generalized model is pretty difficult. So, in the futture, it would make sense to make it a very personal thing to each user. This is not the case in my applications, unless you purchase the Premium plan. In general, creating generalized emotion recognition models and methods is hard. For whatever approach of collecting data, be it through electrodermal sensors or facial recognition.


### 🛠️ My projects using the RF model currently in action:

- 🌐 [phatedapp.com](https://phatedapp.com)  
- 🌐 [portablehackerhouse.com](https://portablehackerhouse.com)

🛡️ The data is not stored on servers, unless the Premium option is purchased in the Phated app to fine-tune the model. It is processed ephemerally.
