import pandas as pd
import hopsworks
import joblib
import datetime
from PIL import Image
from datetime import datetime
import dataframe_image as dfi
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot
import seaborn as sns
import requests
import random



project = hopsworks.login(project= "Scalable_ML_lab1")
fs = project.get_feature_store()

mr = project.get_model_registry()
model = mr.get_model("wine_model", version=5)
model_dir = model.download()
model = joblib.load(model_dir + "/wine_model.pkl")
dataset_api = project.get_dataset_api()
wine_fg = fs.get_feature_group(name="wine", version=1)
df = wine_fg.read() 

monitor_fg = fs.get_or_create_feature_group(name="wine_predictions",
                                            version=3,
                                            primary_key=["datetime"],
                                            description="Wine Prediction/Outcome Monitoring"
                                            )
    

feature_view = fs.get_feature_view(name="wine", version=1)
batch_data = feature_view.get_batch_data()

y_pred = model.predict(batch_data)

predictions_count = {key: 0 for key in range(3, 9)}
i = 0
while True:
    if all(value == 2 for value in predictions_count.values()): break
    offset =  random.randint(1, y_pred.size)
    quality = y_pred[y_pred.size-offset]
    i+=1
    if predictions_count[int(quality)] != 0: continue
    print("done one new prediction, i ={}".format(i))
    predictions_count[int(quality)]+=1

    label = df.iloc[-offset]["quality"]
    
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': [quality],
        'label': [label],
        'datetime': [now],
       }
    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})
    





