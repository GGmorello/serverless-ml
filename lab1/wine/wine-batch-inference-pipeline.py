import os
#import modal
import random
    
LOCAL=True

# if LOCAL == False:
#    stub = modal.Stub()
#    hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","sklearn==1.1.1","dataframe-image"])
#    @stub.function(image=hopsworks_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
#    def f():
#        g()

def g():
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

    project = hopsworks.login(project= "Scalable_ML_lab1")
    fs = project.get_feature_store()
    
    mr = project.get_model_registry()
    model = mr.get_model("wine_model", version=5)
    model_dir = model.download()
    model = joblib.load(model_dir + "/wine_model.pkl")
    
    feature_view = fs.get_feature_view(name="wine", version=1)
    batch_data = feature_view.get_batch_data()

    print("batch_data", batch_data)
    
    y_pred = model.predict(batch_data)
    #print(y_pred)
    
    #offset = 1
  
    offset =  random.randint(1, y_pred.size)
    quality = y_pred[y_pred.size-offset]
    #print("printed quality", quality)

    #flower_url = "https://raw.githubusercontent.com/featurestoreorg/serverless-ml-course/main/src/01-module/assets/" + quality + ".png"
    image_url = "https://raw.githubusercontent.com/GGmorello/serverless-ml/main/lab1/wine/numbers/" + str(quality) + ".png"
    
    print("Wine predicted: " + str(quality))
    resp = requests.get(image_url, stream=True)
    print(resp)
    img = Image.open(requests.get(image_url, stream=True).raw)
    newsize = (100, 100)
    img = img.resize(newsize)            
    img.save("./latest_quality.png")
    dataset_api = project.get_dataset_api()    
    dataset_api.upload("./latest_quality.png", "Resources/images", overwrite=True)
   
    wine_fg = fs.get_feature_group(name="wine", version=1)
    df = wine_fg.read() 
    #print(df)
    label = df.iloc[-offset]["quality"]
    #label_url = "https://raw.githubusercontent.com/featurestoreorg/serverless-ml-course/main/src/01-module/assets/" + label + ".png"
    label_url = "https://raw.githubusercontent.com/GGmorello/serverless-ml/main/lab1/wine/numbers/" + str(int(label)) + ".png"
    print("Actual wine quality: " + str(label))
    resp = requests.get(label_url, stream=True)
    print(resp)
    img = Image.open(requests.get(label_url, stream=True).raw)
    img = img.resize(newsize)           
    img.save("./actual_quality.png")
    dataset_api.upload("./actual_quality.png", "Resources/images", overwrite=True)
    
    monitor_fg = fs.get_or_create_feature_group(name="wine_predictions",
                                                version=2,
                                                primary_key=["datetime"],
                                                description="Wine Prediction/Outcome Monitoring"
                                                )
    
    
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': [quality],
        'label': [label],
        'datetime': [now],
       }
    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})
    
    history_df = monitor_fg.read()
    # Add our prediction to the history, as the history_df won't have it - 
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])

    print("perasa")

    df_recent = history_df.tail(4)
    dfi.export(df_recent, './wine_df_recent.png', table_conversion = 'matplotlib')
    dataset_api.upload("./wine_df_recent.png", "Resources/images", overwrite=True)
    
    print(dataset_api.exists("Resources/images/wine_df_recent.png"))
    
    predictions = history_df[['prediction']]
    labels = history_df[['label']]

    # Only create the confusion matrix when our wine_predictions feature group has examples of all 6 quality values
    print("Number of different wine predictions to date: " + str(predictions.value_counts().count()))
    if predictions.value_counts().count() == 6:
        results = confusion_matrix(labels, predictions)
    
        df_cm = pd.DataFrame(results, range(3,9), range(3,9))
    
        cm = sns.heatmap(df_cm, annot=True)
        fig = cm.get_figure()
        fig.savefig("./wine_confusion_matrix.png")
        dataset_api.upload("./wine_confusion_matrix.png", "Resources/images", overwrite=True)
    else:
        print("You need 6 different quality predictions to create the confusion matrix.")
        print("Run the batch inference pipeline more times until you get 6 different wine flower predictions") 


if __name__ == "__main__":
    if LOCAL == True :
        g()
    # else:
    #     with stub.run():
    #         f()

