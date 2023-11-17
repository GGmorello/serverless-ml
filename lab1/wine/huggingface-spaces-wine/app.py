import gradio as gr
from PIL import Image
import requests
import hopsworks
import joblib
import pandas as pd

project = hopsworks.login(project = "Scalable_ML_lab1")
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("wine_model", version=5)
model_dir = model.download()
model = joblib.load(model_dir + "/wine_model.pkl")
print("Model downloaded")

def wine(fixed_acidity, volatile_acidity, citric_acid, residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,ph,sulphates,alchohol):
    print("Calling function")
#     df = pd.DataFrame([[sepal_length],[sepal_width],[petal_length],[petal_width]], 
    df = pd.DataFrame([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,ph,sulphates,alchohol]], 
                      columns=['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide','density','ph','sulphates','alcohol'])
    print("Predicting")
    print(df)
    # 'res' is a list of predictions returned as the label.
    res = model.predict(df) 
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want 
    # the first element.
#     print("Res: {0}").format(res)
    print("the whole prediction", res)
    print("prediction[0]",res[0])
    # flower_url = "https://raw.githubusercontent.com/featurestoreorg/serverless-ml-course/main/src/01-module/assets/" + res[0] + ".png"
    image_url = "https://raw.githubusercontent.com/GGmorello/serverless-ml/main/lab1/wine/numbers/" + str(res[0]) + ".png"
    resp = requests.get(image_url, stream=True)
    print("image request: ",resp)
    img = Image.open(resp.raw)
    newsize = (100, 100)
    img1 = img.resize(newsize)
    return img1
        
demo = gr.Interface(
    fn=wine,
    title="Wine Quality Predictive Analytics",
    description="Experiment with the wine features to predict the quality of the wine",
    allow_flagging="never",
    inputs=[
        gr.Number(value=9.1, label="fixed_acidity"),
        gr.Number(value=0.3, label="volatile_acidity"),
        gr.Number(value=0.41, label="citric_acid"),
        gr.Number(value=2, label="residual_sugar"),
        gr.Number(value=0.068, label="chlorides"),
        gr.Number(value=10, label="free_sulfur_dioxide"),
        gr.Number(value=24, label="total_sulfur_dioxide"),
        gr.Number(value=0.99523, label="density"),
        gr.Number(value=3.27, label="ph"),
        gr.Number(value=0.85, label="sulphates"),
        gr.Number(value=11.7, label="alcohol")


        ],
    outputs=gr.Image(type="pil")
    #outputs = gr.Number(label = "quality")
)

demo.launch(debug=True,share = True)

