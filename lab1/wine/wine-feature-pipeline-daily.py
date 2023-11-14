import pandas as pd
import random
import hopsworks
import os

def generate_wine(quality, fixed_acidity_range, volatile_acidity_range, citric_acid_range, 
                  residual_sugar_range, chlorides_range, free_sulfur_dioxide_range,
                  total_sulfur_dioxide_range, density_range, ph_range, sulphates_range, alcohol_range):
    """
    Generates a wine sample with specified characteristics
    """
    df = pd.DataFrame({
        "fixed_acidity": [random.uniform(fixed_acidity_range[0], fixed_acidity_range[1])],
        "volatile_acidity": [random.uniform(volatile_acidity_range[0], volatile_acidity_range[1])],
        "citric_acid": [random.uniform(citric_acid_range[0], citric_acid_range[1])],
        "residual_sugar": [random.uniform(residual_sugar_range[0], residual_sugar_range[1])],
        "chlorides": [random.uniform(chlorides_range[0], chlorides_range[1])],
        "free_sulfur_dioxide": [random.uniform(free_sulfur_dioxide_range[0], free_sulfur_dioxide_range[1])],
        "total_sulfur_dioxide": [random.uniform(total_sulfur_dioxide_range[0], total_sulfur_dioxide_range[1])],
        "density": [random.uniform(density_range[0], density_range[1])],
        "ph": [random.uniform(ph_range[0], ph_range[1])],
        "sulphates": [random.uniform(sulphates_range[0], sulphates_range[1])],
        "alcohol": [random.uniform(alcohol_range[0], alcohol_range[1])]
    })
    df['quality'] = quality
    return df

def get_random_wine():
    """
    Returns a DataFrame containing one random iris flower
    """
    virginica_df = generate_wine("Virginica", 8, 5.5, 3.8, 2.2, 7, 4.5, 2.5, 1.4)
    versicolor_df = generate_wine("Versicolor", 7.5, 4.5, 3.5, 2.1, 3.1, 5.5, 1.8, 1.0)
    setosa_df =  generate_wine("Setosa", 6, 4.5, 4.5, 2.3, 1.2, 2, 0.7, 0.3)

    pick_random = random.uniform(3, 8)
    if pick_random >= 2:
        iris_df = virginica_df
        print("Virginica added")
    elif pick_random >= 1:
        iris_df = versicolor_df
        print("Versicolor added")
    else:
        iris_df = setosa_df
        print("Setosa added")

    return iris_df

def main():
    # Use github secrets to pass in the feature store password

    project = hopsworks.login()
    fs = project.get_feature_store()

    wine_df = get_random_wine()

    iris_fg = fs.get_feature_group(name="wine", version=1)
    iris_fg.insert(wine_df)

if __name__ == "__main__":
    main()
