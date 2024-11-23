# Task
Predict vinho verde white wine quality based on 11 features using regression model like linear regression, decision tree regression and random forest. Will be used to predict wine quality based on the given features.



# Information about the dataset
Two datasets are included, related to red and white vinho verde wine samples, from the north of Portugal. The goal is to model wine quality based on physicochemical tests (see [Cortez et al., 2009], http://www3.dsi.uminho.pt/pcortez/wine/).

These datasets can be viewed as classification or regression tasks.  The classes are ordered and not balanced (e.g. there are many more normal wines than excellent or poor ones). Outlier detection algorithms could be used to detect the few excellent or poor wines. Also, we are not sure if all input variables are relevant. So it could be interesting to test feature selection methods.

For more information, read [Cortez et al., 2009].
Input variables (based on physicochemical tests):
1. fixed acidity
2. volatile acidity
3. citric acid
4. residual sugar
5. chlorides
6. free sulfur dioxide
7. total sulfur dioxide
8. density
9. pH
10. sulphates
11. alcohol
    

Output variable (based on sensory data):

12.  quality (score between 0 and 10)



# Create venv and install all the reqs (for *nix based OS)
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```



# Build container and run it
```
docker build -t wq-app .
docker run -p 8000:8000 wq-app
```



# Test
```
POST https://wq-app-g8aja4eggreugrhs.francecentral-01.azurewebsites.net/predict
```

Body
```
{
    "fixed_acidity": 8.1,
    "volatile_acidity": 0.28,
    "citric_acid": 0.4,
    "chlorides": 0.05,
    "free_sulfur_dioxide": 30.0,
    "total_sulfur_dioxide": 97.0,
    "density": 0.9951,
    "ph": 3.26,
    "sulphates": 0.44,
    "alcohol": 13.1
}
```

Response
```
{
    "quality": 6.4
}
```
