# Yelp Restaurant Exploration
## Introduction
A person's choice in dining has always been so subjective.  Each of us has our own preferences and biases when it comes to food, and so the phrase "Where should we eat?" has launched countless debates and stalled decisions.  Yelp has offered us a solution to that age-old debate. Crowd-consensus has provided us with objective evidence that this restaurant here is truly the one for us.

But how much can we truly learn from crowd-consensus?  Is a five-star restaurant with 10 reviews truly the better choice over a four-star with 500?  Are there any features about a restaurant that we might use to predict how it might be rated?  In this project, I'll explore these questions.

*Note: You can check out the full code (with commentary!) in the [Yelp Restaurant Exploration](/notebooks/Yelp%20Restaurant%20Exploration.ipynb) notebook.*
## The Dataset
[Yelp and Kaggle](https://www.kaggle.com/yelp-dataset/yelp-dataset) have partnered to provide us with an extensive dataset around Yelp's primary experiences - business information, user profiles, reviews, tips, and check-ins.   The data is organized into six .json files with over 5,000,000 entries between them.  However, in this project we will only focus on one - [yelp_academic_dataset_business.json](yelp_academic_dataset_business.json), which contains information on over 174,000 business in 11 metropolitan areas.

## Questions to Answer
- What is the distribution of star ratings for restaurants on Yelp?
- Does a restaurant's rating really tell us it's the right choice? How does the number of reviews factor into a rating?
- Using the attributes attached to a restaurant, how well can we predict its star rating?

## First Look

The data set includes information on a wide spectrum of businesses, but as true foodies, we are really only interested in the restaurants.  We'll have to isolate these from the data set, but let's first take a look at the general structure.

```json
{
    "business_id":"1SWheh84yJXfytovILXOAQ",
    "name":"Arizona Biltmore Golf Club",
    "address":"2818 E Camino Acequia Drive",
    "city":"Phoenix",
    "state":"AZ",
    "postal_code":"85016",
    "latitude":33.5221425,
    "longitude":-112.0184807,
    "stars":3.0,
    "review_count":5,
    "is_open":0,
    "attributes":{"GoodForKids":"False"},
    "categories":"Golf, Active Life",
    "hours":null,
}
```
```python
file = "D:\Directory\what_makes_a_hotspot\data\yelp_academic_dataset_business.json"
raw = pd.read_json(file, lines=True)
raw.info()
```
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 192609 entries, 0 to 192608
Data columns (total 14 columns):
address         192609 non-null object
attributes      163773 non-null object
business_id     192609 non-null object
categories      192127 non-null object
city            192609 non-null object
hours           147779 non-null object
is_open         192609 non-null int64
latitude        192609 non-null float64
longitude       192609 non-null float64
name            192609 non-null object
postal_code     192609 non-null object
review_count    192609 non-null int64
stars           192609 non-null float64
state           192609 non-null object
dtypes: float64(3), int64(2), object(9)
memory usage: 20.6+ MB
```
## Isolating Restaurants
We've found a golf course in Phoenix, Arizona, which apparently is not good for kids.  We can see that it has a unique identifier called `business_id`, and that the `attributes` and `categories` fields contain nested data.  We will have to get at this data in our exploration.

We can also see that we have pretty good coverage throughout the fields.  `hours`, `attributes`, and `categories` contain varying amounts of missing fields.  This will need to be accounted for, as well.  Let's first try to isolate our restaurants.

```python
def retrieve_restaurants(file):
    """
    Iterate over JSON file and funnel restaurants to a dataframe for analysis
    """
    # Retrieve JSONReader generator
    reader = pd.read_json(file, lines=True, chunksize=1000)
    
    # Define empty dataframe for storing matches
    cols = ['address', 'attributes', 'business_id', 'categories', 'city', 'latitude', 'longitude', 
            'name', 'postal_code', 'review_count', 'stars', 'state', 'restaurant']
    
    restaurants = pd.DataFrame(columns=cols)
    
    # Iterate over chunks, and funnel matches into dataframe
    for chunk in reader:
        
        chunk["restaurant"] = chunk["categories"].apply(lambda x:
            "Restaurants" in x.replace(" ", "").split(",") if x is not None else False
        )
        
        restaurants = pd.concat([restaurants, chunk[chunk["restaurant"]]], sort=True)
    
    # Set index to business_id
    restaurants.set_index("business_id", inplace=True)
        
    return restaurants
    
restaurants = retrieve_restaurants(file)
restaurants.head(1)['attributes'].values[0]
```
```
{'RestaurantsReservations': 'True',
 'GoodForMeal': "{'dessert': False, 'latenight': False, 'lunch': True, 
                  'dinner': True, 'brunch': False, 'breakfast': False}",
 'BusinessParking': "{'garage': False, 'street': False, 'validated': False, 
                      'lot': True, 'valet': False}",
 'Caters': 'True',
 'NoiseLevel': "u'loud'",
 'RestaurantsTableService': 'True',
 'RestaurantsTakeOut': 'True',
 'RestaurantsPriceRange2': '2',
 'OutdoorSeating': 'False',
 'BikeParking': 'False',
 'Ambience': "{'romantic': False, 'intimate': False, 'classy': False, 'hipster': False, 
               'divey': False, 'touristy': False, 'trendy': False, 'upscale': False, 
               'casual': True}",
 'HasTV': 'False',
 'WiFi': "u'no'",
 'GoodForKids': 'True',
 'Alcohol': "u'full_bar'",
 'RestaurantsAttire': "u'casual'",
 'RestaurantsGoodForGroups': 'True',
 'RestaurantsDelivery': 'False'}
```
## Parsing and Cleaning

Now we're talking.  Dim sum in Mississa-something!  Ouch on the 2.5 stars though.  At least there's a full bar.

Digging in to the `attributes` column, we see there is a wealth of information - the type of meals a restaurant is good for, ambience, alcohol service, noise level, even wifi availability!  Some fields like `Ambience` even have their own nested dictionaries.  In order to properly analyze this set, we'll need to normalize these attributes into our dataframe.

```python
def parse_nested(x):
    """
    Assist in removing some of the artifacts preventing clean parsing of nested data
    """
    
    if x is None:
        return '{"None": True}'
    elif x == "None":
        return '{"None": True}'
    else:
        return x
    
def normalize_nested(df, col):
    """
    Clean a nested column within the attributes field, convert from string to a dictionary, 
    and normalize to fit the data set
    """
    
    norm = df["attributes"].apply(lambda x: x.get(col) if x is not None else '{"None": 1}')
    
    norm = norm.apply(parse_nested)
    
    norm = norm.apply(literal_eval)
    
    norm = json_normalize(norm).set_index(df.index)
    
    for c in norm.columns:
        # Rename columns with prefix
        new_name = col.lower()+"_"+c
        norm.rename(columns={c: new_name}, inplace=True)
        
        # Set True/False to 1/0
        norm[new_name] = norm[new_name].apply(lambda x: 1 if x is True else 0)
        
    norm.fillna(0, inplace=True)
    
    return df.merge(norm, left_index=True, right_index=True)

def parse_attributes(df):
    """
    Parse attribute fields that do not contain nested data and remove unicode artifacts
    """
    
    # Selected Attributes
    attr = ["NoiseLevel", "RestaurantsTableService", "RestaurantsTakeOut", "OutdoorSeating", "Alcohol", 
            "RestaurantsAttire", "RestaurantsDelivery", "RestaurantsGoodForGroups", "GoodForKids"]

    for a in attr:
        
        col = a.lower()
        
        # Set as column
        df[col] = df["attributes"].apply(lambda x: x.get(a) if x is not None else 0)
        
        # Remove unicode references
        df[col] = df[col].str.replace("u'", "").str.replace("'", "")
                
    return df
    
def parse_dummies(df):
    """
    Parse string columns in Attributes field and convert to dummy variables
    """
    
    # Dummy columns
    dummies = ["noiselevel", "alcohol", "restaurantsattire"]

    for d in dummies:
        
        # Get dummies and merge on dataframe
        df = df.merge(pd.get_dummies(df[d], prefix=d), left_index=True, right_index=True)
        
    return df

def format_restaurants(df):
    """
    Normalized nested field, parse attributes, and create dummy columns where necessary.  Remove 
    columns that are extraneous to our analysis, and ensure all fields are either integer or float.
    """
    
    # Normalize Ambience columns
    df = normalize_nested(df, "Ambience")
    
    # Normalize GoodForMeal columns
    df = normalize_nested(df, "GoodForMeal")
    
    # Parse attribute columns
    df = parse_attributes(df)
    
    # Parse dummies
    df = parse_dummies(df)
    
    # Drop extraneous columns
    drop_cols = ['attributes', 'noiselevel', 'alcohol', 'restaurantsattire', 'address', 'categories', 
                 'city', 'latitude', 'longitude', 'name', 'postal_code', 'state', 'restaurant', 'hours',
                 'is_open', 'ambience_None', 'goodformeal_None', 'noiselevel_None', 'alcohol_None',
                 'restaurantsattire_None']
    
    df.drop(columns=drop_cols, inplace=True)
    
    # Set review_count to int
    df["review_count"] = df["review_count"].astype(int)
    
    # If column is still an object, set values to binary
    for col in df.columns:
        if df[col].dtype in [object, bool]:
            df[col] = df[col].apply(lambda x: 1 if x == 'True' else 0)
    
    return df
    
clean = format_restaurants(restaurants)
```
## Cleaned Data
Wow! That was a lot of work!  There were a few eccentricities in the attributes field that required some extra attention.  Mainly, some of the nested dictionaries were inconsistently formatted, which required some additional parsing logic.  Regardless, we now have a normalized dataset with dummy variables for many of the nested items within the attributes field.  Our data is also now entirely numeric, which should make the next analysis steps much smoother.

## Distribution of Star Ratings
![dist1](img/Distribution%20of%20Star%20Ratings.png "Distribution of Star Ratings")
![dist2](img/Cumulative%20Distribution%20of%20Star%20Ratings.png "Cumulative Distribution of Star Ratings")


The distribution of star ratings appears to be slightly skewed towards higher ratings, though fives remain relatively rare.  It seems Yelp reviewers trend towards the "good, but not great" view on most restaurants - at least in the cities we have available to us.  Looking at the cumulative distribution chart, we can see just about 60% of restaurants fall at 3.5 stars or less.

## The Impact of Review Count

But let's dive a bit deeper and see how review count factors in.  Do we see restaurants with a higher number of reviews regress towards the mean of rating? Are there some that resist the pull towards the center?

![dist3](img/Distribution%20of%20Review%20Count%20by%20Star%20Ratings.png "Distribution of Review Count by Star Rating")

Interesting!  We can see our middle-of-the-pack restaurants typically have a higher count of reviews than those on the edges.  One star reviewed restaurants have the least reviews.  This might be from bad reviews driving other Yelpers away, or it could just be that nobody posts a review for the McDonald's on Exit 73 unless they've had a really bad experience.  Five star reviewed restaurants have the second-lowest count, though we see a long tail of outliers that may prove interesting.  Note, the log of review counts was taken here to diminish the impact of outlier restaurants with thousands of reviews.

## Proposing a New Metric

So we see that five-star restaurants typically have smaller review counts than those in the middle of the pack.  What does that tell us?  Can we really trust a rating when only five reviews have been entered?  Wouldn't the 4.5 star restaurant with one hundred reviews be a much safer choice?

The concept of weighted scoring has been brought up by Yelpers before, but usually to the tune of reactions like this -

![Imgur](https://i.imgur.com/7KSEkcw.png)

Fortunately, the folks at [math.stackexchange](https://math.stackexchange.com/questions/942738/algorithm-to-calculate-rating-based-on-multiple-reviews-using-both-review-score) operate more on my frequency and offered the following formula for producing a weighted score -

![formula](img/Formula.PNG "Formula")

To paraphrase, we can combine the notions of quality (`p` for star rating) and quantity (`q` for review count) by assigning each of them weights and normalizing onto a similar scale. `P` becomes the weight (between 0 and 1) we assign to our star rating.  The review count becomes embedded in an exponential function with it's own weight, `Q`; and the output of this is multiplied by the inverse of star's `P` along with the constant `10` make our score function on a nice 1 to 10 scale.  We add these together to get our weighted rating!

Got that?  I recommend reading [the post](https://math.stackexchange.com/questions/942738/algorithm-to-calculate-rating-based-on-multiple-reviews-using-both-review-score) again.  It took me a few times as well.  Anywho, without further ado, let's introduce our restaurant heatscore™!

## Heatscore!™

```python
def calculate_heatscore_tm(x, p, q):
    
    return (p * x["stars"]) + (10*(1 - p) * (1 - np.exp(-(x["review_count"])/q)))
    

clean["heatscore"] = clean.apply(calculate_heatscore_tm, axis=1, args=(0.7, 140))

clean.sort_values("heatscore", ascending=False).head(15)[["review_count", "stars", "heatscore"]]
```
| business_id            | review_count | stars | heatscore |
|------------------------|--------------|-------|-----------|
| Xg5qEQiB-7L6kGJ5F4K3bQ | 1936         | 5.0   | 6.499997  |
| IhNASEZ3XnBHmuuVnWdIwA | 1506         | 5.0   | 6.499936  |
| SSCH4Z2gw-hh2KZy7aH4qw | 552          | 5.0   | 6.441822  |
| ewmTwsZqCHH2gvCeDKz0dw | 543          | 5.0   | 6.437959  |
| q2GzUNQj998GSC8IhkN9hg | 459          | 5.0   | 6.386954  |
| 1qkKfqhO8z2XMzLLDFE96Q | 414          | 5.0   | 6.344098  |
| 52yWGkwnrQXIjvuMjYxsiA | 402          | 5.0   | 6.330146  |
| 8fFTJBh0RB2EKG53ibiBKw | 374          | 5.0   | 6.292540  |
| 3pSUr_cdrphurO6m1HMP9A | 363          | 5.0   | 6.275582  |
| O7UMzd3i-Zk8dMeyY9ZwoA | 350          | 5.0   | 6.253745  |
| xCL38K0oPgK3ydzg4CrvKg | 340          | 5.0   | 6.235512  |
| cePE3rCuUOVSCCAHSjWxoQ | 313          | 5.0   | 6.179253  |
| DkYS3arLOhA8si5uUEmHOw | 5075         | 4.5   | 6.150000  |
| faPVqws-x-5k2CQKDNtHxw | 3576         | 4.5   | 6.150000  |
| hihud--QRriCYZw1zZvW4g | 3449         | 4.5   | 6.150000  |

With the heatscore™ applied, we can see our top restaurants remain those with five stars.  However, around entry thirteen, we start to see the 4.5 star restaurants with very high volumes of reviews. Our formula appears to be working!  Let's play with the parameters a bit to see how they impact the scoring.

![heatscore1](img/Heatscore%20Comparision%20-%20High%20P%20Low%20Q.png "High P Low Q")

A very high `P` and a low `Q` results in a scoring that's fairly close to the standard ratings.  Low star ratings are penalized severely, while high ratings get a huge boost.  Large review counts provide some boost, but not really enough to lift them above restaurants with a higher star review.

![heatscore2](img/Heatscore%20Comparision%20-%20Low%20P%20High%20Q.png "Low P High Q")

Taken to the other extreme, a low `P` and large `Q` all but removes the influence of star scores.  Even very poorly rated restaurants can acheive high scores if their review counts are large.  Let's try to find a happy medium.

![heatscore3](img/Heatscore%20Comparision%20-%20Final.png "Final")

This seems to work pretty well.  A slight preference for star reviews with a 0.65 `P` ensures we don't have two-star restaurants out-performing five-star ones (assuming that have any reasonable number of reviews), and our `Q` value ensures we factor in those highly-popular restaurants whose reviews may have regressed slightly towards the mean.  Long live the heatscore™!

## Predicting a Restaurant's Score

But enough about that.  We spent all that time parsing attribute data, and we've done nothing with it!  Let's see if we have the necessary data at hand to accurately predict a given restaurant's star score.  We'll start by taking a look at the correlation among our features.
![corr](img/Correlation%20Matrix.png "Correlation Matrix")

Our heatscore™ feature definitely does not belong in our modeling set, since it's partially derived from our target variable.  We see some of the ambience variables correlating with certain meal types, along with brunch and breakfast correlating heavily.

Let's first finish up our preprocessing.  We'll first split our data into feature (X) and target (y) sets, dropping our most excellent new heatscore™ metric, as it will likely be heavily correlated to our target star rating variable.  

Next, though the star rating is an integer, it is not continuous.  Because of this, we should consider this a classification problem.  Multiplying the rating by two will give us rounded integers from 1-10, which will allow us pass them to most any classification algorithm.

Next, we'll scale our feature set (really only `review_count`, since our remaining features were all converted to binary dummy variables); and split everything into testing and training sets.

## Modeling

As a first pass, we'll use an ensemble classification model called Gradient Boost.  Ensemble methods iteratively create "weak" learning models such as a decision tree, calculate their error, and use that error inform the next iteration of the model.  This combination of stochastic gradient descent and bootstrapping (hence "Gradient Boost"!) can generate strong results with the right parameter tuning.

```python
X_train, X_test, y_train, y_test = preprocess_data(clean, 0.3, 99)
train_pred, test_pred, train_score, test_score, model = predict_ratings(
        X_train, X_test, y_train, y_test, 99)

print(f"Training Accuracy - {round(train_score, 2)}")
print(f"Test Accuracy - {round(test_score, 2)}")
```
```
Training Accuracy - 0.31
Test Accuracy - 0.29
```
![distcomp](img/Distribution%20of%20Star%20Ratings%20-%20Predicted%20vs%20Actual.png "Distribution Comparison")

Our initial model accurately selected the star rating 29% of the time on our test set.  This is better than random chance (10%), but still less than spectacular.  We can see in the histogram comparison, that our model overly favors 3.5 and 4 star ratings.  The edge ratings almost never received a prediction.

![conf](img/Confusion%20Matrix.png "Confusion Matrix")

The confusion matrix confirms our initial observations.  The model does well predicting 3.5 and 4 star restaurants, specifically.  However, we can see most of the edge reviews get clustered into these categories, as well.

![feat](img/Feature%20Importance.png "Feature Importance")

## Steps for Improvement

Plotting the feature importance is very telling.  Review count overwhelms every other feature in our data set!  In future attempts, we might be better off excluding this feature entirely, or perhaps this is a good indicator that our heatscore™ would be a more appropriate target variable.

Either way, this serves as a good baseline, but there's a lot of room for improvement.  What could our next steps be to make it more robust?

- We removed all of the geographic information in the initial cleaning of the data set.  How might we incorporate this?  Are reviewers in Québec more generous in their ratings than those in Nevada?  What if we brought in additional geographic data such as census information to tell us more about the neighborhoods these restaurants could be found in?
- We also removed the category field after using it to identify restaurants.  There are a lot of these, which might cause some issues with over-dimensionality, but perhaps there are some subcategories that flag more popular restaurants.  Maybe Arizonans are enthusiastic about barbeque in particular.
- Dimensionality reduction techniques such as principle components analysis might help us identify clusters of useful features.
- Finally, parameter tuning can be especially impactful on algorithms like gradient boost.  Grid search techniques might help us identify the optimal parameters for our model and data.

How else might we make the model more robust?

## Bonus!

Let's use our heatscore™ to find the best restaurant options in a specific city.  Las Vegas seems a good choice...

```python
# Define the desired columns
cols = ["name", "categories", "address", "city", "stars", "review_count", "heatscore"]

# Apply heatscore™ to our restaurants dataset
p, q = 0.65, 100
restaurants["heatscore"] = restaurants.apply(calculate_heatscore_tm, axis=1, args=(p, q))
restaurants.sort_values(by=["heatscore"], ascending=False, inplace=True)

# Isolate Las Vegas and output the top 10
vegas = restaurants[restaurants['city'] == 'Las Vegas']
vegas.head(10)[cols]
```

| name                          | categories                                                                                                                     | address                        | city      | stars | review_count | heatscore         |
|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------|--------------------------------|-----------|-------|--------------|-------------------|
| Brew Tea Bar                  | Tea Rooms, Desserts, Cafes, Restaurants, Food, Bubble Tea                                                                      | 7380 S Rainbow Blvd, Ste 101   | Las Vegas | 5.0   | 1506         | 6.749998991692154 |
| Zenaida's Cafe                | Cafes, Breakfast & Brunch, Restaurants                                                                                         | 3430 E Tropicana Ave, Ste 32   | Las Vegas | 5.0   | 374          | 6.666860639040433 |
| J Karaoke Bar                 | Bars, Korean, Restaurants, Asian Fusion, Cocktail Bars, Nightlife, Karaoke, American (New)                                     | 3899 Spring Mountain Rd        | Las Vegas | 5.0   | 363          | 6.657193354568871 |
| Art of Flavors                | Gelato, Ice Cream & Frozen Yogurt, Restaurants, American (New), Desserts, Food                                                 | 1616 S Las Vegas Blvd, Ste 130 | Las Vegas | 5.0   | 350          | 6.644309158021885 |
| Karved                        | Fast Food, Restaurants, Sandwiches, American (New), American (Traditional)                                                     | 3957 S Maryland Pkwy           | Las Vegas | 5.0   | 313          | 6.596987709615371 |
| Bajamar Seafood & Tacos       | Dive Bars, Restaurants, Bars, Tacos, Fast Food, Nightlife, Mexican, Seafood                                                    | 1615 S Las Vegas Blvd          | Las Vegas | 5.0   | 282          | 6.541379200518712 |
| Lip Smacking Foodie Tours     | Hotels, Food, Event Planning & Services, Hotels & Travel, Walking Tours, Food Tours, Restaurants, Tours                        | 3635 Las Vegas Blvd S          | Las Vegas | 5.0   | 257          | 6.48212559101631  |
| Earl of Sandwich              | Food Delivery Services, Salad, Sandwiches, Soup, Food, Event Planning & Services, American (New), Restaurants, Caterers, Wraps | 3667 Las Vegas Blvd S          | Las Vegas | 4.5   | 5075         | 6.425000000000001 |
| Yardbird Southern Table & Bar | Restaurants, American (New), Southern                                                                                          | 3355 Las Vegas Blvd S          | Las Vegas | 4.5   | 3576         | 6.424999999999999 |
| Gangnam Asian BBQ Dining      | Barbeque, Korean, Asian Fusion, Tapas/Small Plates, Japanese, Restaurants                                                      | 4480 Paradise Rd, Ste 600      | Las Vegas | 4.5   | 3449         | 6.424999999999997 |

There you have it! Brew Tea Bar is blowing every other restaurant out of the water, and it isn't even on the Strip!

#### Citations:

Bartulis, Andrius. “Algorithm to Calculate Rating Based on Multiple Reviews (Using Both Review Score and Quantity).” Mathematics Stack Exchange, math.stackexchange.com/questions/942738/algorithm-to-calculate-rating-based-on-multiple-reviews-using-both-review-score.

“Confusion Matrix.” Scikit, scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py.

“Understanding Gradient Boosting Machines.” Towards Data Science, Towards Data Science, 3 Nov. 2018, towardsdatascience.com/understanding-gradient-boosting-machines-9be756fe76ab.

Yelp, Inc. “Yelp Dataset.” Kaggle, 5 Feb. 2019, www.kaggle.com/yelp-dataset/yelp-dataset.