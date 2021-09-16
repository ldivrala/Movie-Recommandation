# Movie Recommandation

Dataset :: [MovieLens Latest Datasets](https://grouplens.org/datasets/movielens/ "MovieLens Latest Datasets")
Tools :: Tensorflow, Numpy, Pandas, Keras
Method :: Collaborative filtering  

* Dataset :: 
  * This dataset was collected and prepared by the GroupLens Reasearch.
  * We had used rating.csv which contains attributes like rating, movieId, userId etc.
  * It contains 100,000 ratings applied to 9,000 movies by 600 users.

* Inspiration::
  * Our aim was to try to analyze the dataset and try to figure out what can be predicted features for movies and user preferences.
  * Predict movie rating with our predicted features so we can recommand a new movie.
  * Is your customer like this movie?
  
  
##### Steps (What we did)
  1. Load our csv file with the help of pandas and see some information.
  2. Standardisation of rating so our gradient descent steps will be fast.
  3. We create our dataset :: \
    X = Movie Feature (20 features, initially random) \
    W = User Prefrence Feature(20 corresponding features, initially random) \
    Y = Movie-User Ratings \
    R = Check Ratings Available (Available = 1, Not Available = 0)
    
    In this dataset X and W are corresponding learnable parameters.
    
  4. Dataset loader creation with the help of tensorflow data api.
  5. Train our parameters with 10000 iterations: To learn best parameters which can predict our available rating prefect.
    
    Predicted_y = X @ tf.transpose(W) 
    
    Loss = sum(R * square(Predicted_y - Y)) / sum(R)
    
    Update X and W with SGD.
  
  6. Now we have our movie features and user features.
  7. We will compare our model predicted rating and default rating for initial 100 movies and 20 users. 
 


