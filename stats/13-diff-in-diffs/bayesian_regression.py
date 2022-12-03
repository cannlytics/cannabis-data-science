

#--------------------------------------------------------------------------
# TODO: Create a re-usuable Bayesian linear regression!!!
#--------------------------------------------------------------------------



#--------------------------------------------------------------------------
# Estimate statistics with Bayesian methods.
# P(Θ|X) = P(X|Θ)*P(Θ) / P(X)
# where:
# p(X|Θ) = p(X|μ) ~ N(μ, σ²)
#--------------------------------------------------------------------------

# TODO: Sample a bigger and bigger sample and see if estimates become more confident.


# Estimate prices.



#--------------------------------------------------------------------------
# Naive Bayes Algorithm.
#--------------------------------------------------------------------------

# # Splitting the dataset into the Training set and Test set
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# # Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

# # Training the Naive Bayes model on the Training set
# from sklearn.naive_bayes import GaussianNB
# classifier = GaussianNB()
# classifier.fit(X_train, y_train)

# # Predicting the Test set results
# y_pred = classifier.predict(X_test)

# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix, accuracy_score
# ac = accuracy_score(y_test,y_pred)
# cm = confusion_matrix(y_test, y_pred)