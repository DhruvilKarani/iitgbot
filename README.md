# iitgbot

## An FAQ bot that answers questions about IIT G

The data was scraped from various sites using Beautiful Soup and Requests libraries in Python (iitg_scrap.py)

Exploratory data analysis including visualizing semantics using Word2Vec wass performed in word_vecs.py (Output in the wordvec.png)

Modelling in subject_classification.py

## Steps followed:

### Data Scraped from Wikipedia, IIT G official sites.

### Exploratory analysis performed

### Heuristics suggests that freshmen ask questions around 3 broad categories - 1)Hostel facilitiees 2)Academics 3)Miscallaneous (extra-curriculars)

### The data was then put into these three categories

### The sentences were preprocessed and then tokenized and then vectorized documents were generated using TF-IDF.

### Dimensionality reduction resulted in features that were fed into Machine Learning model.

### Thus for a new question asked the model was able to classify the category correctly with round 75% accuracy.

### Once the class of the question is known, the answer is to be found in the data beloning to the data in the concerned cluster. Currently trying different approaches to model the same
