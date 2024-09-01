import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Download NLTK stopwords
nltk.download('stopwords')
nltk.download('punkt_tab')

# Load the dataset
# Example CSV structure: 'Description', 'Demand'
df = pd.read_csv('property_data.csv')

# Preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing
df['Processed_Description'] = df['Description'].apply(preprocess_text)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Processed_Description'])

# Target variable
y = df['Demand']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the demand on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Example prediction
def predict_demand(description):
    processed_description = preprocess_text(description)
    description_vector = vectorizer.transform([processed_description])
    predicted_demand = model.predict(description_vector)
    return predicted_demand[0]

# Test the prediction function
sample_description = "Spacious 3-bedroom apartment with a beautiful view, located in a prime area."
predicted_demand = predict_demand(sample_description)
print(f'Predicted Demand: {predicted_demand}')
