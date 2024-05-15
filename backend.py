from flask import Flask, jsonify, request
from flask_cors import CORS,  cross_origin
import pickle
import numpy as np

# Load pickled files
popular_df = pickle.load(open('popular.pkl', 'rb'))
pt = pickle.load(open('pt.pkl', 'rb'))
books = pickle.load(open('books.pkl', 'rb'))
similarity_scores = pickle.load(open('similarity_scores.pkl', 'rb'))

# Initialize Flask application
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}},methods=['GET', 'POST'])

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

#app.use(cors());

@app.route('/', methods=['POST'])
@cross_origin()
def index():
    # Create a dictionary with the required data
    data = {
        'book_name': [str(title) for title in popular_df['Book-Title'].values],
        'author': [str(author) for author in popular_df['Book-Author'].values],
        'image': [str(image_url) for image_url in popular_df['Image-URL-M'].values],
        'votes': [int(num_ratings) for num_ratings in popular_df['num_ratings'].values],
        'rating': [float(avg_rating) for avg_rating in popular_df['avg_rating'].values]
    }
    # Send the data as a JSON response
    return jsonify(data)

@app.route('/recommend_books', methods=['POST'])
@cross_origin()
def recommend():
    user_input = request.json.get('user_input')
    index = np.where(pt.index == user_input)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:5]

    book_names = []
    authors = []
    images = []
   # Iterate through similar items
    for i in similar_items:
        # Filter the books DataFrame for the recommended book
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]

        # Append book information to the respective lists
        book_names.append(str(temp_df.drop_duplicates('Book-Title')['Book-Title'].values[0]))
        authors.append(str(temp_df.drop_duplicates('Book-Title')['Book-Author'].values[0]))
        images.append(str(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values[0]))

    # Create a dictionary to hold the transformed data
    data = {
        'book_name': book_names,
        'author': authors,
        'image': images
    }
    # Send the data as a JSON response
    return jsonify(data)

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug='true', port=8081)

