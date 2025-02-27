from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np

# Load the trained models and data
popular_df = pickle.load(open(r'C:\Users\abhin\Desktop\book_recomendation_system\trained_models\popular_df.pkl', 'rb'))
pt = pickle.load(open(r'C:\Users\abhin\Desktop\book_recomendation_system\trained_models\pt.pkl', 'rb'))
books = pickle.load(open(r'C:\Users\abhin\Desktop\book_recomendation_system\trained_models\books.pkl', 'rb'))
similarity_scores = pickle.load(open(r'C:\Users\abhin\Desktop\book_recomendation_system\trained_models\similarity_scores.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template(
        'index.html',
        book_name=list(popular_df['Book-Title'].values),
        author=list(popular_df['Book-Author'].values),
        image=list(popular_df['Image-URL-M'].values),
        votes=list(popular_df['num_ratings'].values),
        rating=list(popular_df['avg_rating'].values)
    )

@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html', data=None, error_message=None)

@app.route('/recommend_books', methods=['POST', 'GET'])
def recommend():
    if request.method == 'POST':
        user_input = request.form.get('user_input')
    else:  # If it's a GET request, check for a query parameter (optional)
        user_input = request.args.get('book', '')

    if not user_input:
        return redirect(url_for("recommend_ui"))

    # Check if the book exists in the dataset
    if user_input not in pt.index:
        return render_template('recommend.html', data=None, error_message="Book not found in our database. Please try another book.")

    # Find similar books
    index = np.where(pt.index == user_input)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:5]

    data = []
    for i in similar_items:
        temp_df = books[books['Book-Title'] == pt.index[i[0]]].drop_duplicates('Book-Title')
        if not temp_df.empty:
            data.append([
                temp_df['Book-Title'].values[0],
                temp_df['Book-Author'].values[0],
                temp_df['Image-URL-M'].values[0]
            ])

    return render_template('recommend.html', data=data, error_message=None)

if __name__ == '__main__':
    app.run(debug=True, port=5500)
