from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('modelinterface.html')

@app.route('/cluster', methods=['POST'])
def cluster():
    # Read uploaded file and selected columns
    file = request.files['data_file']
    columns = request.form.getlist('columns')
    n_clusters = int(request.form['n_clusters'])

    # Read data into a Pandas DataFrame
    data = pd.read_csv(io.StringIO(file.stream.read().decode("UTF8")), usecols=columns)

    # Preprocess the data
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)

    # Convert gender column to numeric values
    data["Genre"] = data["Genre"].map({"Female": 0, "Male": 1})

    # Scale the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Perform hierarchical clustering
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    y = model.fit_predict(data_scaled)

    # Visualize the clusters
    fig, ax = plt.subplots()
    sns.scatterplot(x=data.iloc[:, 0], y=data.iloc[:, 1], hue=y, palette='deep', ax=ax)
    ax.set_title(f"Hierarchical Clustering with {n_clusters} Clusters")

    # Encode the visualization as base64 for display on the webpage
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8').replace('\n', '')
    buffer.close()

    return render_template('cluster.html', image_base64=image_base64)

if __name__ == '__main__':
    app.run(debug=True)
