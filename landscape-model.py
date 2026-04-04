# -----------------------
# NYUAD Academic Landscape
# Aashma Varma
# -----------------------

# -----------------------
# IMPORT LIBRARIES
# -----------------------

import pandas as pd # dataframes
import textwrap # text wrapping for hover boxes
from sentence_transformers import SentenceTransformer # text -> vector 
from sklearn.cluster import KMeans # ml algo to group similar items
from sklearn.preprocessing import normalize # comparing vectors
import umap # high-dimensional data -> 2D
import plotly.express as px # interactive plotting
import numpy as np

# -----------------------
# LOAD DATA
# -----------------------

file = pd.read_csv("faculty_data.csv")

# Checking if loaded properly
print("Columns found:", file.columns.tolist())
print(f"Total faculty: {len(file)}")

# Cleaning up division label (forgot to do in scraper D:)
file["Faculty"] = (
    file["Faculty"].str.replace("-", " ").str.title()
)

# Checking if done correctly, and count of each division
print("\nDivisions found:")
print(file["Faculty"].value_counts())

# Cleaning up research text
file["Research Text"] = (
    file["Full Research Paragraph"].fillna("").str.strip()
)

# I already made sure while scraping every faculty without research area has "Not listed"
# however just in case...
# -----------------------
# QUESTION: should I remove those faculty with "not listed"? they're ruining the plot of the graph a little
# -----------------------

file = file[file["Research Text"] != ""].reset_index(drop=True)
print(f"\n{len(file)} faculty members with research text.")

# Wrap text so the hover boxes have a consistent width
file["Research Text"] = file["Research Text"].apply(
    lambda t: "<br>".join(textwrap.wrap(t, width=60))
)

# same for job title
file["Job Title"] = file["Job Title"].apply(
    lambda t: "<br>".join(textwrap.wrap(t, width=60))
)

# -----------------------
# SENTENCE EMBEDDING
# -----------------------

print("\nLoading embedding model...")
# Download the pre-trained model we want
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")
# each research paragraph converted to a vector
embeddings = model.encode(
    file["Full Research Paragraph"].tolist(), # list of all the texts in research column
    show_progress_bar = True, # for terminal
    batch_size = 32 # process 32 texts at a time
)

# normalize vectors so they are comparable in scale
# watdatmean?? = rn embeddings storing vectors with diff magnitude [0.2, 0.5, ... -0.1]
# but we only care abt the direction of vector not size, so normalize all magnitudes to 1
# COSINE SIMILARITY!
embeddings = normalize(embeddings)
print(f"Embeddings shape: {embeddings.shape}")

# -----------------------
# K-MEANS CLUSTERING
# QUESTION: Should i use elbow method??
# -----------------------

K = 8 # number of clusters we want
print(f"\nRunning K-Means with k={K}...")

# initialize KMeans clustering
kmean_v = KMeans(n_clusters=K, random_state=42, n_init=10)
# assign each faculty to a cluster based on similarity
file["Cluster"] = kmean_v.fit_predict(embeddings).astype(str)

# Printing to check
print("Cluster sizes:")
print(file["Cluster"].value_counts().sort_index())

# -----------------------
# UMAP - COMPRESSING TO 2D
# -----------------------

print("\nRunning UMAP...")
# Reduce 384 dimensional vectors -> 2d coordinates
reducer = umap.UMAP(
    n_neighbors = 8,
    min_dist = 0.6,
    random_state = 42
)

# -----------------------
# SCATTER PLOT
# -----------------------

# convert embeddings to 2d
coords_2d = reducer.fit_transform(embeddings) 
# store x and y coords into the dataframe
file["x"] = coords_2d[:, 0]
file["y"] = coords_2d[:, 1]

print("\nBuilding interactive plot...")

# create scatter plot
figure = px.scatter(
    file,
    x="x",
    y="y",
    color="Cluster", # colour each point by cluster group
    hover_name="Name", # name shown in bold while hovering
    # additional info while hovering:
    hover_data={
        "Job Title": True,
        "Faculty": True,
        "Research Text": True,
        "x": False,
        "y": False,
        "Cluster": False
    },
    title="NYUAD Academic Landscape",
    labels={"Cluster": "Research Cluster", "Faculty": "Faculty Division"},
    width=1100,
    height=750,
    template="plotly_white"
)

# style of markers (data points)
figure.update_traces(
    marker = dict(size=7, line=dict(width=0.8, color="white")),
    selector=dict(mode="markers")
)

# cleaning up the figure
figure.update_layout(
    showlegend=False, # remove legend of clusters
    font_family="Arial",
    title_font_size=18,
    xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=""),
    yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=""),
    plot_bgcolor="#fafafa",
)

# save the interactive plot as HTML file
figure.write_html("nyuad_landscape.html")

print("\nDone! Open nyuad_landscape.html in your browser.")
 
figure.show()