# NYUAD Academic Landscape

**Author:** Aashma Varma  
**Course:** Human-Centered Data Science, taught by Professor Minsu Park @ NYU Abu Dhabi

An interactive visualization of the NYUAD faculty research landscape, mapping professors by the similarity of their research interests using machine learning.

---

## What this project does

This project takes NYUAD faculty data scraped from the university website and produces an interactive 2D map where each dot represents a faculty member. Professors who share similar research interests appear closer together, and colors indicate which research cluster they belong to.

The goal is to reveal natural "academic communities" that cut across official faculty divisions. For example, an engineer and a social scientist who both work on urban data might end up in the same cluster despite belonging to different departments.

---

## How it works

The pipeline has four main stages:

1. **Data collection** — Faculty names, job titles, research interests, and divisions were scraped from the NYUAD website and stored in a CSV file.

2. **Sentence embeddings** — Each professor's research text is converted into a 384-dimensional vector using the `all-MiniLM-L6-v2` model from the `sentence-transformers` library. Professors with similar research interests produce vectors that are mathematically close to each other.

3. **K-Means clustering** — The vectors are grouped into k clusters (k=8) using K-Means. All professors in the same cluster share broadly similar research themes.

4. **UMAP dimensionality reduction** — The 384-dimensional vectors are compressed down to 2D using UMAP, making it possible to plot everyone on a scatter plot while approximately preserving their closeness relationships.

The final output is an interactive Plotly chart saved as `index.html`. Hover over any dot to see the professor's name, job title, faculty division, and research interests.

---

## Files

| File | Description |
|------|-------------|
| `nyuad_landscape.py` | Main Python script — runs the full pipeline |
| `nyuad-landscape.ipynb` | Jupyter Notebook — walkthrough with explanation |
| `faculty_data.csv` | Scraped faculty data (name, job title, email, research text, division, profile URL) |
| `index.html` | The interactive visualization — open in any browser |

---

## How to run it yourself

**1. Install dependencies**
```
pip install pandas sentence-transformers scikit-learn umap-learn plotly
```

**2. Run the script**
```
python nyuad_landscape.py
```

This will generate a fresh `index.html` in the same folder. The embedding model (~90 MB) is downloaded automatically on the first run and cached locally after that.

---

## Libraries used

- [pandas](https://pandas.pydata.org/) — data loading and cleaning
- [sentence-transformers](https://www.sbert.net/) — converting research text into numerical vectors
- [scikit-learn](https://scikit-learn.org/) — K-Means clustering
- [umap-learn](https://umap-learn.readthedocs.io/) — dimensionality reduction from 384D to 2D
- [plotly](https://plotly.com/python/) — interactive visualization
