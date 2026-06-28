

$ python src/preprocess.py

```
--- BEFORE ---
{
  "id": 1,
  "title": "Blade Runner",
  "year": "1982",
  "genres": "Science Fiction, Neo-Noir,Thriller",
  "summary": "<p>In a dystopian Los Angeles of 2019, a special police officer hunts down bioengineered humans called &#39;replicants&#39; who have escaped to Earth.</p>  As he investigates, he begins to question the boundary between human and machine, and what makes a life worth living."
}

--- AFTER ---
{
  "id": 1,
  "title": "Blade Runner",
  "year": 1982,
  "genres": [
    "Science Fiction",
    "Neo-Noir",
    "Thriller"
  ],
  "primary_genre": "Science Fiction",
  "summary": "In a dystopian Los Angeles of 2019, a special police officer hunts down bioengineered humans called 'replicants' who have escaped to Earth. As he investigates, he begins to question the boundary between human and machine, and what makes a life worth living."
}

Saved 100 records to data\movies_clean.json
```


$ python src/umap_analysis.py
Loaded (16, 1536) from data\embeddings\umap_items_openai_text_small.json, using seed=42
Saved plots\umap_analysis.png

Outliers included for visual inspection:
  - How to fix a flat tire
  - Going fishing at the lake
  - Pasta carbonara recipe
  - Learning Python basics


