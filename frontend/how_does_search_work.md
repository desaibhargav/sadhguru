**The Search feature, found in `recommender.py`, employs a two-stage process:**

> 1. First, the search query is encoded as a vector just as our database. Next, the `top_k` semantic similarity matches between the encoded query and the encoded corpus of `block` are returned, we'll call them **candidates.**

> 2. Further, these candidates are then **re-ranked** by another NLP model that is trained on the task of ranking how relevant a passage is given a query. Specifically, the model accepts an input of form `(query, passage)` and returns a `cross-score` which represents the relevance of the `passage` in answering the `query`.

After re-ranking, the top few hits, along with their `start_duration`, are returned and displayed in the app.


