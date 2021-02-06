**The NLP models used are described below:**

> 1. **Distil roBERTa** `paraphrase-distilroberta-base-v1`: This model is trained on the semantic textual similarity task. Specifically, between a given a text (query in our case) and any other text (a text from the database in our case), it is trained to produce vector encodings of the texts in such a way that any measure of vector similarity (cosine similarity in our case) between the two vecotrs, returns a high score. 

> 2. **Electra** `ms-marco-electra-base`: In contrast to the preceding model, this model does not return a vector encdoing. Rather, this model takes `(query, passage)` as input, and produces an output value between 0 and 1 indicating how relevant the `passage` is in answering the `query`. 

Both the models are obtained from the `SentenceTransformer` library. 

