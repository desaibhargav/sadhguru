**The Explore feature, found in `recommender.py`, is also a two-stage process:**

> 1. First, just as with the Search feature, the query from the user is encoded. However, here, the encoded query is then matched with **both**, the encoded corpus of `video_title` and the encoded corpus of `video_descripton`, resulting in `top_k` hits from both these corpuses. 

> 2. Searching for hits in both the encoded corpuses of `video_title` and `video_description` results in a **robust** method which avoids false positives, as well as missing highly relevant videos. The videos suggested by hits from both these corpuses are merged and the top few hits are returned as recommendations. 


