**The reason for such a design choice was because it allows us to kill three birds with one stone:**

> 1. First and foremost, some videos can be very long, which means the transcript for the same is a massive string, and we need to avoid hitting the maximum **input length limits** of NLP models.

> 2. Secondly, it is always good to maintain the inputs at a length on which the models being used were trained, to stay as close as poossible to the training set for **optimum results.**

> 3. But most importantly, the purpose for splitting transcripts to blocks is so that the recommendations can be **targeted to a snippet** within a video. The vision is to recommend many snippets from various videos highly relevant to the query, rather than entire videos themselves in which matching snippets have been found (which may sometimes be long and the content may not always be related to the query).

The obtained blocks are then encoded into vectors using a NLP model that is trained on a contexual semantic similarity task.