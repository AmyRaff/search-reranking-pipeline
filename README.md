This repo contains code for a reranking-search model project. The project pipeline is discussed further in the **flyte** directory, and contains the following steps:
- Dataset loading (list of documents)
- Prompt engineering (generate synthetic queries for each document)
- Consistency checks (filter out poor queries)
- Negative sampling (handle dataset bias)
- Model finetuning

The overall code pipeline produces a fine-tuned model which works to re-rank search results when a query is entered into the search bar. There are two implementations - normal Python ( in **scripts** ) and Flyte ( in **Flyte** ). The Flyte implementation works to create a remote pipeline to more conveniently and efficiently fine-tune models.

Prompt engineering experiments used to decide which prompts should be included in the pipeline are detailed in the **prompt_engineering** directory.
