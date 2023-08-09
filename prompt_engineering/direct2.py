import pandas as pd
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI
import os
import openai
from dotenv import load_dotenv, find_dotenv


# local file with texts and original generated queries

path = "data/generated_queries/synthesis/sample_synthesis_prompt_synthesis1.jsonl"

# openai key code

_ = load_dotenv(find_dotenv())  # read local .env file
os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = os.environ["OPENAI_API_KEY"]

# query generation prompt

query_prompt_2 = [
        SystemMessage(
        content="""The following is a friendly conversation between a human and an AI 
    assistant. The AI assistant is talkative and provides lots of specific details 
    from its context. If the AI assistant does not know the answer to a question, 
    it truthfully says it does not know. The AI assistant is a world level intelligence 
    analyst expert in analysing events.

    Current conversation:"""
    ),
    HumanMessage(
        content="""Generate a search query for which you would expect the text to 
        appear in a search engine.
        The search query is more likely to be a question than a statement.
        The query should be as short as possible.
        The query should not be overly specific to the text.
        This is the text:
            """
    ),
    HumanMessagePromptTemplate.from_template("{document}"),
]



def compare(i, prompt):
    
    # get text and original query from df
    
    df = pd.read_json(path, lines=True)
    df.index = range(len(df))

    doc = df["text"][i]
    orig_query = df["synthetic_query"][i]

    # initialise query generation llm
    
    llm = ChatOpenAI(temperature=0)

    # build query generation chain
    
    query_prompt = ChatPromptTemplate(messages=prompt, input_variables=["document"])
    query_chain = LLMChain(llm=llm, prompt=query_prompt, verbose=False, 
                           output_key="query")
    
    output = query_chain.run(doc)
    
    # get query string (weird formatting fix)
    query = output.split("Query: ")[1].split("\n")[0]
    
    return query, orig_query, doc

# code to analyse results

prompt = query_prompt_2
prompt_used = "query_prompt_2"

queries, originals, docs = [], [], []
for i in range(5):
    query, orig, doc = compare(i, prompt)
    queries.append(query)
    originals.append(orig)
    docs.append(doc)
    
    
with open('direct_{}.txt'.format(prompt_used), 'w') as f:
    for i in range(len(docs)):
        f.write('New Query: ' + queries[i] + '\n')
        f.write('Original Query: ' + originals[i] + '\n')
        f.write('Text: ' + docs[i] + '\n\n')