import pandas as pd
from langchain.chains import SequentialChain, LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI
import os
import openai
from dotenv import load_dotenv, find_dotenv


# region event-prompts

prompt_msgs_1 = [
    SystemMessage(
        content="""The following is a friendly conversation between a human and an AI 
    assistant. The AI assistant is talkative and provides lots of specific details 
    from its context. If the AI assistant does not know the answer to a question, 
    it truthfully says it does not know. The AI assistant is a world level intelligence 
    analyst expert in analysing events.

    Current conversation:"""
    ),
    HumanMessage(
        content="""Extract a list of events from the text. An event is "a thing that 
        happens or takes place, especially one of importance."
        You must only report events with a notable impact on the world.
        Do not simply extract a sentence from the text, extrapolate the event from the 
        text and include all the relevant context.
        You must not repeat the same event multiple times, list it once including all 
        the available context.
        This is the text:
            """
    ),
    HumanMessagePromptTemplate.from_template("{document}"),
    HumanMessage(
        content="Tips: Make sure you use the correct format. Think step by step"
    ),
]

prompt_msgs_2 = [
    SystemMessage(
        content="""The following is a friendly conversation between a human and an AI 
    assistant. The AI assistant is talkative and provides lots of specific details 
    from its context. If the AI assistant does not know the answer to a question, 
    it truthfully says it does not know. The AI assistant is a world level intelligence 
    analyst expert in analysing events.

    Current conversation:"""
    ),
    HumanMessage(
        content="""Extract a list of events from the text.
        An event is "a thing that happens or takes place, especially one of importance."
        You must only report events with a notable impact on the world.
        Do not simply extract a sentence from the text, extrapolate the event from the 
        text and include all the relevant context.
        You must not repeat the same event multiple times, list it once including all 
        the available context.
        This is the text:
        """
    ),
    HumanMessagePromptTemplate.from_template("{document}"),
    HumanMessage(
        content="Tips: Make sure you use the correct format. Think step by step"
    ),
    AIMessagePromptTemplate.from_template(template="{events}", name="AI Assistant"),
    HumanMessage(
        content="""Use the following keys to extract information from the text about 
        the events in the list.
        You must distinguish information on the event from information on the reporting 
        of the event.
        The 'event' key reports the short description of the event as previously 
        extracted.
        The 'context' key reports all the available context of the event as previously 
        extracted.
        The 'reporter' is who is reporting the event.
        'who' describes who is the protagonist of the event.
        The value of the 'what' property must be described in under 10 words, not 
        exactly reporting the original text.
        'where' describes the location of the event, 'when' the time or date.
        'why' describes the reason or motivation behind the event.
        'how' describes how it happened.
        'impact' should be a value among the following: political impact, economic 
        impact, environmental impact, social impact, impact on legislation, warfare 
        impact, casualties.
        'modality' should be a value among the following: confirmed, predicted, 
        suggested.
        'event_type' should be the type of event identified in one or two words.
        The value must be 'unknown' if you cannot tell from the context, do not try to 
        guess information not present in the text, and you must not leave empty values.
        """
    ),
    HumanMessage(
        content="Tips: Make sure to answer in the correct format: a json dictionary \
        with an events key mapping to a list of jsons, one json for each event."
    ),
]

# endregion

# local file with texts and original generated queries
path = "data/generated_queries/synthesis/sample_synthesis_prompt_synthesis1.jsonl"

# openai key code
_ = load_dotenv(find_dotenv())  # read local .env file
os.environ["OPENAI_API_KEY"] = "key"
openai.api_key = os.environ["OPENAI_API_KEY"]

def compare(i, prompt):
    
    # get text and original query from df
    
    df = pd.read_json(path, lines=True)
    df.index = range(len(df))

    doc = df["text"][i]
    orig_query = df["synthetic_query"][i]

    # initialise llm
    
    llm = ChatOpenAI(temperature=0)

    # build event chain
    
    prompt1 = ChatPromptTemplate(messages=prompt_msgs_1, input_variables=["document"])
    chain1 = LLMChain(llm=llm, prompt=prompt1, verbose=False, output_key="events")
    prompt2 = ChatPromptTemplate(
        messages=prompt_msgs_2,
        input_variables=["document", "events"],
    )
    chain2 = LLMChain(
        llm=llm,
        prompt=prompt2,
        verbose=False,
        output_key="events_list",
    )
    chain = SequentialChain(
        chains=[chain1, chain2],
        verbose=False,
        input_variables=["document"],
    )

    # outputs list of events 
    
    output = chain.run(doc)
    
    # get event string to input into next chain
    
    event = output.split('"event": ')[1].split(",")[0]
    
    # initialise llm for query generation
    
    llm = ChatOpenAI(temperature=0)
    
    # build query generation chain

    query_prompt = ChatPromptTemplate(messages=prompt, input_variables=["document"])
    query_chain = LLMChain(llm=llm, prompt=query_prompt, verbose=False, 
                           output_key="query")
    
    output = query_chain.run(event)
    
    # get query string (weird formatting fix)
    query = output.split("Query: ")[1].split("\n")[0]
    
    return query, orig_query, doc


# query generation prompt

query_prompt_1 = [
    SystemMessage(
        content="""The following is a friendly conversation between a human and an AI 
    assistant. The AI assistant is talkative and provides lots of specific details 
    from its context. If the AI assistant does not know the answer to a question, 
    it truthfully says it does not know. The AI assistant is a world level intelligence 
    analyst expert in analysing events.

    Current conversation:"""
    ),
    HumanMessage(
        content="""Generate a query which is satisfied by the text.
        The query should be as short as possible.
        The query should not be overly specific to the text.
        The query is equally likely to be a question or a sentence.
        This is the text:
            """
    ),
    HumanMessagePromptTemplate.from_template("{document}"),
]

# code to analyse results

prompt = query_prompt_1
prompt_used = "query_prompt_1"

queries, originals, docs = [], [], []
for i in range(5):
    query, orig, doc = compare(i, query_prompt_1)
    queries.append(query)
    originals.append(orig)
    docs.append(doc)
    
with open('with_events_{}.txt'.format(prompt_used), 'w') as f:
    for i in range(len(docs)):
        f.write('New Query: ' + queries[i] + '\n')
        f.write('Original Query: ' + originals[i] + '\n')
        f.write('Text: ' + docs[i] + '\n\n')