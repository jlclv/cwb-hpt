from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
    )

from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFium2Loader
import os
import tiktoken
from openai import OpenAI
import requests, uuid, json

def get_search_index(name: str):
    fields = [
        SimpleField(
            name="id", 
            type=SearchFieldDataType.String, 
            key=True
        ),
        SearchField(
            name="metadata",
            type=SearchFieldDataType.String,
            sortable=False,
            filterable=False,
            facetable=False,
            searchable=True,
        ),
        SearchField(
            name="isInternal",
            type=SearchFieldDataType.Boolean,
            sortable=False,
            filterable=True,
            facetable=True,
        ),
        SearchField(
            name="en_content",
            type=SearchFieldDataType.String,
            sortable=False,
            filterable=False,
            facetable=False,
            searchable=True,
            analyzer_name="en.microsoft",
        ),
        SearchField(
            name="vi_content",
            type=SearchFieldDataType.String,
            sortable=False,
            filterable=False,
            facetable=False,
            searchable=True,
            analyzer_name="vi.microsoft",
        ),
        SearchField(
            name="en_content_vector", 
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="hpt-vector-config",
        ),
        SearchField(
            name="vi_content_vector", 
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="hpt-vector-config",
        ),
    ]

    vector_search = VectorSearch(
        profiles=[
            VectorSearchProfile(
                name="hpt-vector-config", 
                algorithm_configuration_name="hpt-algorithms-config"
                )],
        algorithms=[
            HnswAlgorithmConfiguration(
                name="hpt-algorithms-config")]
    )

    semantic_search = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name="hpt-semantic-config",
                prioritized_fields=SemanticPrioritizedFields(
                    content_fields=[
                        SemanticField(field_name="en_content"),
                        SemanticField(field_name="vi_content")]))
        ]
    )

    return SearchIndex(name=name, fields=fields, vector_search=vector_search, semantic_search=semantic_search)


def recursive_chunking(documents, chunk_size=600, chunk_overlap=125):
    recursive_text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name=tiktoken.encoding_for_model("gpt-3.5-turbo").name,
    chunk_size=chunk_size, 
    chunk_overlap=chunk_overlap
    )

    recursive_text_splitter_chunks = recursive_text_splitter.split_documents(documents)
    return recursive_text_splitter_chunks


def get_embedding(text, key, model="text-embedding-ada-002"):
   client = OpenAI(api_key=key)
   
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def get_file_stats(filename):    
    loader = PyPDFium2Loader(filename)
    pages = loader.load_and_split()
    text = ""

    for page in pages:
        text += page.page_content

    result_pages = len(pages)
    result_words = len(text.split())
    
    return f"Number of pages: {result_pages}\nNumber of words: {result_words}"

def openai_chatcompletion(prompt, gptmodel, key):
    # context loosely based on https://github.com/Azure-Samples/ai-chat-protocol/blob/main/README.md 
    context = """
        You are an assistant that helps company employees with their ISO27K questions, and questions about Internal regulations of ISO 27001. Be detailed and complete with your answers.
        Answer ONLY with the information above. 
        If there isn't enough information below, say you don't know. 
        Do not make up your own answers. 
        If asking a clarifying question to the user would help, ask the question.
        If the question is not in English, answer in the language used in the question.
        Each source contains a metadata that has the name followed by colon and the actual information, 
        always include the metadata document for each fact you use in the response. 
        Use square brackets to reference the metadata, for example [info1.pdf Page:0]. 
        Don't combine metadata, list each metadata separately, for example [info1.pdf Page:1][info2.pdf Page:2].
    """

    client = OpenAI(api_key=key)

    completion = client.chat.completions.create(
    model= gptmodel,
    messages=[
        {"role": "system", "content": context},
        {"role": "user", "content": prompt}
    ]
    )

    return completion.choices[0].message  

def ask_data(query, language, service_endpoint, index_name, search_key, openai_key, gptmodel="gpt-3.5-turbo-0125", topn=3, min_score=0.5):
    search_client = SearchClient(service_endpoint, index_name, AzureKeyCredential(search_key))
    vector_query = VectorizedQuery(vector=get_embedding(query, openai_key), k_nearest_neighbors=3, fields=f"{language}_content_vector")
    results = search_client.search(
            query,
            vector_queries=[vector_query],
            top=topn,
            query_type="semantic",
            semantic_configuration_name="hpt-semantic-config",
            select=["metadata", f"{language}_content"],
        )

    fulltext_list = []
    
    for result in results:
        reference = result[f"{language}_content"] + " "
        reference += result["metadata"]
        fulltext_list.append(reference)

    fulltext = "".join(fulltext_list)
    answer_reference = openai_chatcompletion(fulltext, gptmodel, openai_key)

    return (answer_reference)

def translate_chunk(text, subscription_key, region, endpoint, langFrom="en", langTo="vi"):
    path = '/translate?api-version=3.0'
    params = f'&from={langFrom}&to={langTo}'
    constructed_url = endpoint + path + params

    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
        'Ocp-Apim-Subscription-Region': region,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }

    body = [{
        'text' : text
    }]
    request = requests.post(constructed_url, headers=headers, json=body)
    response = request.json()

    return(json.dumps(response[0]["translations"][0]["text"], sort_keys=True, indent=4, ensure_ascii=False, separators=(',', ': ')))