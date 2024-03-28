# Tell me around...
The [tell_me_around](https://github.com/mirko-pace/chargedrop-tools/blob/main/tell_me_around.ipynb) notebook shows a very simple way to transform the output of the Overpass APIs into the description of the surroundings of a location.
In my use case, I wanted this description of shops and other commercial activities around an EV charger to build some content for the [chargedrop.app](https://chargedrop.app) landing pages.

## What do you need
First of all: I assume you want to use an open-source LLM and run it on your laptop or GPU-loaded box in the cloud. Why? It's so incredibly easy, inexpensive and it allows you to fine-tune your own model 
and generate content with the tone, structure, and language you prefer.
You cannot find an easier way to start right away than to download [Ollama](https://ollama.com) and plug it in directly into your Python script.

Moreover, you'll need to install the [Langchain](https://pypi.org/project/langchain/) library as well as [Overpass](https://pypi.org/project/overpass/) and [Geopy](https://pypi.org/project/geopy/).

## How it works
These are the basic steps, wrapped in two functions:

### Call Overpass
Here, we ask Overpass APIs to tell us all the nodes (entities) around a given Latitude  and Longitude pair within a radius (by default set to 500 meters).

As I'm looking for businesses I'm keeping only the items with a valid `properties.name` field. All items and some additional data is then collated in a simple text object.
Check the [Overpass documentation](https://wiki.openstreetmap.org/wiki/Overpass_API) to customize and extend the info you want to extract. You'll find out there's a lot you can do!

```python
def call_overpass(api,lat,long,radius=500):
    result = api.get(f"node(around:{radius},{lat},{long})")
    res = pd.json_normalize(result,['features'])
    result_string = ''
    if 'properties.name' in res.columns:
        pd_around = res[res['properties.name'].notnull()].fillna("")
        for index, item in pd_around.iterrows():
            result_string += f"Name: {item['properties.name']} Shop: {item['properties.shop']} Amenity: {item['properties.cuisine']} {item['properties.amenity']}"+"\n"
    return result_string
```

### Create your RAG chain
This is a classic Retrieval-Augmented Generation recipe that you can find in the Langchain documentation. In simple words we take the text we just built from the Overpass API, we load the document in Chroma which is a specialized database 
that will handle the [tokenization](https://www.datacamp.com/blog/what-is-tokenization) of your text and the creation of [embeddings](https://towardsdatascience.com/what-is-embedding-and-what-can-you-do-with-it-61ba7c05efd8), as well as provide a retrieval method to your "chatbot" to query the document and retrieve the right context to answer your questions.

Why `chain_type="stuff"`? In this way, we're telling the chain to use all the documents available in the context of the prompt. For this use case that's exactly what I needed but you may want to implement a more refined way to refine your source 
and extract only the relevant portions if you want to implement a proper chatbot that will answer specific questions, or your document is very large.

```python
def create_chain(llm,document):
    # I save the document to a file because I cannot figure out a good way to load it into LangChain
    # as a string/text :( 
    file_path = 'output.txt'
    with open(file_path, 'w') as file:
        file.write(document)
    loader = TextLoader("./output.txt")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(), chain_type="stuff",
        return_source_documents=True)
    return qa_chain
```

### Chat with your document!
At this point you can just start asking questions to your chain and let it describe what's around your Lan&Long point :) 
Switch from LLama2 to Mistral or Openchat models and see the very different styles of the answers.

```python
response = qa_chain(f"""All names contained in the document are businesses around {location}, can you
        please describe all available businesses and shops in the surroundings in plain English? Use a descriptional tone.""")
print(response['result'])
```
