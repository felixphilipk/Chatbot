import os
from flask import Flask, request, jsonify
import warnings
import torch.nn.functional as F
from typing import Dict
import torch
import logging
from openfabric_pysdk.utility import SchemaUtil
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText, SimpleTextSchema
from openfabric_pysdk.context import Ray, State
from openfabric_pysdk.loader import ConfigClass
from transformers import AutoTokenizer,AutoModelForQuestionAnswering,AutoModel
from langchain.docstore.document import Document as LangchainDocument
from datasets import load_dataset
import datasets
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset






def mean_pooling(model_output,attention_mask):
     token_embeddings = model_output[0]
     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
     return torch.sum(token_embeddings*input_mask_expanded,1)/torch.clamp(input_mask_expanded.sum(1),min=1e-9)


def generate_embeddings(texts,tokenizer,model,batch_size=32):
    all_embeddings =[]
    for i in tqdm(range(0,len(texts),batch_size)):
        batch_texts = texts[i:i+batch_size]     
        encoded_input = tokenizer(batch_texts,padding=True,truncation=True,return_tensors='pt',max_length=512)
        with torch.no_grad():
            model_output = model(**encoded_input)
        sentence_embeddings = mean_pooling(model_output,encoded_input['attention_mask'])
        all_embeddings.append(sentence_embeddings)
    embeddings = torch.cat(all_embeddings, dim=0)
    embeddings = F.normalize(embeddings,p=2,dim=1)
    return embeddings

############################################################
# Callback function called on update config
############################################################
def config(configuration: Dict[str, ConfigClass], state: State):
   
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model= AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    ds = load_dataset("HuggingFaceH4/orca_dpo_pairs")
    #ds = load_dataset('ag_news', split='train[:1000]')
    print(ds)
    state.knowledge_base =[
        LangchainDocument(page_content=entry["prompt"],metadata={"chosen":entry["chosen"],"rejected":entry["rejected"]})
        for entry in ds["train_prefs"]
    ]
    #state.knowledge_base = [
    #    LangchainDocument(page_content=entry["text"], metadata={"label": entry["label"]})
    #    for entry in ds
    #]
    documents = [doc["prompt"] for doc in ds["train_prefs"]]
    #documents = [doc["text"] for doc in ds]
    state.embeddings= generate_embeddings(documents,tokenizer,model) 
    state.tokenizer= tokenizer
    state.model =model
    
def find_most_similar(query_embedding,document_embedding):
     similarity_scores = cosine_similarity(query_embedding,document_embedding)
     most_similar_idx= similarity_scores.argsort()[0][-1]
     return most_similar_idx


############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: Ray, state: State) -> SimpleText:
    tokenizer = state.tokenizer
    model = state.model
    output = []
    for text in request.text:
        query_embedding = generate_embeddings([text],tokenizer,model)      
        most_similar_idx = find_most_similar(query_embedding,state.embeddings)
        response = state.knowledge_base[most_similar_idx].page_content
        output.append(response)
        return SchemaUtil.create(SimpleText(), dict(text=output))


app = Flask(__name__)
@app.route('/chat',methods =['POST'])
def chat():
    data = request.get_json(force=True)
    schema = SimpleTextSchema()
    chat_request =schema.load(data)
    global state
    chat_response = execute(chat_request,None,state)
    return jsonify({"response":chat_response.text})

if __name__ == '__main__':
        print("Starting Flask app...")
        logging.basicConfig(level=logging.DEBUG)
        state = State()
        config(None,state)
        app.logger.setLevel(logging.DEBUG)
        app.run(debug=True, port=5000, use_reloader=False) 