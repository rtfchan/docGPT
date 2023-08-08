#!/usr/bin/env python3
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import AzureOpenAI
from langchain.prompts import PromptTemplate
from constants import CHROMA_SETTINGS
import os


from flask import Flask, render_template, request

app = Flask(__name__)

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
embeddingDeploymentName = os.environ.get('EMBEDDING_DEPLOYMENT_NAME')
llmModelDeploymentName = os.environ.get('LLM_MODEL_DEPLOYMENT_NAME')
llmModelType = os.environ.get('LLM_MODEL_TYPE')
llmTemperature = float(os.environ.get('LLM_TEMPERATURE'))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))


embeddings = OpenAIEmbeddings(deployment=embeddingDeploymentName,chunk_size=1)
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})

llm = AzureOpenAI(deployment_name=llmModelDeploymentName, model_name=llmModelType, temperature=llmTemperature)
    # Custom Template
system_template = \
"You are a single-answer intelligent assistant helping employees by answering one question and providing three recommended followup questions using sources from internal documents. " \
"Answer the question using only the data provided in the information sources below. Try to answer with as much details as possible. " \
"Provide an answer in the same language it was asked. " \
"If you cannot answer using the sources below, just say 'Sorry I don't know.' " \
"After your answer, in a new line, say 'Recommended Questions:', and provide a list of up to three recommended questions (questions only) for the user that are only relavant to the sources. Do not provide answers to your recommended questions. Recommended questions should be in the same langauge as the user's question. " \
"After the recommended questions, end the conversation with '<|im_end|>'. There should be nothing after the recommended questions. No more questions and answers after 'Recommended Questions'. There is no need to say 'End of conversation' at the end. There is no need to say 'End of conversation' at the end.\n" \
"\nSources:\n{context}" \
"\n\nThe only question: {question}" \
"\n\nThe only answer:"

qa_prompt = PromptTemplate.from_template(system_template)

qa = RetrievalQA.from_llm(llm=llm, retriever=retriever, verbose=True, prompt=qa_prompt ,return_source_documents=1)

@app.route("/")
def root():
    return render_template("index.html")

@app.route('/', methods=['POST'])
def askQuestion():
    question = request.form['question_text']
    res = qa(question)
    answer, docs = res['result'], res['source_documents']    
    rawSources = ''
    for document in docs:
        rawSources += '\n' + '> ' + document.metadata["source"] + ':' + '\n' + document.page_content + '\n' + '----------------------------------------------------'
    sources = rawSources.split('\n')

    return render_template("index.html", question=question, answer=answer, sources=sources)
if __name__ == "__main__":
    app.run()