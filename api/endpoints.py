import os
from dotenv import load_dotenv

from fastapi import APIRouter
from schemas.models import Question, Answer

from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

router = APIRouter()

@router.post("/ask", response_model=Answer)
async def ask_question(question: Question):
  loader = CSVLoader(file_path="docs/dataset.csv")
  docs = loader.load()

  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  splits = text_splitter.split_documents(docs)
  vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

  retriever = vectorstore.as_retriever()
  prompt = hub.pull("rlm/rag-prompt")

  def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

  rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
  )

  answer = rag_chain.invoke(question.text)

  return Answer(text=answer)