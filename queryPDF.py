from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import os

#使用するファイル
file_path = "PDF_FILE"
#使用するapi、環境変数から取得
openai_api_key = os.getenv("OPENAI_API_KEY")
#分割するチャンクサイズ
chunk_size = 100

#Document object作成
loader = PyPDFLoader(file_path=file_path).load()

#適切なチャンクサイズに分割
text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
texts = text_splitter.split_documents(loader)

#embeddingsとFAISSオブジェクトを作成、ベクトル化して保存
embeddings = OpenAIEmbeddings(api_key=openai_api_key)
db = FAISS.from_documents(texts, embeddings)

#retriever作成
retriever = db.as_retriever(search_kwargs=dict(k=5))

def rag_answer(retriever,model,question):
    #questionをもとにretrieverで関連性のあるものを5件抽出
    retrieved_docs = retriever.invoke(question)
    #ChatPromptTemplateで文字を参照して、質問の回答を返すテンプレートを作成
    system_message_template =  "Based on the following documents:\n" + "\n".join([doc.page_content for doc in retrieved_docs])
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_message_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template("{text}")
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    #モデルに質問して、返答をreturn
    response = model.invoke(chat_prompt.format_prompt(text=question).to_messages())
    return response.content  

#モデルを作成
model = ChatOpenAI(model="gpt-3.5-turbo",temperature=0, openai_api_key=openai_api_key)

#質問する
print("QUESTION:")
question = input()
answer = rag_answer(retriever,model,question)
print(answer)