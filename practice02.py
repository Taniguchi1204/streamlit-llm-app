import os 
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)

system_template = "あなたは、{genre}の詳しいAIです。ユーザーからの質問に対して100文字以内で答えてください。"
human_template = "{question}"

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ]
)

messages = prompt.format_messages(genre="フィットネス", question="筋トレの効果は何ですか？")

result = llm.invoke(messages)

print(result.content)

