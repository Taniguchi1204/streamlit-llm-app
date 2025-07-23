from dotenv import load_dotenv
import os
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# .envからAPIキーを読み込む
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

template = '''
以下の専門用語を、初心者向けに説明してください。
専門用語: {term}
説明：
'''

prompt = PromptTemplate(
    input_varibles=["term"],
    template=template
)

message = prompt.format(term="AI")
print(message)

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

messages = [
    SystemMessage(content="You are a helpful assistant that explains technical terms in a beginner-friendly way."),
    HumanMessage(content=message)
]

result = llm.invoke(messages)

print(result.content)

