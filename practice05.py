import os 
from dotenv import load_dotenv
from langchain.prompts import FewShotPromptTemplate, PromptTemplate, SemanticSimilarityExampleSelector
from langchain_openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import SystemMessage, HumanMessage

load_dotenv()

examples = [

    {"input": "猫", "output": "哺乳類"},

    {"input": "ITエンジニアついて教えてください。", "output": "ITエンジニアは、システム開発や保守を行う技術者です。"},

    {"input": "犬", "output": "哺乳類"},

    {"input": "パリでおすすめの観光地は？", "output": "エッフェル塔、ルーブル美術館、ノートルダム大聖堂は、パリでの人気観光地です。"},

    {"input": "亀", "output": "爬虫類"},

]

prompt_template = """
入力: {input}
出力: {output}
"""

prompt = PromptTemplate(
    input_variables=["input", "output"],
    template=prompt_template
)

example_selector = SemanticSimilarityExampleSelector.from_examples(

    examples=examples,

    embeddings=OpenAIEmbeddings(),

    vectorstore_cls=FAISS,

    k=2

)

few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=prompt,
    prefix="以下の質問に対して、適切な回答を提供してください。",
    suffix="入力: {input}\n出力: ",
    input_variables=["input"],
    example_separator="\n\n"
)


messages = few_shot_prompt.format(input="イルカ")

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)

messages_2 = [
    SystemMessage(content="あなたは、質問に対して適切な回答を提供するAIです。"),
    HumanMessage(content=messages)
]

result = llm.invoke(messages_2)

print(result.content)