import os 
from dotenv import load_dotenv
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv()

examples = [
    {"prefecture": "Tokyo", "population": "13929286", "area": "2194 km²"},
    {"prefecture": "Kanagawa", "population": "9200166", "area": "2415 km²"},
    {"prefecture": "Osaka", "population": "8839469", "area": "1905 km²"},
]

examples_template = """"
prefecture: {prefecture}, population: {population}, area: {area}
"""

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=PromptTemplate(
        input_variables=["prefecture", "population", "area"],
        template=examples_template
    ),
    prefix="以下の情報を基に、都道府県の人口と面積を答えてください。",
    suffix="prefecture: {prefecture}\npopulation: \narea:",
    input_variables=["prefecture"],
    example_separator="\n\n"
)

message = prompt.format(prefecture="岡山県")

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)

messages = [

    SystemMessage(content="You are a helpful assistant."),

    HumanMessage(content=message),

]

result = llm.invoke(message)

print(result.content)