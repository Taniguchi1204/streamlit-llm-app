import os 
from dotenv import load_dotenv
from langchain.output_parsers import EnumOutputParser
from enum import Enum
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv()

class Prefecture(Enum):
    Prefecture1 = "東京"
    Prefecture2 = "大阪"
    Prefecture3 = "北海道"


output_parser = EnumOutputParser(enum=Prefecture)

format_instructions = output_parser.get_format_instructions()

prompt_template = PromptTemplate(
    template="以下の都道府県から1つ選んでください。ただし回答は都道府県名のみとしてください。\n{format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": format_instructions}
)

prompt = prompt_template.format_prompt()

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)

messages = [
    SystemMessage(content="あなたは、都道府県を選ぶAIです。"),
    HumanMessage(content=prompt.text)
]

result = llm.invoke(messages)

result2 = output_parser.parse(result.content)

print(result2.value)