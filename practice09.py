import os
from dotenv import load_dotenv
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

class TrabelPlan(BaseModel):
    destination: list[str] = Field(description="旅行先の都市名のリスト"),
    activities: list[str] = Field(description="旅行先でのアクティビティのリスト"),
    duration: int = Field(description="旅行の期間（日数）"),
    budget: float = Field(description="旅行の予算（円）")

output_parser = PydanticOutputParser(pydantic_object=TrabelPlan)

format_instruction = output_parser.get_format_instructions()

template = """
次のフォーマットで旅行プランを作成してください。
{format_instruction}
旅行のテーマ: {theme}
"""

prompt_template = PromptTemplate(
    template=template,
    input_variables=["theme"],
    partial_variables={"format_instruction": format_instruction}
)

prompt = prompt_template.format_prompt(theme="家族旅行")

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)

messages = [
    SystemMessage(content="あなたは、旅行プランを作成するAIです。"),
    HumanMessage(content=prompt.text)
]

result = llm.invoke(messages)

result2 = output_parser.parse(result.content)

print(result2)