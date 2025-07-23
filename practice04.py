import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import LengthBasedExampleSelector, FewShotPromptTemplate, PromptTemplate
from langchain.schema import SystemMessage, HumanMessage

load_dotenv()

examples = [
    {"prefecture": "東京", "region": "関東"},
    {"prefecture": "大阪", "region": "近畿"},
    {"prefecture": "北海道", "region": "北海道・東北"},
    {"prefecture": "福岡", "region": "九州・沖縄"},
    {"prefecture": "愛知", "region": "中部"},
    {"prefecture": "広島", "region": "中国・四国"},
    {"prefecture": "京都", "region": "近畿"},
    {"prefecture": "神奈川", "region": "関東"},
    {"prefecture": "埼玉", "region": "関東"},
    {"prefecture": "千葉", "region": "関東"},
    {"prefecture": "茨城", "region": "関東"},
    {"prefecture": "静岡", "region": "中部"},
    {"prefecture": "宮城", "region": "北海道・東北"},
    {"prefecture": "新潟", "region": "中部"},
    {"prefecture": "長野", "region": "中部"},
    {"prefecture": "岐阜", "region": "中部"},
    {"prefecture": "栃木", "region": "関東"},
    {"prefecture": "群馬", "region": "関東"},
    {"prefecture": "山梨", "region": "中部"},
    {"prefecture": "福島", "region": "北海道・東北"},
    {"prefecture": "岡山", "region": "中国・四国"},
    {"prefecture": "熊本", "region": "九州・沖縄"},
]

prompt_template = """
都道府県: {prefecture}
地域: {region}
"""

prompt = PromptTemplate(
    input_variables=["prefecture", "region"],
    template=prompt_template
)

example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=prompt,
    max_length=20,
)

few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=prompt,
    prefix="以下の都道府県とその地域を答えてください。",
    suffix="都道府県: {prefecture}\n地域: ",
    input_variables=["prefecture"],
    example_separator="\n\n"
)

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)

message = few_shot_prompt.format(prefecture="愛知県")

messages = [
    SystemMessage(content="あなたは日本の都道府県とその地域を答えるAIです。"),
    HumanMessage(content=message)
]

result = llm.invoke(messages)

print(result.content)

