import os
from dotenv import load_dotenv
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv()

output_parser = CommaSeparatedListOutputParser()

format_instructions = output_parser.get_format_instructions()

prompt_template = PromptTemplate(
    template="AI開発にお勧めのプログラミング言語を教えてください。\n{format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": format_instructions}
)

prompt = prompt_template.format_prompt()

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)

messages = [
    SystemMessage(content="あなたは、AI開発にお勧めのプログラミング言語を教えるAIです。"),
    HumanMessage(content=prompt.text)
]

result = llm.invoke(messages)

result2 = output_parser.parse(result.content)

print(result2)