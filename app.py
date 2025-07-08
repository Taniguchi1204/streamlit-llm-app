import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# .envからAPIキーを読み込む
load_dotenv()

# 専門家の種類とシステムメッセージ
EXPERTS = {
    "ヘルスケアの専門家": "あなたはプロフェッショナルなヘルスケアコンサルタントです。健康的なライフスタイル、食事、運動、メンタルヘルスについて、やさしく具体的にアドバイスしてください。医学的根拠も添えてください。",
    "教育の専門家": "あなたは教育の専門家であり、家庭教師のように親しみやすく丁寧に学習を支援します。学習意欲の向上、学習計画、苦手克服などに焦点を当てて回答してください。"
}

def ask_llm(user_input: str, expert_type: str) -> str:
    """
    入力テキストと専門家タイプを受け取り、LLMの回答を返す
    """

    system_message = SystemMessage(content=EXPERTS[expert_type])
    human_message = HumanMessage(content=user_input)
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)
    response = llm([system_message, human_message])
    return response.content

# --- Streamlit UI ---
st.title("専門家AIチャットデモ（LangChain×Streamlit）")
st.markdown("""
このアプリは、入力した質問に対して、選択した分野の専門家AIが回答します。
1. 専門家の種類を選択
2. 質問を入力
3. 送信ボタンを押すと、AIの回答が表示されます。
""")

expert_type = st.radio("専門家の種類を選択してください", list(EXPERTS.keys()))
user_input = st.text_area("質問を入力してください")

if st.button("送信"):
    if user_input.strip() == "":
        st.warning("質問を入力してください。")
    else:
        with st.spinner("AIが回答中..."):
            try:
                answer = ask_llm(user_input, expert_type)
                st.success("AIの回答:")
                st.write(answer)
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
