from langchain.llms import OpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st

st.set_page_config(page_title="AI Search ChatBot", page_icon="🌏", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""", unsafe_allow_html=True)

st.title("AI Search ChatBot 🌏")

with st.sidebar:
    st.sidebar.image("image/57045756.png", use_column_width=True)
    st.sidebar.markdown(
    f"""
    Powered by the GitHub: [DaanishQ](https://github.com/DaanishQ)
    """, unsafe_allow_html=True)
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    
if not openai_api_key:
    st.warning("Please add your OpenAI API key to continue the GPT-3.5 Turbo conversation.")
    st.stop()


llm = OpenAI(temperature=0, model="gpt-3.5-turbo-0613", openai_api_key=openai_api_key, streaming=True)
tools = load_tools(["ddg-search"])
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

if prompt := st.chat_input(placeholder="What is this data about?"):
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(prompt, callbacks=[st_callback])
        st.write(response)
