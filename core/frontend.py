from http.client import responses
import tempfile
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from loaders import *
from decouple import config
import os
from dotenv import load_dotenv

TIPOS_ARQUIVOS_VALIDOS = [
    "Site",
    "Youtube",
    "PDF",
    "CSV",
    "TXT",
]

# CONFIG_MODELOS = {
#     "OpenAI": {"modelos": ["gpt-4o-mini", "gpt-4o "], "chat": ChatOpenAI},
#     "Groq": {"modelos": ["llama-3.1-70b-versatile", "gemma2-9b-it"], "chat": ChatGroq},
# }


CONFIG_MODELOS = {
    "OpenAI": {"modelos": ["gpt-4o-mini"], "chat": ChatOpenAI},
}

load_dotenv()


def carrega_arquivos(tipo_arquivo, arquivo):
    if tipo_arquivo == "Site":
        documento = carrega_site(arquivo)
    if tipo_arquivo == "Youtube":
        documento = carrega_youtube(arquivo)
    if tipo_arquivo == "PDF":
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        documento = carrega_pdf(nome_temp)
    if tipo_arquivo == "CSV":
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        documento = carrega_csv(nome_temp)
    if tipo_arquivo == "TXT":
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        documento = carrega_txt(nome_temp)

    return documento


def carrega_modelo(provedor, modelo, tipo_arquivo, arquivo):
    documento = carrega_arquivos(tipo_arquivo, arquivo)

    system_message = """Você é um assistente amigável chamado Oráculo.
    Você possui acesso às seguintes informações vindas 
    de um documento {}: 

    ####
    {}
    ####

    Utilize as informações fornecidas para basear as suas respostas.

    Sempre que houver $ na sua saída, substitua por S.

    Se a informação do documento for algo como "Just a moment...Enable JavaScript and cookies to continue" 
    sugira ao usuário carregar novamente o Oráculo!""".format(
        tipo_arquivo, documento
    )
    template = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("placeholder", "{chat_history}"),
            ("system", "{input}"),
        ]
    )

    chat = CONFIG_MODELOS[provedor]["chat"](
        model_name=modelo,
        api_key=config("OPENAI_API_KEY"),
    )
    chain = template | chat
    st.session_state["chain"] = chain


def chat():
    st.header("🌐 Bem vindo ao nosso Chat", divider=True)
    chain = st.session_state.get("chain")
    if chain is None:
        st.error("Carregue o Oráculo")
        st.stop()
    memoria = st.session_state.get("memoria", ConversationBufferMemory())

    for message in memoria.buffer_as_messages:
        chat_general = st.chat_message(message.type)
        chat_general.markdown(message.content)

    input_user = st.chat_input("fala comigo bebê")
    if input_user:
        chat_general = st.chat_message("human")
        chat_general.markdown(input_user)

        chat = st.chat_message("ai")
        resposta = chat.write_stream(
            chain.stream(
                {"input": input_user, "chat_history": memoria.buffer_as_messages}
            )
        )

        memoria.chat_memory.add_user_message(input_user)
        memoria.chat_memory.add_ai_message(resposta)
        st.session_state["memoria"] = memoria


def sidebar():
    tabs = st.tabs(["Upload de Arquivos", "Seleção de Modelos"])
    error_placeholder = st.empty()
    with tabs[0]:
        tipo_arquivo = st.selectbox(
            "Selcione o tipo de arquvio", TIPOS_ARQUIVOS_VALIDOS
        )
        if tipo_arquivo == "Site":
            arquivo = st.text_input("Digite a URL do Site")
        if tipo_arquivo == "Youtube":
            arquivo = st.text_input("Digite a URL do vídeo")
        if tipo_arquivo == "PDF":
            arquivo = st.file_uploader("Suba o arquivo PDF", type=[".pdf"])
        if tipo_arquivo == "CSV":
            arquivo = st.file_uploader("Suba o arquivo CSV", type=[".csv"])
        if tipo_arquivo == "TXT":
            arquivo = st.file_uploader("Suba o arquivo TXT", type=[".txt"])

    with tabs[1]:
        provedor = st.selectbox(
            "Selcione o provedor do modelo que deseja utilizar", CONFIG_MODELOS.keys()
        )
        modelo = st.selectbox(
            "Selecione o modelo que deseja utilizar",
            CONFIG_MODELOS[provedor]["modelos"],
        )

        # api_key_provide = st.text_input(
        #     f"Digite a API KEY do provedor {provedor}",
        #     value=st.session_state.get(f"api_key{provedor}"),
        # )

        # st.session_state[f"api_key{provedor}"] = api_key_provide
    if st.button("Chamar o Oráculo", use_container_width=True):
        if not arquivo:
            error_placeholder.error(
                "Este campo é obrigatório. Por favor, preencha antes de enviar."
            )
        elif arquivo:
            error_placeholder.empty()
            carrega_modelo(provedor, modelo, tipo_arquivo, arquivo)


def main():
    with st.sidebar:
        sidebar()
    chat()


if __name__ == "__main__":
    main()
