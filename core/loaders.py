from langchain_community.document_loaders import (
    TextLoader,
    WebBaseLoader,
    YoutubeLoader,
    CSVLoader,
    PyPDFLoader,
)


def carrega_site(url):
    loader = WebBaseLoader(url)
    lista_documentos = loader.load()
    documento = "\n\n".join([doc.page_content for doc in lista_documentos])
    return documento


def carrega_youtube(url):
    loader = YoutubeLoader(url, add_video_info=False, language=["pt"])
    lista_documentos = loader.load()
    documento = "\n\n".join([doc.page_content for doc in lista_documentos])
    return documento


def carrega_csv(path):
    loader = CSVLoader(path)
    lista_documentos = loader.load()
    documento = "\n\n".join([doc.page_content for doc in lista_documentos])
    return documento


def carrega_txt(path):
    loader = TextLoader(path)
    lista_documentos = loader.load()
    documento = "\n\n".join([doc.page_content for doc in lista_documentos])
    return documento


def carrega_pdf(path):
    loader = PyPDFLoader(path)
    lista_documentos = loader.load()
    documento = "\n\n".join([doc.page_content for doc in lista_documentos])
    return documento
