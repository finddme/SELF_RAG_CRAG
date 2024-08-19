import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

def crawling_and_processing():
    print("--- sart crawling ---")
    request1 = requests.get("https://finddme.github.io/")
    html1 = request1.text
    soup1 = BeautifulSoup(html1, 'html.parser')
    links1 = soup1.select('h4 > a')
    urls=[]
    tags=["llm","dev","natural"]
    
    for link in links1:
        res={"title":"","url":""}
        if link.has_attr('href'):
            href=link.get('href')
            for t in tags:
                if t in href.split("/")[1]:
                    # urls.append("https://finddme.github.io"+href)
                    res["title"]+=link.find("li").get_text().strip("\n        ")
                    res["url"]+="https://finddme.github.io"+href
                    urls.append(res)

    docs =[]
    for url in urls:
        w_context=WebBaseLoader(url["url"]).load()[0]
        w_context.metadata["title"] = url["title"]
        docs.append([w_context])

    docs_list = [item for sublist in docs for item in sublist]
    print("--- finish crawling ---")

    print("--- data split ---")
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)
    print("--- split done ---")

    print("--- data restructure ---")
    space_check=re.compile("\s{1,}")
    chunks=[]
    pass_first=["finddme ","⊹  Portfolio","© 2024 viein_serenity_singularity_simplicity_savage. This work is liscensed under CC BY-NC 4.0."]
    for i in doc_splits:
        # c={'text':i.page_content, 'title':i.metadata["source"].split("/")[-2]}
        c={'text':i.page_content, 'source_title':i.metadata["title"], "source":i.metadata["source"]}
        if c["text"].split("\n")[0] in pass_first:pass
        else:
            # save_c={'text':re.sub(space_check," ",c['text']),'title':c['title']}
            save_c={'text':re.sub(space_check," ",c['text']),'source_title':c['source_title'],'source':c['source']}
            chunks.append(save_c)
    return chunks