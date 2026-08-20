import os
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

st.header("YouTube RAG Bot")

video_id = st.text_input("Enter YouTube Video ID")
question = st.text_input("Ask a question about the video")

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

if st.button("Ask"):
    if not video_id or not question:
        st.warning("Please enter both a video ID and a question.")
        st.stop()

    FAISS_PATH = f"faiss_index_{video_id}"

    if os.path.exists(FAISS_PATH):
        vector_store = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        with st.spinner("Fetching..."):
            try:
                fetched_transcript = YouTubeTranscriptApi().fetch(video_id, languages=['en'])
                transcript_list = fetched_transcript.to_raw_data()
                transcript = " ".join(chunk["text"] for chunk in transcript_list)
            except TranscriptsDisabled:
                st.error("No captions available for this video.")
                st.stop()
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.stop()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.create_documents([transcript])

            vector_store = FAISS.from_documents(chunks, embeddings)
            vector_store.save_local(FAISS_PATH)

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    prompt = PromptTemplate(
        template="""
          You are a helpful assistant.
          Answer ONLY from the provided transcript context.
          If the context is insufficient, just say you don't know.

          {context}
          Question: {question}
        """,
        input_variables=['context', 'question']
    )

    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")
    parser = StrOutputParser()

    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    main_chain = parallel_chain | prompt | llm | parser
    with st.spinner("Generating answer..."):
        result = main_chain.invoke(question)

    st.write(result)
