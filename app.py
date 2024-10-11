import time
import streamlit as st
from src.logger import logger
from src.helper import (
    create_vectorstore,
    load_vectorstore,
    get_pdf_elements,
    get_elements_list,
    get_text_summary,
    get_table_summary,
    get_image_summary,
    create_multi_vector_retriever,
    get_conversational_chain,
    plt_img_base64
)

def display_chat_history(chat_history):
    for message in chat_history:
        if message['role'] == 'user':
            st.write("**User** 👨🏻‍💻: ", message['content'])
        elif message['role'] == 'assistant':
            if message['content']['image']:
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write("**Reply** 📝: ", message['content']['text'])
                with col2:
                    st.image(f"data:image/jpeg;base64,{message['content']['image']}", caption="Relevant Image", use_column_width=True)

            else:
                st.write("**Reply** 📝: ", message['content']['text']) 

def user_input():
    #with st.form(key='user_question_form'):
    col1, col2 = st.columns([4, 1])
    with col1:
        user_question = st.text_input("Ask a Question from the PDF File", key="user_question_input")
    with col2:
        st.text("")
        st.text("")
        submit_button = st.button("✈️", key="submit_button", help="Submit your question")

    if submit_button and user_question:
        if st.session_state.retriever is None or st.session_state.loaded_vectorstore is None:
            st.error("Please upload PDF documents and click 'Submit & Process' to initialize the conversation.")
            return

        # Use the retriever and loaded_vectorstore from session state
        result, relevant_image = get_conversational_chain(
            st.session_state.retriever,
            st.session_state.loaded_vectorstore,
            user_question)

        # Update chat history
        st.session_state.chat_history.append({"content": user_question, "role": "user"})
        # Append assistant response with text and optional image
        assistant_response = {
            "text": result,
            "image": plt_img_base64(relevant_image) if relevant_image else None
        }
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

        display_chat_history(st.session_state.chat_history)
def main():
    st.set_page_config(page_title="Information Retrieval")
    st.header("Multimodal Conversational Assistant💁")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "loaded_vectorstore" not in st.session_state:
        st.session_state.loaded_vectorstore = None

    user_input()

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF File and Click on the Submit & Process Button", 
            accept_multiple_files=False
        )
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.error("No PDF document uploaded. Please upload a PDF file.")
            else:
                try:
                    start_time = time.time()
                    with st.spinner("Processing PDF..."):
                        # Step 1: Process the PDF
                        raw_pdf_elements = get_pdf_elements(pdf_docs)
                        logger.info("Raw PDF elements processed successfully.")
                        #st.success("Raw PDF elements extracted.")

                    with st.spinner("Extracting Narrative Text and Table..."):
                        NarrativeText, Table = get_elements_list(raw_pdf_elements)
                        logger.info("Text and Table extracted successfully.")
                        #st.success("Text and Table extracted.")

                        # Step 2: Get summaries for Narrative Text, Tables and Images
                    with st.spinner("Generating text summaries..."):
                        text_summaries = get_text_summary(NarrativeText)
                        logger.info("Text summaries generated successfully.")
                        #st.success("Text summaries generated.")

                    with st.spinner("Generating table summaries..."):
                        table_summaries = get_table_summary(Table)
                        logger.info("Table summaries generated successfully.")
                        #st.success("Table summaries generated.")

                    with st.spinner("Extracting image summaries..."):
                        img_base64_list, image_summaries = get_image_summary()
                        logger.info("Image summaries extracted successfully.")
                        #st.success("Image summaries extracted.")

                        # Step 3: Create the vector store
                    with st.spinner("Creating vector store..."):
                        create_vectorstore()  # Deletes old vector store if it exists
                        logger.info("Vector store created successfully.")

                        # Step 4: Load or create the vector store
                    with st.spinner("Loading vector store..."):
                        loaded_vectorstore = load_vectorstore()
                        if loaded_vectorstore is None:
                            st.error("Failed to load vector store.")
                            return
                        logger.info("Vector store loaded successfully.")

                        # Step 5: Create the multi-vector retriever
                    with st.spinner("Creating multi-vector retriever..."):
                        try:
                            retriever_multi_vector_img = create_multi_vector_retriever(
                                loaded_vectorstore,
                                text_summaries,
                                NarrativeText,
                                table_summaries,
                                Table,
                                image_summaries,
                                img_base64_list
                            )
                            logger.info("Multi-vector retriever created successfully.")
                            #st.success("Multi-vector retriever created successfully.")
                        except Exception as e:
                            st.error("Error creating multi-vector retriever.")
                            logger.error(f"Error creating multi-vector retriever: {e}")

                        # Step 6: Store retriever and vector store for use
                        st.session_state.retriever = retriever_multi_vector_img
                        st.session_state.loaded_vectorstore = loaded_vectorstore
                        st.success(f"Done in {(time.time() - start_time) / 60:.2f} minutes.")

                except Exception as e:
                    st.error("An error occurred while processing the PDF.")
                    logger.error(f"Error during PDF processing: {e}")

        if st.button("Reset Conversation"):
            st.session_state.chat_history = []
            #st.session_state.retriever = None
            #st.session_state.loaded_vectorstore = None
            st.success("Conversation has been reset.")
    # Footer
    st.markdown("""
    <style>
    .developer-label {
        position: fixed;
        bottom: 0;
        width: calc(100% - var(--sidebar-width, 0px));
        text-align: center;
        background-color: #f0f0f0;
        padding: 3px;
        border-top: 1px solid #ddd;
        left: var(--sidebar-width, 0px);
    }
    </style>
    <div class="developer-label">
        <p>Developed by Kousik Naskar | Email: <a href="mailto:kousik23naskar@gmail.com">kousik23naskar@gmail.com</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()