import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.memory import ConversationBufferMemory # 몇줄 전까지의 대화 내용 기억 

def main() :
    
    st.set_page_config(page_title="Hansol Guard", 
                       page_icon=":construction:", 
                       layout="centered",)
    
    st.title("Hansol Guard QA ChatBot :construction:")
    
    with st.sidebar : 
        st.subheader("사고 정보 입력란")
        st.markdown(" ")
        work_process = st.text_input("**✅ 작업 프로세스**", value = "ex) 절단작업, 타설작업")
        accident_object = st.text_input("**✅ 사고객체**", value = "ex) 공구류")
        human_accident = st.text_input("**✅ 인적사고**", value = "ex) 끼임")
        property_accident = st.text_input("**✅ 물적사고**", value = "ex) 없음")
    
    if "conversation" not in st.session_state : 
        st.session_state.conversation = None 
    
    if "chat_history" not in st.session_state : 
        st.session_state.chat_history = None 
    
    if 'message' not in st.session_state :
        st.session_state['messages'] = [{"role" : "assistant" , 
                                        "content" : "안녕하세요! 사고 상황을 설명해주시면, 사고 처리에 필요한 정보를 제공해드리겠습니다!"}]
        
    for message in st.session_state.messages : 
        with st.chat_message(message['role']) :
            st.markdown(message['content']) # message의 role에 따라서 content를 markdown으로 출력
            
    history = StreamlitChatMessageHistory(key = "chat_messages")
    
    # chat logic
    if query := st.chat_input("사고 원인을 알려주세요.") : # 질문창 
        st.session_state.messages.append({"role" : "user", "content" : query})
        
        with st.chat_message("user") : 
            st.markdown(query)
        
        with st.chat_message("assistant") : 
            
            chain = st.session_state.conversation
            
            with st.spinner("Thinking...") :
                result = chain({"question" : query}) # chain을 통해 LLM의 답변이 나오도록 함 
                response = result["answer"]
                
                source_documents = result["source_documents"]
                
                st.markdown(response)
                with st.expander("참고 문서 확인") :
                    st.markdown("이 답변은 다음 문서들을 참고하여 작성되었습니다.")
                    for i , doc in enumerate(source_documents) :
                        st.markdown(f"{i}.{doc.metadata['source']}" , help = doc.page_content)
                        
        st.session_state.messages.append({"role" : "assistant", "content" : response})
    

def get_conversation_chain(vector_stotre , retreiver , custom_llm) :
    
    llm = custom_llm(tempearture = 0)
    
    conversation_chain = ConversationalRetrievalchain.from_llm(
        llm = llm , 
        chain_type = "stuff" ,
        retriever = retriever,
        return_source_documents = True,
        memory = ConversationBufferMemory(
            memory_key = "chat_history", # chat_history의 키 값을 가진 채팅 기록 가져와서 이것을 context에 집어넣어 이전 대화를 기억하게 하기
            return_messages = True,
            output_key = "answer"), # 답변만을 history에 저장하겠다.
        verbose = True,
        get_chat_history = lambda h : h # 메모리가 들어온 그래도 chat history를 사용하겠다.
        )
    
    return conversation_chain
    
if __name__ == "__main__" :
    main()

        
        
