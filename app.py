import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory # 대회 기록 저장 
from langchain.memory import ConversationBufferMemory # 이전 대화 기억 

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
    
    # 초기 인삿말 출력 
    if 'message' not in st.session_state :
        st.session_state['messages'] = [{"role" : "assistant" , 
                                        "content" : "안녕하세요! 사고 상황을 설명해주시면, 사고 처리에 필요한 정보를 제공해드리겠습니다!"}]
    
    # 이전 대화 표시 
    for message in st.session_state.messages : 
        with st.chat_message(message['role']) :
            st.markdown(message['content']) # message의 role에 따라서 content를 markdown으로 출력
            
    history = StreamlitChatMessageHistory(key = "chat_messages") # langchain 메모리 설정 
    

    if query := st.chat_input("사고 원인을 알려주세요.") : # 사용자의 메세지를 입력으로 저장 (query)
        
        st.session_state.messages.append({"role" : "user", 
                                          "content" : query}) # 사용자의 입력을 대화에 추가 
        
        with st.chat_message("user") : 
            st.markdown(query)
            
        # 사고 정보과 사고원인 결합 
        combined_query = f"""
        {query} 로 인해 사고가 발생했습니다. 
        해당 사고는 {work_process} 중 발생했으며, 관련 사고 객체는 {accident_object} 입니다. 
        이로 인한 인적피해는 {human_accident} 이고, 물적피해는 {property_accident} 로 확인됩니다.
        재발 방지 대책 및 향후 조치 계획은 무엇인가요?
        """
        
        # llm 답변을 conversation에 추가 
        st.session_state.conversation = get_conversation_chain(vectorstore , llm)
        
        with st.chat_message("assistant") : 
            
            chain = st.session_state.conversation
            
            with st.spinner("Thinking...") :
                result = chain({"question" : combined_query}) # chain을 통해 LLM의 답변이 나오도록 함 
                response = result["answer"]
                
                source_documents = result["source_documents"]
                
                st.markdown(response)
                with st.expander("참고 문서 확인") :
                    st.markdown("이 답변은 다음 문서들을 참고하여 작성되었습니다.")
                    for i , doc in enumerate(source_documents) :
                        st.markdown(f"{i}.{doc.metadata['source']}" , help = doc.page_content)
                        
        st.session_state.messages.append({"role" : "assistant", "content" : response}) # 생성된 답변을 history에 추가 



# 입력받은 text로 retreiver 하여 미리 정의된 vector db에서 유사 문서 가져오기.  
# llm에 유사문서 넣고 llm 답변 generate 하기. 
# chain 생성하기하여 streamlit 에서 적용하기. 


    
if __name__ == "__main__" :
    main()

        
        
