import streamlit as st
from ollama import Client
import time

def main():
    st.title("🦙 Ollama LLM Chat - Advanced")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "available_models" not in st.session_state:
        st.session_state.available_models = []
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        ollama_host = st.text_input("Ollama Server URL", "http://localhost:11434")
        client = Client(host=ollama_host)
        
        # Model management section
        st.subheader("Model Management")
        new_model = st.text_input("Model to pull (name:tag)", "llama2")
        
        if st.button("Pull Model"):
            with st.spinner(f"Pulling {new_model}..."):
                try:
                    progress = client.pull(new_model, stream=True)
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for chunk in progress:
                        if "completed" in chunk and "total" in chunk:
                            percent = chunk["completed"] / chunk["total"]
                            progress_bar.progress(percent)
                            status_text.text(f"Downloading: {percent*100:.1f}%")
                    
                    st.success(f"Model {new_model} pulled successfully!")
                    time.sleep(2)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error pulling model: {str(e)}")
        
        # Refresh available models
        if st.button("Refresh Models"):
            try:
                models = client.list()
                st.session_state.available_models = [model["name"] for model in models["models"]]
                st.success("Models refreshed!")
            except Exception as e:
                st.error(f"Error refreshing models: {str(e)}")
        
        # Get available models if not already loaded
        if not st.session_state.available_models:
            try:
                models = client.list()
                st.session_state.available_models = [model["name"] for model in models["models"]]
            except Exception as e:
                st.error(f"Could not connect to Ollama server: {str(e)}")
                st.session_state.available_models = ["llama2"]  # default
        
        model_name = st.selectbox(
            "Select Model",
            st.session_state.available_models,
            index=0
        )
        
        # Generation parameters
        st.subheader("Generation Parameters")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
        max_tokens = st.number_input("Max Tokens", 100, 4096, 512)
        top_p = st.slider("Top-P", 0.0, 1.0, 0.9)
        repeat_penalty = st.slider("Repeat Penalty", 1.0, 2.0, 1.1)
        
        if st.button("Clear Chat"):
            st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to ask?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # Stream the response
                for chunk in client.generate(
                    model=model_name,
                    prompt=prompt,
                    stream=True,
                    options={
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "top_p": top_p,
                        "repeat_penalty": repeat_penalty
                    }
                ):
                    full_response += chunk.get("response", "")
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

if __name__ == "__main__":
    main()