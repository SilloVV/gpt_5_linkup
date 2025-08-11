import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv
from linkup import LinkupClient
import json

# Load environment variables
load_dotenv()

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Linkup tool definition
linkup_tool = {
    "type": "function",
    "name": "search_linkup",
    "description": "Search for information using Linkup API to get current web information",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to find information on the web"
            },
            "depth": {
                "type": "string",
                "enum": ["standard", "deep"],
                "description": "The depth of search to perform. Standard for quick results, deep for comprehensive search."
            },
            "output_type": {
                "type": "string",
                "enum": ["sourcedAnswer", "searchResults"],
                "description": "Type of output to return. sourcedAnswer for a complete answer with sources, searchResults for raw search results."
            }
        },
        "required": ["query"]
    }
}

def search_linkup(query, depth="standard", output_type="sourcedAnswer"):
    linkup_client = LinkupClient(api_key=os.getenv("LINKUP_API_KEY"))
    response = linkup_client.search(
        query=query,
        depth=depth,
        output_type=output_type,
        include_domains=["legifrance.gouv.fr", "service-public.fr", "conseil-constitutionnel.fr", "assemblee-nationale.fr", "senat.fr", "insee.fr", "data.gouv.fr"]
    )
    return response

def get_gpt5_response_with_sources(question, conversation_history=None):
    """
    Get GPT-5 response with Linkup sources and conversation context
    
    Args:
        question: Current user question
        conversation_history: Previous messages in the conversation
    
    Returns:
        tuple: (response_text: str, sources: list of tuples (title, url))
    """
    # Build context with conversation history
    if conversation_history:
        context = "Contexte de conversation précédente:\n"
        for msg in conversation_history[-6:]:  # Keep last 6 messages for context
            context += f"{msg['role']}: {msg['content']}\n"
        context += f"\nNouvelle question: {question}"
        input_text = context
    else:
        input_text = question
    
    response = openai_client.responses.create(
        model="gpt-5-nano",
        input=input_text,
        text={"verbosity": "medium"},
        tools=[linkup_tool],
        instructions="Vous êtes un assistant juridique spécialisé dans le droit français. Répondez de manière précise, professionnelle et pédagogique. Citez toujours les articles de loi pertinents avec leurs références exactes. Structurez vos réponses avec des titres clairs. Adaptez votre niveau de langage à l'interlocuteur tout en restant rigoureux juridiquement. Si une question sort de votre domaine d'expertise juridique, redirigez vers les bonnes ressources. Utilisez un ton bienveillant mais autoritaire sur les questions de droit."
    )
    
    output_text = ""
    tool_results = []
    sources = []

    for item in response.output:
        if hasattr(item, "content") and item.content is not None:
            for content in item.content:
                if hasattr(content, "text"):
                    output_text += content.text
        elif item.type == "function_call" and item.name == "search_linkup":
            args = json.loads(item.arguments)
            linkup_result = search_linkup(**args)
            tool_results.append(linkup_result)
            
            # Extract title and URL tuples from linkup result
            if hasattr(linkup_result, 'sources') and linkup_result.sources:
                for source in linkup_result.sources:
                    if hasattr(source, 'url') and hasattr(source, 'name'):
                        sources.append((source.name, source.url))

    # Follow up with tool results if any
    if tool_results:
        follow_up_response = openai_client.responses.create(
            model="gpt-5-nano",
            input=f"Question: {question}\nTool results: {tool_results[0]}\nProvide a complete answer using this information.",
            text={"verbosity": "medium"},
            instructions="Vous êtes un assistant juridique spécialisé dans le droit français. Répondez de manière précise, professionnelle et détaillée. Citez toujours les articles de loi pertinents avec leurs références exactes. Structurez vos réponses avec des titres clairs. Adaptez votre niveau de langage à l'interlocuteur tout en restant rigoureux juridiquement. Si une question sort de votre domaine d'expertise juridique, redirigez vers les bonnes ressources. Utilisez un ton bienveillant mais autoritaire sur les questions de droit.Répond en format markdown."
        )
        
        output_text = ""  # Reset to get only the follow-up response
        for item in follow_up_response.output:
            if hasattr(item, "content") and item.content is not None:
                for content in item.content:
                    if hasattr(content, "text"):
                        output_text += content.text

    return output_text, sources

# Streamlit app configuration
st.set_page_config(
    page_title="Assistant GPT-5 avec Sources",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Assistant GPT-5 avec Sources Web")
st.markdown("Posez vos questions et obtenez des réponses avec des sources fiables.")

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display sources if available
        if message["role"] == "assistant" and "sources" in message:
            if message["sources"]:
                st.markdown("### 📚 Sources:")
                for i, (title, url) in enumerate(message["sources"], 1):
                    st.markdown(f"{i}. [{title}]({url})")

# Chat input
if prompt := st.chat_input("Posez votre question..."):
    # Add user message to conversation history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Recherche en cours..."):
            try:
                response_text, sources = get_gpt5_response_with_sources(
                    prompt, 
                    st.session_state.messages[:-1]  # Pass conversation history (excluding current message)
                )
                
                # Display response
                st.markdown(response_text)
                
                # Display sources
                if sources:
                    st.markdown("### 📚 Sources:")
                    for i, (title, url) in enumerate(sources, 1):
                        st.markdown(f"{i}. [{title}]({url})")
                
                # Add assistant response to conversation history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response_text,
                    "sources": sources
                })
                
            except Exception as e:
                st.error(f"Erreur: {str(e)}")
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"Désolé, une erreur s'est produite: {str(e)}",
                    "sources": []
                })

# Sidebar with conversation controls
with st.sidebar:
    st.header("🔧 Contrôles")
    
    if st.button("🗑️ Effacer la conversation"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown(f"**Messages dans la conversation:** {len(st.session_state.messages)}")
    
    st.header("ℹ️ À propos")
    st.markdown("""
    Cet assistant utilise:
    - **GPT-5** pour les réponses
    - **Linkup API** pour les sources web
    - **Contexte conversationnel** pour maintenir la cohérence
    
    Les sources sont limitées aux domaines officiels français.
    """)