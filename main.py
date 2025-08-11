from openai import OpenAI
import os
from dotenv import load_dotenv
from linkup import LinkupClient

# Load environment variables from .env file
load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

def get_gpt5_response_with_sources(question):
    """
    Get GPT-5 response with Linkup sources
    
    Returns:
        tuple: (response_text: str, sources: list of tuples (title, url))
    """
    response = openai_client.responses.create(
        model="gpt-5-nano",
        input=question,
        text={"verbosity": "medium"},
        tools=[linkup_tool],
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
            import json
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
            text={"verbosity": "medium"}
        )
        
        for item in follow_up_response.output:
            if hasattr(item, "content") and item.content is not None:
                for content in item.content:
                    if hasattr(content, "text"):
                        output_text += content.text

    return output_text, sources

question = " Explique moi les principes de l'article L121-2 du code de commerce ?"

response_text, sources = get_gpt5_response_with_sources(question)

print(f"Réponse: {response_text}")
print(f"\nSources trouvées: {len(sources)}")
for i, (title, url) in enumerate(sources, 1):
    print(f"{i}. {title}: {url}")