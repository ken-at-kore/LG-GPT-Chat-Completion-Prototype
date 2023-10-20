from aibot import StreamlitAiBot
from aibot import AiFunction
import requests
import json




def run():
    if not StreamlitAiBot.is_initialized():
        StreamlitAiBot.initialize(streamlit_page_title='LG AiBot Prototype',
                            openai_model='gpt-3.5-turbo',
                            model_temperature=0.25,
                            system_prompt_engineering=open('prompts & content/system prompt.md').read(),
                            welcome_message=open('prompts & content/welcome message.md').read(),
                            ai_functions=[SearchForLGProducts()]
        )
    StreamlitAiBot.run()



class SearchForLGProducts(AiFunction):

    def get_spec(self):
        return {
            "name": "search_for_lg_products",
            "description": "Search for an LG product.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                        "description": "The search criteria that will be used by a semantic search system to search for LG products.",
                    },
                    "product_category": {
                        "type": "string",
                        "description": "The category of the product. Examples: TVs, Laptops, Refrigerators, Dishwashers"
                    }
                },
                "required": ["search_query", "product_category"],
            }
        }
    

    def execute(self, args) -> 'AiFunction.Result':

        search_query = args['search_query']
        product_category = args['product_category']
        embedding_vector = self.get_embedding(search_query)
        product_results = self.query_pinecone_db(embedding_vector, namespace=product_category)
        results_to_show = product_results['matches'][:3]
        return AiFunction.Result(str(results_to_show))
    


    def get_embedding(self, input):

        # Endpoint URL
        url = "https://lgretail2.openai.azure.com/openai/deployments/AdaEmbedding002/embeddings?api-version=2023-03-15-preview"
        
        # Headers
        headers = {
            "Content-Type": "application/json",
            "api-key": "216ce4fb4d4c475b89d2c2c13fa5bd25"
        }
        
        # Body
        payload = {
            "input": input
        }
        
        # Make the POST request
        response = requests.post(url, json=payload, headers=headers)
        
        # Extract the embedding vector from the response (assuming the response contains 'vector' key)
        embedding_vector = response.json()['data'][0]['embedding']
        
        return embedding_vector
    

    # Function to query Pinecone DB
    def query_pinecone_db(self, query_vector, namespace, top_k=7):
        # Endpoint URL
        url = "https://lg-pim-dev-d6c0941.svc.us-west4-gcp-free.pinecone.io/query"
        
        # Parameters
        payload = {
            "vector": [query_vector],
            "topK": top_k,
            # "filter": {"Battery": "80Wh"},
            "includeMetadata": True,
            "includeValues": False,
            "namespace": namespace
        }
        
        headers = {
            "Content-Type": "application/json",
            "Api-Key": "ae2c9a79-7351-4615-a187-ba292601824f"
        }
        
        # Make the POST request
        response = requests.post(url, json=payload, headers=headers)

        print(response.json())
        
        # Return the response (can be parsed as needed)
        return response.json()
    


if __name__ == "__main__":
    run()
