from aibot import StreamlitAiBot
from aibot import AiFunction
import requests
import json



def run():
    if not StreamlitAiBot.is_initialized():
        StreamlitAiBot.initialize(streamlit_page_title='LG Chatbot',
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
                    "search_criteria": {
                        "type": "string",
                        "description": "The search criteria that will be used by a semantic search system to search for LG products." \
                            "Example: Reliable stainless steel dishwasher with adjustable rack and QuadWash",
                    },
                    "product_category": {
                        "type": "string",
                        "description": "The category of the product. Possible values: 'TVs', 'Dishwashers', 'Refrigerators', " \
                            "'Washers & Dryers', 'Cooking Appliances', 'Air Purifiers', 'Vacuums', 'Air Conditioners', 'Dehumidifiers', " \
                            "'Projectors', 'Monitors', 'Speakers', 'Appliances Accessories', 'Sound Bars', 'TV & Home Theater Accessories', " \
                            "'Computing Accessories', 'Wireless Headphones', 'Burners & Drives', 'Laptops', 'Blu-ray & DVD Players', " \
                            "'Digital Storage', 'Mobile Accessories'"
                    },
                    "price_filter": {
                        "type": "string",
                        "description": 'MongoDB Query Language (MQL) query to filter products by price. Example: {"price":{"$lt":500}}'
                    }
                },
                "required": ["search_query", "product_category"],
            }
        }
    


    def execute(self, args) -> 'AiFunction.Result':

        search_query = args['search_criteria']
        product_category = args['product_category']
        if 'price_filter' in args:
            price_filter = {"$and":[json.loads(args['price_filter']),{"price":{"$ne":0}}]}
        else:
            price_filter = {"price": {"$ne": 0}}
        # price_filter = {"$and":[{"$lt": 2000},{"price":{"$ne":0}}]} # For error testing
        embedding_vector = self.get_embedding(search_query)
        product_results = self.query_pinecone_db(embedding_vector, namespace=product_category, filter=price_filter)
        if 'matches' in product_results:
            results_to_show = product_results['matches'][:2]
            return AiFunction.Result(str(results_to_show), pin_to_memory=True)
        elif 'message' in product_results:
            return AiFunction.ErrorResult(product_results['message'])
        else:
            return AiFunction.ErrorResult(str(product_results))
    

    # Use a Kore.ai Azure service to generate an embedding vector of the search query
    def get_embedding(self, input):
        # Endpoint URL
        url = "https://lgretail2.openai.azure.com/openai/deployments/AdaEmbedding002/embeddings?api-version=2023-03-15-preview"
        
        # Make the POST request
        payload = {"input": input}
        headers = {"Content-Type":"application/json","api-key":"216ce4fb4d4c475b89d2c2c13fa5bd25"}
        response = requests.post(url, json=payload, headers=headers)
        
        # Extract the embedding vector from the response (assuming the response contains 'vector' key)
        embedding_vector = response.json()['data'][0]['embedding']
        
        return embedding_vector
    


    # Function to query Pinecone DB
    def query_pinecone_db(self, query_vector, namespace, filter:dict=None, top_k=7):
        # Endpoint URL
        url = "https://lg-pim-dev-d6c0941.svc.us-west4-gcp-free.pinecone.io/query"
        
        # Parameters
        payload = {
            "vector": [query_vector],
            "topK": top_k,
            "includeMetadata": True,
            "includeValues": False,
            "namespace": namespace
        }
        if filter is not None:
            payload['filter'] = filter

        # Make the POST request
        headers = {"Content-Type":"application/json","Api-Key":"ae2c9a79-7351-4615-a187-ba292601824f"}
        response = requests.post(url, json=payload, headers=headers)
        
        # Return the response (can be parsed as needed)
        return response.json()
    


if __name__ == "__main__":
    run()
