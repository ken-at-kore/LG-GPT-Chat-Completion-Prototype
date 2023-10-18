from aibot import AiFunction

class SearchForLGProducts(AiFunction):

    def get_spec(self):
        return {
            "name": "search_for_products",
            "description": "Get the best TVs LG offers",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                        "description": "A search query string to use to search for LG products.",
                    }
                },
                "required": ["search_query"],
            }
        }
    

    def execute(self, args) -> 'AiFunction.Result':
        return AiFunction.Result("LG OLED evo G3 55 inch 4K Smart TV 2023, $1917.83")