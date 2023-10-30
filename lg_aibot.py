import requests
import json
import typing
from typing import List
import traceback
import streamlit as st
from datetime import datetime
import openai
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory



class AiFunction:
    def __init__(self):
        assert hasattr(self.__class__, 'get_spec'), f'{self.__class__.__name__} must define a get_spec() method'
        assert hasattr(self.__class__, 'execute'), f'{self.__class__.__name__} must define an execute(args) method'

    class Result:
        def __init__(self, value:str, pin_to_memory=False):
            self.value = value
            self.pin_to_memory = pin_to_memory
            self.is_an_error_result = False

    class ErrorResult(Result):
        def __init__(self, error_message:str):
            super().__init__('Error: ' + error_message)
            self.is_an_error_result = True

    class Collection:
        def __init__(self, ai_functions:List=[]):
            self.functions = {}
            for ai_func in ai_functions:
                assert isinstance(ai_func, AiFunction), f"{ai_func} is not an instance of AiFunction"
                function_name = ai_func.get_spec()['name']
                self.functions[function_name] = ai_func
        
        def is_empty(self):
            return not self.functions

        def get_function_specs(self):
            if not self.is_empty():
                return [function.get_spec() for function in self.functions.values()]
            return None
        
        def get_function(self, name):
            return self.functions.get(name)



class StreamlitAiBot:

    @staticmethod
    def initialize(streamlit_page_title:str="My AiBot",
                 openai_model:str="gpt-3.5-turbo",
                 openai_api_key:str=None,
                 model_temperature:float=0.25,
                 system_prompt_engineering:str='You are a helpful assistant.',
                 welcome_message:str='Hello! How can I assist you today?',
                 ai_functions:List[AiFunction]=None
    ):
        print("AiBot: Initializing session.")
        bot = StreamlitAiBot(streamlit_page_title=streamlit_page_title,
                             openai_model=openai_model,
                             openai_api_key=openai_api_key,
                             model_temperature=model_temperature,
                             system_prompt_engineering=system_prompt_engineering,
                             welcome_message=welcome_message,
                             ai_functions=ai_functions
        )
        st.session_state['ai_bot'] = bot


    
    @staticmethod
    def is_initialized():
        return 'ai_bot' in st.session_state



    def __init__(self, **kwargs):
        # Initialize fields
        self.streamlit_page_title = kwargs.get('streamlit_page_title')
        self.openai_model = kwargs.get('openai_model')
        openai_api_key = kwargs.get('openai_api_key')
        self.model_temperature = kwargs.get('model_temperature')
        self.system_prompt_engineering = kwargs.get('system_prompt_engineering')
        self.welcome_message = kwargs.get('welcome_message')

        # Initialize AiFunctions
        ai_funcs_arg = kwargs.get('ai_functions', None)
        self.ai_functions = AiFunction.Collection(ai_funcs_arg) if ai_funcs_arg is not None else AiFunction.Collection()

        # Initialize Conversation Memory
        self.convo_memory = StreamlitAiBot.ConvoMemory(self.system_prompt_engineering, self.welcome_message)

        # Initialize internal configs
        self.do_cot = False
        self.max_function_errors_on_turn = 1
        self.max_main_gpt_calls_on_turn = 4

        # Set the OpenAI key and model
        if openai_api_key is not None and openai_api_key != '':
            openai.api_key = openai_api_key
            st.session_state["openai_model"] = self.openai_model
            print("AiBot: Using user OpenAI API key.")
        else:
            openai.api_key = st.secrets["OPENAI_API_KEY"]
            st.session_state["openai_model"] = self.openai_model
            print("AiBot: Using configured OpenAI API key.")



    @staticmethod
    def runBot():
        """
        Get the AiBot from the Streamlit session and run it.
        """
        bot = st.session_state['ai_bot']
        assert bot is not None, "StreamlitAiBot has not been initialized"
        # assert isinstance(bot, StreamlitAiBot), "Streamlit session ai_bot is not of type StreamlitAiBot" # This assertion always fails when deployed to Streamlit Cloud for some reason
        bot.run()



    def run(self):
        """
        Run AiBot's main loop. The bot takes a turn.
        """

        print("AiBot: Running.")

        # Display title
        # (This needs to happen on every Streamlit run)
        st.title(self.streamlit_page_title)
        st.caption(f"Model: {st.session_state['openai_model']}")

        # Initialize UI messages
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": self.welcome_message}
            ]

        # Re-render UI messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Get, store and render user message
        if user_input := st.chat_input("Enter text here", key="real_input"):
            print(f"AiBot: User input: {user_input}")

            # Disable the input UI
            # st.chat_input(" . . .", key="disabled_input", disabled=True) #TODO: This isn't working

            # Display user input
            user_input = user_input.replace('$','\$') # Try to sanitize against LaTeX markup
            st.session_state.messages.append({"role": "user", "content": user_input}) # Add it to the UI thread
            with st.chat_message("user"):
                st.markdown(user_input) # Render it

            # Summarize conversation memory before proceeding
            self.convo_memory.summarize_memory()
            self.convo_memory.add_user_msg(user_input) # Add user input to conversation memory

            # Initialize processing counters
            self.function_error_count = 0
            self.call_and_process_count = 0

            # Call GPT with the input and process results
            self.call_and_process_gpt()

            # Re-run the Streamlit app to re-enable the input UI
            # st.rerun() #TODO: This isn't working

        else:
            print("AiBot: Didn't process input.")


    def call_and_process_gpt(self):
        """
        Use the chat history in Conversation Memory to call ChatGPT.
        Process the bot message and function call results.
        """

        # Keep track of how often call & processing happens this turn
        self.call_and_process_count += 1

        # Prepare GPT messages (conversation context). Include Chain of Thought logging for improved decision making.
        convo_context = self.convo_memory.get_messages()
        if self.do_cot:
            cot_result = self.generate_chain_of_thought_logging()
            if cot_result:
                convo_context += [{'role': 'function', 'name': 'do_chain_of_thought_logging', 'content': cot_result}]

        # Prepare assistant response UI
        function_call_name = ""
        function_call_response = ""
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            bot_content_response = ""

            # Prepare OpenAI GPT call parameters
            chat_completion_params = {
                'model': self.openai_model,
                'messages': convo_context,
                'stream': True,
                'temperature': self.model_temperature
            }

            # Add function calls & maybe error handling prompt engineering
            too_many_errors = self.function_error_count > self.max_function_errors_on_turn
            too_many_main_gpt_calls = self.call_and_process_count > self.max_main_gpt_calls_on_turn #TODO: Is it > or >=?
            if not self.ai_functions.is_empty() and not too_many_errors and not too_many_main_gpt_calls:
                chat_completion_params['functions'] = self.ai_functions.get_function_specs()
            if too_many_errors:
                chat_completion_params['messages'] += [{'role': 'system', 'content': 'There are function calling errors. Apologize to the user and ask them to try again.'}]

            # Call OpenAI GPT (Response is streamed)
            print("AiBot: Calling GPT & streaming...")
            for response in openai.ChatCompletion.create(**chat_completion_params):

                # Handle content stream
                if not response.choices[0].delta.get("function_call",""):
                    content_chunk = response.choices[0].delta.get("content", "")
                    content_chunk = content_chunk.replace('$','\$')
                    bot_content_response += content_chunk
                    full_response += content_chunk
                    message_placeholder.markdown(full_response + "▌")
                
                # Handle function call stream
                else:
                    call = response.choices[0].delta.function_call
                    if function_call_name == "":
                        function_call_name = call.get("name", "")
                    function_call_response += call.get("arguments", "")

            if not function_call_response:
                message_placeholder.markdown(full_response)
            else: 
                message_placeholder.markdown('Just a sec 🔍')

        # Handle no function call
        if not function_call_response:
            print(f"AiBot: GPT assistant message: {bot_content_response}")
            # Store the bot content
            self.convo_memory.add_assistant_msg(bot_content_response)
            st.session_state.messages.append({"role": "assistant", "content": bot_content_response})

        # Handle function call
        else:
            print(f'AiBot: Got GPT function call "{function_call_name}". Content: "{bot_content_response}". Arguments: {function_call_response}')

            # Store bot content including function call name and arguments
            self.convo_memory.add_assistant_msg(bot_content_response, function_call_name, function_call_response)

            # Add bot content to UI thread
            if bot_content_response != "":
                st.session_state.messages.append({"role": "assistant", "content": bot_content_response})

            # Execute function call
            function_obj = self.ai_functions.get_function(function_call_name)
            assert function_obj is not None, f'Function {function_call_name} is not defined in the function collection'
            try:
                func_call_result = function_obj.execute(json.loads(function_call_response))
                assert isinstance(func_call_result, AiFunction.Result), f"func_call_results for {function_call_name} must be of type AiFunction.Result, not {type(func_call_result)}"
            except Exception as e:
                error_info = str(e)
                print(f"AiBot: Error executing function {function_call_name}.\n{error_info}")
                traceback.print_exc()
                func_call_result = AiFunction.ErrorResult(f"Caught exception when executing function {function_call_name}: {error_info}")
                # exception_traceback = traceback.format_exc()
                # raise e

            # Process function call result
            func_call_result_str = func_call_result.value
            if func_call_result.is_an_error_result:
                self.function_error_count += 1

            # Store query results for GPT
            print(f"AiBot: Function execution result: {func_call_result_str}")
            self.convo_memory.add_function_msg(function_call_name, func_call_result_str, func_call_result.pin_to_memory)

            # Recursively call this same function to process the query results
            self.call_and_process_gpt()



    def generate_chain_of_thought_logging(self) -> str:
        """"
        Implements the Chain of Thought technique by having the bot explain the context and then the next step.
        This generated CoT content is then fed into the next GPT call.
        """

        cot_func_spec = {
            "name": "do_chain_of_thought_logging",
            "description": "Log your chain of thought.",
            "parameters": {
                "type": "object",
                "properties": {
                    "describe_the_context": {
                        "type": "string",
                        "description": "In one sentence, describe the conversation context.",
                    },
                    "describe_what_you_should_do_next": {
                        "type": "string",
                        "description": "In one sentence, describe what you think you should do next."
                    }
                },
                "required": ["describe_the_context", "describe_what_you_should_do_next"],
            }
        }
        messages = self.convo_memory.get_messages()
        funcs = self.ai_functions.get_function_specs() + [cot_func_spec]
        print("AiBot: Calling GPT for chain of thought...")
        try:
            response = openai.ChatCompletion.create(
                model=self.openai_model,
                messages=messages,
                temperature=self.model_temperature,
                functions=funcs,
                function_call={"name": "do_chain_of_thought_logging"}
            )
        except Exception as e:
            print("Caught error when doing chain of thought.")
            traceback.print_exc()
            return None
        response = json.loads(response.choices[0].message.function_call.arguments)
        cot_result = f"{response['describe_the_context']} | {response['describe_what_you_should_do_next']}"
        print(f"AiBot: Chain of thought: {cot_result}")
        return cot_result
        


    class ConvoMemory:
        """
        This encapsulates the conversation memory and context. Beside keeping track of the running messages list,
        this memory generates conversation summaries to reduce tokens sent to the GPT context window.
        """

        def __init__(self, system_prompt_engineering:str = "You are a helpful assistant.",
                     bot_welcome_message:str = None):
            
            self.system_prompt_engineering = system_prompt_engineering # The primary content for the system message

            llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0, openai_api_key=openai.api_key)
            self.langchain_memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=500, return_messages=True,
                                                                    human_prefix='user', ai_prefix='assistant')
            
            self.convo_summary_text = None # The conversatoin summary that will be appended to the system message content
            self.interaction_messages = [] # The messages excluding the first system message
            self.last_bot_message = None
            self.function_result_to_pin = None
            self.pinned_function_result = None
            if bot_welcome_message is not None:
                welcome_gpt_message = {'role':'assistant', 'content':bot_welcome_message}
                self.last_bot_message = welcome_gpt_message
                self.langchain_memory.save_context({'input': "<Bot initialized>"}, {'output': bot_welcome_message})

        def add_user_msg(self, content:str):
            self.interaction_messages.append({'role':'user', 'content':content})

        def add_assistant_msg(self, content:str, function_call_name:str=None, function_call_arguments:str=None):
            msg = {'role':'assistant', 'content':content}
            if function_call_name is not None or function_call_arguments is not None:
                assert function_call_name is not None and function_call_arguments is not None, "Both function_call_name and function_call_arguments need to be set."
                msg['function_call'] = {'name': function_call_name, 'arguments': function_call_arguments}
            self.interaction_messages.append(msg)

        def add_function_msg(self, function_name:str, content:str, pin_func_to_memory:bool=False):
            self.interaction_messages.append({'role':'function', 'name':function_name, 'content':content})
            if pin_func_to_memory:
                self.function_result_to_pin = {'name': function_name, 'result': content}
                previous_msg = self.interaction_messages[-2]
                if previous_msg['role'] == 'assistant' and 'function_call' in previous_msg \
                        and previous_msg['function_call']['name'] == function_name:
                    self.function_result_to_pin['arguments'] = previous_msg['function_call']['arguments']
                self.pinned_function_result = None

        def get_messages(self):
            result_system_content = self.system_prompt_engineering
            result_system_content += '\n\nCurrent date: ' + datetime.now().strftime('%Y-%m-%d')
            if self.convo_summary_text is not None:
                result_system_content += "\n\n---\n\nCONVERSATION SUMMARY\n\n" + self.convo_summary_text
            if self.pinned_function_result is not None:
                result_system_content += f"\n\n---\n\nLAST FUNCTION CALL"
                result_system_content += f"\n\nFunction name: {self.pinned_function_result['name']}"
                result_system_content += f"\nFunction call arguments: {self.pinned_function_result['arguments']}"
                result_system_content += f"\n\nFunction result: {self.pinned_function_result['result']}"
            result_messages = [{'role':'system', 'content':result_system_content}]
            if self.last_bot_message is not None:
                result_messages.append(self.last_bot_message)
            result_messages += self.interaction_messages
            return result_messages
        
        def summarize_memory(self):
            if len(self.interaction_messages) > 1:
                print("AiBot: Summarizing memory...")
                user_content = next((d['content'] for d in self.interaction_messages if d['role'] == 'user'), None)
                assistant_content = '\n\n'.join([d['content'] for d in self.interaction_messages if d['role'] == 'assistant'])
                self.langchain_memory.save_context({'input': user_content}, {'output': assistant_content})
                self.convo_summary_text = str(self.langchain_memory.load_memory_variables({}))
                if self.function_result_to_pin is not None:
                    self.pinned_function_result = self.function_result_to_pin
                
                assert self.interaction_messages[-1]['role'] == 'assistant', "Last interaction message was not an assistant message"
                self.last_bot_message = self.interaction_messages[-1]
                self.interaction_messages = []
                print(f"AiBot: Got summary: {self.convo_summary_text}")



        def kens_convo_summary_algorithm(self):
            """
            Deprecated. This will get replaced by the LangChain system.
            """
            #TODO: Move numbers to configuration
            memory_messages_capacity_threshold = 8
            target_msg_count_to_summarize = 4
            if len(self.interaction_messages) > memory_messages_capacity_threshold:
                print("AiBot: Compressing memory...")
                target_messages = self.interaction_messages[:target_msg_count_to_summarize] #TODO: Get the older messages
                compression_system_prompt = 'You are an expert conversation designer and a helpful assistant.'
                compression_user_prompt = 'Summarize the following GPT conversation. Be specific and include facts. ' \
                                            'But you must write less than 150 words.\n\n'
                if self.convo_summary_text is not None:
                    compression_user_prompt += f'Include this partial summary info in your summary: {self.convo_summary_text}\n\nNew part of conversation: '
                compression_user_prompt += json.dumps(target_messages, indent=2)
                try:
                    response = openai.ChatCompletion.create(
                        model='gpt-3.5-turbo', #TODO: Ability to configure this
                        messages=[{'role':'system','content':compression_system_prompt},
                                  {'role':'user','content':compression_user_prompt}],
                        temperature=0, #TODO: Ability to configure this
                    )
                    self.convo_summary_text = response.choices[0].message.content
                    self.interaction_messages = self.interaction_messages[target_msg_count_to_summarize:] # Remove old non-system messages from memory
                    print(f"AiBot: Got summary: {self.convo_summary_text}")
                except Exception as e:
                    print("Caught error when doing memory compression.")
                    traceback.print_exc()









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
                    # "kitchen_appliance_color_preference": {
                    #     "type": "string",
                    #     "description": "For kitchen appliances like dishwashers and refrigerators you will ask for the user's color preference before usings this function."
                    # },
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

        if 'matches' not in product_results:
            if 'message' in product_results:
                return AiFunction.ErrorResult(product_results['message'])
            else:
                return AiFunction.ErrorResult(str(product_results))

        vector_db_result = product_results['matches'][0]
        sku = vector_db_result['metadata']['sku']
        dynamodb_info = self.get_dynamo_db_product_details(sku)
        vector_db_result['additional_info'] = dynamodb_info

        return AiFunction.Result(str(vector_db_result), pin_to_memory=True)
    

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
    


    def get_dynamo_db_product_details(self, sku:str):
        # Endpoint URL
        url = "https://retailassist-poc.kore.ai/pimWrapper/v1/getItems"

        # Parameters
        headers = {
            'Stage': 'dev',
            'x-secret-key': 'edc6f2b0-0b7a-11ec-82a8-0242ac130003',
            'Content-Type': 'application/json'
        }
        data = {
            "ids": [sku],
            "PIMEnv": "lg-pim-uat"
        }

        # Make the POST request and parse it
        response = requests.post(url, headers=headers, json=data)
        parsed_response = json.loads(response.text)
        product_info = parsed_response[sku]
        product_info['product_url'] = 'https://www.lg.com/us' + product_info['pdpUrl']
        del product_info['pdpUrl']
        return product_info
    


def run():

    # Maybe initialize the bot
    if not StreamlitAiBot.is_initialized():

        # Setup Streamlit page configs
        page_title = 'LG Chatbot'
        st.set_page_config(
            page_title=page_title,
            page_icon="🤖",
        )

        # Check for a GPT-4 key in ULR parameters
        openai_model = 'gpt-3.5-turbo'
        openai_api_key = None
        query_params = st.experimental_get_query_params()
        if 'gpt4-key' in query_params:
            openai_model = 'gpt-4'
            openai_api_key = query_params['gpt4-key'][0]

        # Initialize the AIBot
        StreamlitAiBot.initialize(streamlit_page_title=page_title,
                            # openai_model='gpt-4',
                            openai_model=openai_model,
                            openai_api_key=openai_api_key,
                            model_temperature=0.1,
                            system_prompt_engineering=open('prompts & content/system prompt.md').read(),
                            welcome_message=open('prompts & content/welcome message.md').read(),
                            ai_functions=[SearchForLGProducts()]
        )

    # Run the AIBot
    StreamlitAiBot.runBot()



if __name__ == "__main__":
    run()
