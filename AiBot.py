import typing
from typing import List
import streamlit as st
from datetime import datetime
import openai
import json



class AiFunction:
    def __init__(self):
        assert hasattr(self.__class__, 'get_spec'), f'{self.__class__.__name__} must define a get_spec() method'
        assert hasattr(self.__class__, 'execute'), f'{self.__class__.__name__} must define an execute(args) method'

    class Result:
        def __init__(self, val):
            self.value = val

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
    
    def __init__(self, **kwargs):
        # Initialize fields
        self.streamlit_page_title = kwargs.get('streamlit_page_title')
        self.openai_model = kwargs.get('openai_model')
        self.model_temperature = kwargs.get('model_temperature')
        self.system_prompt_engineering = kwargs.get('system_prompt_engineering')
        self.welcome_message = kwargs.get('welcome_message')

        # Initialize AiFunctions
        ai_funcs_arg = kwargs.get('ai_functions', None)
        self.ai_functions = AiFunction.Collection(ai_funcs_arg) if ai_funcs_arg is not None else AiFunction.Collection()

        # Initialize chain of thought config
        self.do_cot = True



    @staticmethod
    def initialize(streamlit_page_title:str="My AiBot",
                 openai_model:str="gpt-3.5-turbo",
                 model_temperature:float=0.25,
                 system_prompt_engineering:str='You are a helpful assistant.',
                 welcome_message:str='Hello! How can I assist you today?',
                 ai_functions:List[AiFunction]=None
    ):
        print("AiBot: Initializing session.")
        bot = StreamlitAiBot(streamlit_page_title=streamlit_page_title,
                             openai_model=openai_model,
                             model_temperature=model_temperature,
                             system_prompt_engineering=system_prompt_engineering,
                             welcome_message=welcome_message,
                             ai_functions=ai_functions
        )
        st.session_state['ai_bot'] = bot


    
    @staticmethod
    def is_initialized():
        return 'ai_bot' in st.session_state



    @staticmethod
    def run():
        """
        Get the AiBot from the Streamlit session and run it.
        """
        bot = st.session_state['ai_bot']
        assert bot is not None, "StreamlitAiBot has not been initialized"
        assert isinstance(bot, StreamlitAiBot), "Streamlit session ai_bot is not of type StreamlitAiBot"
        bot.runInstance()



    def runInstance(self):
        """
        Run AiBot's main loop. The bot takes a turn.
        """

        print("AiBot: Running.")

        # TODO: Move initializations to constructor
        # Set Streamlit app meta info
        st.set_page_config(
            page_title=self.streamlit_page_title,
            page_icon="🤖",
        )
        st.title(self.streamlit_page_title)

        # Set the OpenAI key and model
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        if "openai_model" not in st.session_state:
            st.session_state["openai_model"] = self.openai_model

        # Set up the GPT system prompt with the current date
        system_promt = self.system_prompt_engineering + '\n\nCurrent date: ' + datetime.now().strftime('%Y-%m-%d')

        # Initialize UI messages and GPT messages
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": self.welcome_message}
            ]
            st.session_state["gpt_messages"] = [
                {"role": "system", "content": system_promt},
                {"role": "assistant", "content": self.welcome_message}
            ]

        # Re-render UI messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Get, store and render user message
        if prompt := st.chat_input("Enter text here"):
            print(f"AiBot: User input: {prompt}")
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state["gpt_messages"].append({"role": "user", "content": prompt.replace('$','\$')})
            with st.chat_message("user"):
                st.markdown(prompt.replace('$','\$')) # Try to avoid LaTeX markup

            # Initialize processing counters
            self.function_error_count = 0
            self.call_and_process_count = 0

            # Call GPT with the input and process results
            self.call_and_process_gpt()

        else:
            print("AiBot: Didn't process input.")


    def call_and_process_gpt(self):
        """
        Use the chat history in st.session_state['gpt_messages'] to call ChatGPT.
        Process the bot message and function call results.
        """

        # Keep track of how often call & processing happens this turn
        self.call_and_process_count = self.call_and_process_count + 1

        # Prepare GPT messages (conversation context). Include chain of thought logging for improved decision making.
        convo_context = st.session_state["gpt_messages"]
        if self.do_cot:
            cot_str = self.do_chain_of_thought_logging()
            convo_context = convo_context + [{"role": "function", "name": "do_chain_of_thought_logging", "content": cot_str}]

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

            # Add function calls
            # TODO: Refactor magic numbers below as configurations
            if not self.ai_functions.is_empty() and \
                    self.function_error_count < 1 and \
                    self.call_and_process_count < 4:
                chat_completion_params['functions'] = self.ai_functions.get_function_specs()

            # Call OpenAI GPT (Response is streamed)
            print("AiBot: Calling GPT for chat completion...")
            for response in openai.ChatCompletion.create(**chat_completion_params):

                # Handle content stream
                if not response.choices[0].delta.get("function_call",""):
                    content_chunk = response.choices[0].delta.get("content", "")
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
            st.session_state['gpt_messages'].append({"role": "assistant", "content": bot_content_response})
            st.session_state.messages.append({"role": "assistant", "content": bot_content_response})

        # Handle function call
        else:
            print(f'AiBot: Got GPT function call "{function_call_name}". Content: "{bot_content_response}". Arguments: {function_call_response}')
            # Store bot content including function call name and arguments
            st.session_state['gpt_messages'].append({"role": "assistant", "content": bot_content_response,
                                                        "function_call": {"name": function_call_name, 
                                                                        "arguments": function_call_response}})
            if bot_content_response != "":
                st.session_state.messages.append({"role": "assistant", "content": bot_content_response})

            # Execute function call
            function_obj = self.ai_functions.get_function(function_call_name)
            assert function_obj is not None, f'Function {function_call_name} is not defined in the function collection'
            try:
                func_call_results = function_obj.execute(json.loads(function_call_response))
                assert isinstance(func_call_results, AiFunction.Result), f"func_call_results for {function_call_name} must be of type AiFunction.Result, not {type(func_call_results)}"
            except Exception as e:
                print(f"AiBot: Caught function calling error {function_call_name}.\n{e}")
                self.function_error_count = self.function_error_count + 1
                func_call_results = AiFunction.Result(str(e))
                # raise e

            func_call_results_str = func_call_results.value

            # Store query results for GPT
            print(f"AiBot: Function execution result: {func_call_results_str}")
            st.session_state['gpt_messages'].append({"role": "function", "name": function_call_name, 
                                                     "content": func_call_results_str})

            # Recursively call this same function to process the query results
            self.call_and_process_gpt()



    def do_chain_of_thought_logging(self):

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
        funcs = self.ai_functions.get_function_specs() + [cot_func_spec]
        print("AiBot: Calling GPT for chain of thought...")
        response = openai.ChatCompletion.create(
            model=self.openai_model,
            messages=st.session_state["gpt_messages"],
            temperature=self.model_temperature,
            functions=funcs,
            function_call={"name": "do_chain_of_thought_logging"}
        )
        response = json.loads(response.choices[0].message.function_call.arguments)
        cot_result = f"{response['describe_the_context']} | {response['describe_what_you_should_do_next']}"
        print(f"AiBot: Chain of thought: {cot_result}")
        return cot_result
        

