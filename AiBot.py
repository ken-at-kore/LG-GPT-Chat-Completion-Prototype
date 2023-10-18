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
        self.streamlit_page_title = kwargs.get('streamlit_page_title', "My AiBot")
        self.openai_model = kwargs.get('openai_model', "gpt-3.5-turbo")
        self.model_temperature = kwargs.get('model_temperature', 0.25)
        self.show_function_activity = kwargs.get('show_function_activity', False)
        self.system_prompt_engineering = kwargs.get('system_prompt_engineering', 'You are a helpful assistant.')
        self.welcome_message = kwargs.get('welcome_message', 'Hello! How can I assist you today?')

        ai_funcs_arg = kwargs.get('ai_functions', None)
        self.ai_functions = AiFunction.Collection(ai_funcs_arg) if ai_funcs_arg is not None else AiFunction.Collection()



    @staticmethod
    def initialize(streamlit_page_title:str="My AiBot",
                 openai_model:str="gpt-3.5-turbo",
                 model_temperature:float=0.25,
                 show_function_activity:bool=False,
                 system_prompt_engineering:str='You are a helpful assistant.',
                 welcome_message:str='Hello! How can I assist you today?',
                 ai_functions:List[AiFunction]=None
    ):
        bot = StreamlitAiBot(streamlit_page_title=streamlit_page_title,
                             openai_model=openai_model,
                             model_temperature=model_temperature,
                             show_function_activity=show_function_activity,
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
        bot = st.session_state['ai_bot']
        assert bot is not None, "StreamlitAiBot has not been initialized"
        assert isinstance(bot, StreamlitAiBot), "Streamlit session ai_bot is not of type StreamlitAiBot"
        bot.runInstance()



    def call_and_process_gpt(self):
        """
        Use the chat history in st.session_state['gpt_messages'] to call ChatGPT.
        Process the bot message and function call results.
        """

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
                'messages': st.session_state["gpt_messages"],
                'stream': True,
                'temperature': self.model_temperature
            }
            if not self.ai_functions.is_empty():
                chat_completion_params['functions'] = self.ai_functions.get_function_specs()

            # Call OpenAI GPT (Response is streamed)
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
                    if self.show_function_activity:
                        if function_call_response == "":
                            full_response += "\n\n`Query: "
                        full_response += call.get("arguments", "")
                        message_placeholder.markdown(full_response + "`▌")

            if not function_call_response:
                message_placeholder.markdown(full_response)
            elif not self.show_function_activity: 
                message_placeholder.markdown('Just a sec 🔍')
            else:
                message_placeholder.markdown(full_response + "`" if function_call_response else "")

        # Handle no function call
        if not function_call_response:
            # Store the bot content
            st.session_state['gpt_messages'].append({"role": "assistant", "content": bot_content_response})
            st.session_state.messages.append({"role": "assistant", "content": bot_content_response})

        # Handle function call
        else:
            # Store bot content including function call name and arguments
            st.session_state['gpt_messages'].append({"role": "assistant", "content": bot_content_response,
                                                        "function_call": {"name": function_call_name, 
                                                                        "arguments": function_call_response}})
            if self.show_function_activity:
                st.session_state.messages.append({"role": "assistant", "content": full_response + "`"})
            elif bot_content_response != "":
                st.session_state.messages.append({"role": "assistant", "content": bot_content_response})

            # Execute function call
            function_obj = self.ai_functions.get_function(function_call_name)
            assert function_obj is not None, f'Function {function_call_name} is not defined in the function collection'
            try:
                func_call_results = function_obj.execute(json.loads(function_call_response))
                assert isinstance(func_call_results, AiFunction.Result), f"func_call_results for {function_call_name} must be of type AiFunction.Result, not {type(func_call_results)}"
            except Exception as e:
                # print(e)
                # func_call_results = AiFunction.Result(str(e))
                raise e

            func_call_results_str = func_call_results.value

            # Store query results for GPT
            st.session_state['gpt_messages'].append({"role": "function", "name": "query_lg_dishwasher_products", 
                                                        "content": func_call_results_str})
            
            # Render query results
            if self.show_function_activity:
                with st.chat_message("query result"):
                    st.markdown(func_call_results_str)
                    st.session_state.messages.append({"role": "query result", "content": func_call_results_str})


            # Recursively call this same function to process the query results
            self.call_and_process_gpt()




    def runInstance(self):

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
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state["gpt_messages"].append({"role": "user", "content": prompt.replace('$','\$')})
            with st.chat_message("user"):
                st.markdown(prompt.replace('$','\$')) # Try to avoid LaTeX markup

            # Call GPT with the input and process results
            self.call_and_process_gpt()

    
