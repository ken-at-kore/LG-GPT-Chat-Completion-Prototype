import typing
from typing import List
import traceback
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

        # Initialize Conversation Memory
        self.convo_memory = StreamlitAiBot.ConvoMemory(self.system_prompt_engineering, self.welcome_message)

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
        if user_input := st.chat_input("Enter text here"):
            print(f"AiBot: User input: {user_input}")
            user_input = user_input.replace('$','\$') # Try to sanitize against LaTeX markup
            self.convo_memory.add_user_msg(user_input) # Add user input to conversation memory
            st.session_state.messages.append({"role": "user", "content": user_input}) # Add it to the UI thread
            with st.chat_message("user"):
                st.markdown(user_input) # Render it

            # Initialize processing counters
            self.function_error_count = 0
            self.call_and_process_count = 0

            # Call GPT with the input and process results
            self.call_and_process_gpt()

            # Consider compressing conversation memory
            self.convo_memory.maybe_compress_memory()

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
            convo_context = convo_context + [{"role": "function", "name": "do_chain_of_thought_logging", "content": cot_result}]

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
                func_call_results = function_obj.execute(json.loads(function_call_response))
                assert isinstance(func_call_results, AiFunction.Result), f"func_call_results for {function_call_name} must be of type AiFunction.Result, not {type(func_call_results)}"
            except Exception as e:
                print(f"AiBot: Caught function calling error {function_call_name}.\n{e}")
                self.function_error_count += 1
                func_call_results = AiFunction.Result(str(e))
                # raise e

            func_call_results_str = func_call_results.value

            # Store query results for GPT
            print(f"AiBot: Function execution result: {func_call_results_str}")
            self.convo_memory.add_function_msg(function_call_name, func_call_results_str)

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
            self.convo_summary = None # The conversatoin summary that will be appended to the system message content
            self.interaction_messages = [] # The messages excluding the first system message
            if bot_welcome_message is not None:
                self.interaction_messages.append({'role':'assistant', 'content':bot_welcome_message})

        def add_user_msg(self, content:str):
            self.interaction_messages.append({'role':'user', 'content':content})

        def add_assistant_msg(self, content:str, function_call_name:str=None, function_call_arguments:str=None):
            msg = {'role':'assistant', 'content':content}
            if function_call_name is not None or function_call_arguments is not None:
                assert function_call_name is not None and function_call_arguments is not None, "Both function_call_name and function_call_arguments need to be set."
                msg['function_call'] = {'name': function_call_name, 'arguments': function_call_arguments}
            self.interaction_messages.append(msg)

        def add_function_msg(self, function_name:str, content:str):
            self.interaction_messages.append({'role':'function', 'name':function_name, 'content':content})

        def get_messages(self):
            result_system_content = self.system_prompt_engineering
            result_system_content += '\n\nCurrent date: ' + datetime.now().strftime('%Y-%m-%d')
            if self.convo_summary is not None:
                result_system_content += "\n\n---\n\nCONVERSATION SUMMARY\n\n" + self.convo_summary
            return [{'role':'system', 'content':result_system_content}] + self.interaction_messages
        
        def maybe_compress_memory(self):
            #TODO: Move numbers to configuration
            memory_messages_capacity_threshold = 8
            target_msg_count_to_summarize = 4
            if len(self.interaction_messages) > memory_messages_capacity_threshold:
                print("AiBot: Compressing memory...")
                target_messages = self.interaction_messages[:target_msg_count_to_summarize] #TODO: Get the older messages
                compression_system_prompt = 'You are an expert conversation designer and a helpful assistant.'
                compression_user_prompt = 'Summarize the following GPT conversation. Be specific and include facts. ' \
                                            'But you must write less than 150 words.\n\n'
                if self.convo_summary is not None:
                    compression_user_prompt += f'Include this partial summary info in your summary: {self.convo_summary}\n\nNew part of conversation: '
                compression_user_prompt += json.dumps(target_messages, indent=2)
                try:
                    response = openai.ChatCompletion.create(
                        model='gpt-3.5-turbo', #TODO: Ability to configure this
                        messages=[{'role':'system','content':compression_system_prompt},
                                  {'role':'user','content':compression_user_prompt}],
                        temperature=0, #TODO: Ability to configure this
                    )
                    self.convo_summary = response.choices[0].message.content
                    self.interaction_messages = self.interaction_messages[target_msg_count_to_summarize:] # Remove old non-system messages from memory
                    print(f"AiBot: Got summary: {self.convo_summary}")
                except Exception as e:
                    print("Caught error when doing memory compression.")
                    traceback.print_exc()
