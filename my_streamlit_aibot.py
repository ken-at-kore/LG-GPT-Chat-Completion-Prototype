from aibot import StreamlitAiBot
from ai_functions.search_for_lg_products import SearchForLGProducts



def run():
    if not StreamlitAiBot.is_initialized():
        StreamlitAiBot.initialize(streamlit_page_title='LG AiBot Prototype',
                            openai_model='gpt-3.5-turbo',
                            model_temperature=0.25,
                            show_function_activity=False,
                            system_prompt_engineering=open('prompts & content/welcome message.md').read(),
                            welcome_message=open('prompts & content/welcome message.md').read(),
                            ai_functions=[SearchForLGProducts()]
        )
    StreamlitAiBot.run()

    

if __name__ == "__main__":
    run()
