'''
Description:  
Author: Huang J
Date: 2025-03-31 11:13:16
'''

import logging
from typing import Optional
from openai import OpenAI
from google.genai import types
from google import genai


logging.basicConfig(level=logging.INFO,format='%(asctime)s-%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)

def llm_response_api(
    user_prompt: str,  
    model_type: str,   
    model_name: str,   
    api_key: str,      
    base_url: Optional[str] = None, 
    system_prompt: Optional[str] = None,  
    limit_api: int = 5  
) -> str:  
    """
    Sends a request to a language model (ChatGPT or Gmini) and returns the generated response.

    This function handles communication with two types of models: 'ChatGPT' and 'Gmini'. 
    It sends the provided `user_prompt` to the selected model and returns the response from the model.
    Additionally, it handles errors and retries up to `limit_api` times if the API call fails.

    Args:
        user_prompt (str): The prompt text provided by the user to generate a response.
        model_type (str): The type of model to use ('ChatGPT' or 'Gmini').
        model_name (str): The specific model name (e.g., "gpt-3.5-turbo").
        api_key (str): The API key used to authenticate with the language model service.
        base_url (Optional[str]): An optional base URL for the API service. Default is None.
        system_prompt (Optional[str]): An optional system-level instruction to guide the model's behavior. Default is None.
        limit_api (int): The maximum number of times the API should retry on failure. Default is 5.

    Returns:
        str: The generated response from the model.

    Raises:
        ValueError: If the API call fails repeatedly (as per `limit_api`).
        ValueError: If the `model_type` is invalid.

    """

    api_count = 0
    match model_type:
        case 'ChatGPT':
            if base_url:
                client = OpenAI(api_key=api_key, base_url=base_url)
            else:
                client = OpenAI(api_key=api_key)
                
            if system_prompt:
                while True:
                    try:
                        response = client.chat.completions.create(
                                model=model_name,
                                messages=[
                                        {
                                            "role": "system",
                                            "content": [
                                                {
                                                    "type": "text",
                                                    "text": system_prompt
                                                }
                                            ]
                                        },
                                        {
                                            "role": "user",
                                            "content": [
                                                {
                                                    "type": "text",
                                                    "text": user_prompt
                                                }
                                            ]
                                        }
                                    ],
                                temperature=0.6
                            )
                        break
                    except Exception as e:
                        api_count+=1
                        logger.info(f'{model_type} model API call response error.Error message:{e}')
                    if api_count==limit_api:
                        raise ValueError(f'{model_type} model API call failed {api_count} times consecutively. Please check the API status.')
            else:
                while True:
                    try:
                        response = client.chat.completions.create(
                                model=model_name,
                                messages=[
                                        {
                                            "role": "user",
                                            "content": [
                                                {
                                                    "type": "text",
                                                    "text": user_prompt
                                                }
                                            ]
                                        }
                                    ],
                                temperature=0.6
                            )
                        break
                    except Exception as e:
                        api_count+=1
                        logger.info(f'{model_type} model API call response error.Error message:{e}')
                    if api_count==limit_api:
                        raise ValueError(f'{model_type} model API call failed {api_count} times consecutively. Please check the API status.')
            content = response.choices[0].message.content
            return content
        case 'Gmini':
            client = genai.Client(api_key=api_key)
            safety_settings = [
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                        # method=types.HarmBlockMethod.PROBABILITY
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                        # method=types.HarmBlockMethod.PROBABILITY
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                        # method=types.HarmBlockMethod.PROBABILITY
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                        # method=types.HarmBlockMethod.PROBABILITY
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                        threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                        # method=types.HarmBlockMethod.PROBABILITY
                    )
                ]
            if system_prompt:
                while True:
                    try:
                        response = client.models.generate_content(
                                model=model_name,
                                contents=user_prompt,
                                config=types.GenerateContentConfig(
                                        system_instruction=system_prompt,
                                        safety_settings=safety_settings
                                    )
                            )
                        break
                    except Exception as e:
                        api_count+=1
                        logger.info(f'{model_type} model API call response error.Error message:{e}')
                    if api_count==limit_api:
                        raise ValueError(f'{model_type} model API call failed {api_count} times consecutively. Please check the API status.')
            else:
                while True:
                    try:
                        response = client.models.generate_content(
                                model=model_name,
                                contents=user_prompt,
                                config=types.GenerateContentConfig(
                                        safety_settings=safety_settings
                                    )
                            )
                        break
                    except Exception as e:
                        api_count+=1
                        logger.info(f'{model_type} model API call response error.Error message:{e}')
                    if api_count==limit_api:
                        raise ValueError(f'{model_type} model API call failed {api_count} times consecutively. Please check the API status.')
            content = response.text
            return content
        case _:
            logger.info(f'model_type: {model_type}')
            raise ValueError("Please check the 'model_type' parameter. It can only take values {cgatgpt, gemini}.")
