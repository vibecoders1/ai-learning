import os
import sys
import json
import httpx
import asana
from typing import Optional, Dict, List, Any
from asana.rest import ApiException
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize OpenAI client with error handling
try:
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    client = OpenAI(api_key=api_key)
    model = os.getenv('OPENAI_MODEL', 'gpt-4')

    # Validate model access
    test_response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "test"}],
        max_tokens=5
    )
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    print("Please check your OPENAI_API_KEY and OPENAI_MODEL environment variables.")
    sys.exit(1)

# Initialize Asana client with error handling
try:
    asana_token = os.getenv('ASANA_ACCESS_TOKEN')
    if not asana_token:
        raise ValueError("ASANA_ACCESS_TOKEN environment variable is not set")

    configuration = asana.Configuration()
    configuration.access_token = asana_token
    api_client = asana.ApiClient(configuration)
    tasks_api_instance = asana.TasksApi(api_client)
except Exception as e:
    print(f"Error initializing Asana client: {e}")
    print("Please check your ASANA_ACCESS_TOKEN environment variable.")
    sys.exit(1)

def create_asana_task(task_name: str, due_on: str = "today") -> str:
    """
    Creates a task in Asana given the name of the task and when it is due.

    Args:
        task_name (str): The name of the task in Asana
        due_on (str): The date the task is due in the format YYYY-MM-DD. If not given, the current day is used

    Returns:
        str: The API response of adding the task to Asana or an error message if the API call threw an error
    """
    if not task_name or not task_name.strip():
        raise ValueError("Task name cannot be empty")

    if due_on == "today":
        due_on = str(datetime.now().date())
    else:
        try:
            datetime.strptime(due_on, '%Y-%m-%d')
        except ValueError:
            raise ValueError("due_on must be in YYYY-MM-DD format")

    project_id = os.getenv("ASANA_PROJECT_ID")
    if not project_id:
        raise ValueError("ASANA_PROJECT_ID environment variable is not set")

    task_body = {
        "data": {
            "name": task_name,
            "due_on": due_on,
            "projects": [project_id]
        }
    }

    try:
        api_response = tasks_api_instance.create_task(task_body, {})
        return json.dumps(api_response, indent=2)
    except ApiException as e:
        return f"Exception when calling TasksApi->create_task: {e}"

def get_tools() -> List[Dict[str, Any]]:
    """
    Returns the list of available tools for the AI to use.

    Returns:
        List[Dict[str, Any]]: List of tool configurations
    """
    return [{
        "type": "function",
        "function": {
            "name": "create_asana_task",
            "description": "Creates a task in Asana given the name of the task and when it is due",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_name": {
                        "type": "string",
                        "description": "The name of the task in Asana"
                    },
                    "due_on": {
                        "type": "string",
                        "description": "The date the task is due in the format YYYY-MM-DD. If not given, the current day is used"
                    },
                },
                "required": ["task_name"]
            },
        },
    }]

def prompt_ai(messages: List[Dict[str, str]]) -> str:
    """
    Sends a prompt to the AI and handles any tool calls.

    Args:
        messages (List[Dict[str, str]]): The conversation history

    Returns:
        str: The AI's response
    """
    try:
        # First, prompt the AI with the latest user message
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=get_tools()
        )

        if not completion or not completion.choices:
            raise ValueError("No response received from OpenAI API")

        response_message = completion.choices[0].message
        tool_calls = response_message.tool_calls

        # Handle tool calls if any
        if tool_calls:
            available_functions = {
                "create_asana_task": create_asana_task
            }

            messages.append(response_message)

            for tool_call in tool_calls:
                try:
                    function_name = tool_call.function.name
                    if function_name not in available_functions:
                        raise ValueError(f"Unknown function: {function_name}")

                    function_to_call = available_functions[function_name]
                    function_args = json.loads(tool_call.function.arguments)
                    function_response = function_to_call(**function_args)

                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response
                    })
                except json.JSONDecodeError as e:
                    print(f"Error parsing function arguments: {e}")
                    return "I encountered an error processing the function arguments."
                except Exception as e:
                    print(f"Error executing function {function_name}: {e}")
                    return "I encountered an error while trying to execute the requested action."

            try:
                second_response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
                return second_response.choices[0].message.content
            except Exception as e:
                print(f"Error in second API call: {e}")
                return "I encountered an error while processing the response."

        return response_message.content

    except Exception as e:
        print(f"Error in OpenAI API call: {str(e)}")
        return "I encountered an error while processing your request. Please try again."

def main() -> None:
    """Main function to run the chat loop"""
    messages = [{
        "role": "system",
        "content": f"You are a personal assistant who helps manage tasks in Asana. The current date is: {datetime.now().date()}"
    }]

    print("Chat initialized. Type 'q' to quit.")

    while True:
        try:
            user_input = input("\nChat with AI (q to quit): ").strip()

            if user_input.lower() == 'q':
                print("Goodbye!")
                break

            if not user_input:
                print("Please enter a message.")
                continue

            messages.append({"role": "user", "content": user_input})
            ai_response = prompt_ai(messages)

            print(f"\nAI: {ai_response}")
            messages.append({"role": "assistant", "content": ai_response})

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Please try again.")

if __name__ == "__main__":
    main()