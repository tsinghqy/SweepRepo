import os
import anthropic

# Retrieve the API key from environment variables
api_key = ''

# Check if the API key is provided
if api_key is None:
    raise ValueError("API key is not provided. Please set the ANTHROPIC_API_KEY environment variable.")

# Initialize the Anthropogenic client with your API key
client = anthropic.Anthropic(api_key=api_key)

def modify_prompt(prompt):
    """
    Modify the prompt by inserting it after the provided text.

    Parameters:
    - prompt (str): The prompt to be inserted.

    Returns:
    - str: The modified prompt.
    """
    provided_text = ("You are a brilliant and meticulous engineer assigned to write code for the following Github issue. "
                     "When you write code, the code works on the first try, is syntactically perfect and is fully "
                     "implemented. You have the utmost care for the code that you write, so you do not make mistakes "
                     "and every function and class will be fully implemented. When writing tests, you will ensure the "
                     "tests are fully implemented, very extensive and cover all cases, and you will make up test data "
                     "as needed. Take into account the current repository's language, frameworks, and dependencies. "
                     "After you are done, I want you to generate a new file with the code changes you created.")
    return f"{provided_text} {prompt}"

# Take the prompt as user input
user_prompt = input("Enter the prompt: ")

# Modify the prompt
modified_prompt = modify_prompt(user_prompt)

# Create a message using the model, parameters, and user message
message = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    temperature=0.0,
    system="",
    messages=[
        {"role": "user", "content": modified_prompt}
    ]
)
generated_code = message.content
# Check if the message was successfully created
if isinstance(generated_code, anthropic.TextBlock):
    generated_code = generated_code.to_text()
    # Write the generated code to a new file
output_file_path = "output_file.py"
with open(output_file_path, "w") as file:
    file.write(generated_code)
print(f"Generated code has been saved to '{output_file_path}'")