import anthropic

# Initialize the Anthropogenic client with your API key
client = anthropic.Anthropic(api_key="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")

# Take the prompt as user input
prompt = input("Enter the prompt: ")

# Create a message using the model, parameters, and user message
message = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    temperature=0.0,
    system="",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

# Print the generated content
print(message.content)
