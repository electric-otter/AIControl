from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
import subprocess

# Initialize the AI model and tokenizer
model_name = 'microsoft/DialoGPT-medium'  # Example model, replace with the specific Microsoft model if different
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def generate_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def execute_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout.decode('utf-8') if result.stdout else "Command executed successfully."
    except subprocess.CalledProcessError as e:
        return f"Error executing command: {e.stderr.decode('utf-8')}"

def main():
    print("Welcome to AI PC Control!")
    while True:
        user_input = input("Enter a command: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting...")
            break

        # Generate the AI response
        ai_prompt = f"Execute the following command on my PC: {user_input}"
        ai_response = generate_response(ai_prompt)

        # Execute the command
        output = execute_command(ai_response)
        print(output)

if __name__ == "__main__":
    main()
