# import boto3
# import json
# import os
# from dotenv import load_dotenv

# load_dotenv()

# # 👇 Replace this with your actual profile ARN
# inference_profile_arn = "arn:aws:bedrock:us-west-2:240647218770:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0"

# bedrock = boto3.client(
#     "bedrock-runtime",
#     region_name="us-west-2",
#     aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
#     aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
#     aws_session_token=os.getenv("AWS_SESSION_TOKEN")
# )

# body = {
#     "anthropic_version": "bedrock-2023-05-31",
#     "max_tokens": 200,
#     "top_k": 250,
#     "stop_sequences": [],
#     "temperature": 1,
#     "top_p": 0.999,
#     "messages": [
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": "What is the capital of France?"
#                 }
#             ]
#         }
#     ]
# }

# response = bedrock.invoke_model(
#     modelId=inference_profile_arn,  # 🔥 Use profile ARN as modelId
#     contentType="application/json",
#     accept="application/json",
#     body=json.dumps(body)
# )

# response_body = json.loads(response["body"].read())
# print(response_body["content"][0]["text"])


import boto3
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Replace with your actual inference profile ARN
inference_profile_arn = "arn:aws:bedrock:us-west-2:240647218770:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0"

# Validate required environment variables
required_vars = ["AWS_DEFAULT_REGION", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
for var in required_vars:
    if not os.getenv(var):
        raise ValueError(f"Missing required environment variable: {var}")

class ClaudeConversation:
    def __init__(self):
        self.bedrock = boto3.client(
            "bedrock-runtime",
            region_name=os.getenv("AWS_DEFAULT_REGION"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=os.getenv("AWS_SESSION_TOKEN")
        )
        self.conversation_history = []
    
    def ask(self, question, max_tokens=1000, temperature=0.7):
        """Ask a question and maintain conversation history"""
        
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": [{"type": "text", "text": question}]
        })
        
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "top_k": 250,
            "stop_sequences": [],
            "temperature": temperature,
            "top_p": 0.999,
            "messages": self.conversation_history
        }

        try:
            response = self.bedrock.invoke_model(
                modelId=inference_profile_arn,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body)
            )
            
            response_body = json.loads(response["body"].read())
            assistant_response = response_body["content"][0]["text"]
            
            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_response}]
            })
            
            return assistant_response
            
        except Exception as e:
            return f"Error: {e}"
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []

# Example usage:
if __name__ == "__main__":
    chat = ClaudeConversation()
    
    print("Claude Chat with History (type 'quit' to exit, 'clear' to clear history)")
    
    while True:
        question = input("\nYou: ")
        
        if question.lower() in ['quit', 'exit', 'q']:
            break
        elif question.lower() == 'clear':
            chat.clear_history()
            print("Conversation history cleared.")
            continue
        
        response = chat.ask(question)
        print(f"Claude: {response}")