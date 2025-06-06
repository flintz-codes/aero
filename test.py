import boto3

bedrock = boto3.client("bedrock", region_name="us-west-2")
response = bedrock.list_inference_profiles()

print(response)  # or however the response returns available models
