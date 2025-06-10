import boto3
import os
import random
import json


def initialize_s3_client(region='us-west-2'):
    """Initializes and returns an S3 client."""
    return boto3.client(
        's3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
        region_name=region
    )


def download_all_files_from_folder(s3_client, bucket_name, prefix, local_dir='downloaded_docs'):
    """Downloads all files from a specific folder in an S3 bucket."""
    os.makedirs(local_dir, exist_ok=True)

    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    for page in pages:
        for obj in page.get('Contents', []):
            key = obj['Key']
            file_name = os.path.basename(key)
            if file_name:
                local_path = os.path.join(local_dir, file_name)
                print(f"Downloading: {key} → {local_path}")
                s3_client.download_file(bucket_name, key, local_path)


def download_random_sample_files(s3_client, bucket_name, sample_size=100, local_dir='downloaded_docs'):
    """Downloads a random sample of files from an S3 bucket."""
    os.makedirs(local_dir, exist_ok=True)

    all_keys = []
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name)

    for page in pages:
        for obj in page.get('Contents', []):
            key = obj['Key']
            if not key.endswith('/'):
                all_keys.append(key)

    sampled_keys = random.sample(all_keys, min(sample_size, len(all_keys)))

    for key in sampled_keys:
        file_name = os.path.basename(key)
        local_path = os.path.join(local_dir, file_name)
        print(f"Downloading: {key} → {local_path}")
        s3_client.download_file(bucket_name, key, local_path)


def extract_text_from_ocr_files(source_folder='downloaded_docs', destination_folder='structured_output_folder'):
    """Processes OCR output files in JSON format and saves extracted LINE text."""
    os.makedirs(destination_folder, exist_ok=True)

    for filename in os.listdir(source_folder):
        if filename.endswith('.txt'):
            source_path = os.path.join(source_folder, filename)
            dest_path = os.path.join(destination_folder, filename)

            with open(source_path, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"❌ Skipping {filename}: Not valid JSON.")
                    continue

                lines = [block['Text'] for block in data.get('Blocks', []) if block.get('BlockType') == 'LINE']

                with open(dest_path, 'w') as out_file:
                    out_file.write('\n'.join(lines))

            print(f"✅ Processed and saved: {filename}")

    print("\n🎯 All OCR files have been converted and saved in the output folder.")

# if __name__ == "__main__":
#     s3 = initialize_s3_client()
#     bucket = 'aerocons-test-bedrock'
#     folder_prefix = 'Demo Airline-MSN 4321/Records Hub/'

#     # Download all files from a folder
#     download_all_files_from_folder(s3, bucket, folder_prefix)

#     # Or download a random sample of 100 files
#     # download_random_sample_files(s3, bucket, sample_size=100)

#     # Process OCR files
#     extract_text_from_ocr_files()
