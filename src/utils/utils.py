import os
from pinecone import Pinecone
import re
import boto3
from urllib.parse import urlparse
from fastapi import UploadFile
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import logging
from openai import OpenAI
from pathlib import Path
from pydantic import BaseModel, Field
from urllib.parse import quote

ARYN_API_KEY = os.environ["ARYN_API_KEY"]
os.environ["PINECONE_ENVIRONMENT"] = "us-east-1-aws"
#index_name = 'new-index'
index_name = 'guntested-index'
aryn_json_output_dir = 'tmp_aryn_json_dir'
# Initialize Pinecone client
pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment=os.environ.get("PINECONE_ENVIRONMENT")
)
index_logs_file_url = 's3://gunbrain-docs-e/index_logs.json'
upsert_logs_file_url = 's3://gunbrain-docs-e/upsert_logs.json'
# index_logs_file_url = 's3://gunbrain-docs-e/index_file.json'
# upsert_logs_file_url = 's3://gunbrain-docs-e/upsert_file.json'
aryn_cache_dir_path  = os.path.join(os.environ['HOME'], '.cache', 'aryn', 'ingest')


s3 = boto3.client('s3')

def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.hasHandlers():
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)

        logger.addHandler(ch)


setup_logger()

class InitialFieldsExtraction(BaseModel):
    title: str = Field(description="Title of Document")
    authors: list[str] =  Field(description="Author of Document")
    manufacturers: list[str] = Field(description="Manufacturers of Guns if mentioned")
    date_published: str = Field(description="Published Date of Document if mentioned")
    issue_date: str = Field(description="Issue Date of Document if mentioned")
    origin_type: str = Field(description="Source type")

client = OpenAI()

# Your S3 bucket name
BUCKET_NAME = "gunbrain-docs-e"
main_directory_path = "document_main"
processed_docs_folder = "processed"
# main_directory_path = "documents"
# processed_docs_folder = "processed_doc"
processed_docs_folder_s3_url = f"s3://{BUCKET_NAME}/{processed_docs_folder}/"
aryn_ignored_elements = [
    "PageBreak", "Header", "Footer", "CodeSnippet", "PageNumber", "UncategorizedText"]


def split_response_and_citations(answer: str):
    # Capture digits inside square brackets
    citation_pattern = r'\[(\d+(?:,\s*\d+)*)\]'

    citations = re.findall(citation_pattern, answer)

    citations = [list(map(int, citation.split(', ')))
                 for citation in citations]

    response = re.sub(citation_pattern, '', answer).strip()

    return response, citations


def rename_json_file(input_path):
    directory = os.path.dirname(input_path)
    new_name = os.path.basename(input_path).replace(".pdf.json", ".json")
    new_path = os.path.join(directory, new_name)

    os.rename(input_path, new_path)
    return new_path


def parse_s3_url(s3_url):
    """
    Parse an S3 URL into bucket name and folder path.

    :param s3_url: S3 URL (e.g., s3://bucket-name/folder-name/)
    :return: Tuple of bucket name and folder path
    """
    parsed_url = urlparse(s3_url)
    bucket_name = parsed_url.netloc
    folder_path = parsed_url.path.lstrip('/')  # Remove leading slash
    return bucket_name, folder_path


def move_files_s3(source_url, destination_url):
    # Parse the S3 URLs
    source_bucket, source_folder = parse_s3_url(source_url)
    destination_bucket, destination_folder = parse_s3_url(destination_url)

    # List files in the source folder
    response = s3.list_objects_v2(
        Bucket=source_bucket, Prefix=source_folder)
    if 'Contents' not in response:
        logging.info(f"No files found in {source_folder}.")
        return

    for obj in response['Contents']:
        source_key = obj['Key']  # Full key of the object in source folder

        # Skip the folder itself (if exists as a key)
        if source_key == source_folder:
            continue

        # Construct the destination key
        destination_key = source_key.replace(
            source_folder, destination_folder, 1)

        # Copy the file to the destination
        s3.copy_object(
            Bucket=destination_bucket,
            CopySource={'Bucket': source_bucket, 'Key': source_key},
            Key=destination_key
        )

        # Delete the file from the source
        s3.delete_object(Bucket=source_bucket, Key=source_key)

        logging.info(f"Moved: {source_key} -> {destination_key}")


def delete_s3_object(s3_url):
    bucket_name, source_key = parse_s3_url(s3_url)
    s3.delete_object(Bucket=bucket_name, Key=source_key)



def list_files_from_s3_url(s3_url):
    bucket_name, prefix = parse_s3_url(s3_url)

    file_paths = []
    paginator = s3.get_paginator('list_objects_v2')

    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                # Filter out the folder itself
                if obj['Key'] != prefix + '/':
                    file_paths.append(f"s3://{bucket_name}/{obj['Key']}")

    return file_paths


def get_filenames_in_folder_local(folder_path):
    return [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]


def upload_to_s3(file, object_name):
    try:
        # Check if the file is a PDF (by filename)
        if file.filename.lower().endswith('.pdf'):
            extra_args = {
                "ContentType": "application/pdf",
                "ContentDisposition": "inline"
            }
            s3.upload_fileobj(
                file.file,
                BUCKET_NAME,
                object_name,
                ExtraArgs=extra_args
            )
        else:
            s3.upload_fileobj(
                file.file,
                BUCKET_NAME,
                object_name
            )
        s3_file_path = f"s3://{BUCKET_NAME}/{object_name}"
        return s3_file_path
    except Exception as e:
        print(f"An error occurred: {e} here error")
        return None



def download_file_from_s3(s3_url: str, input_dir: str):
    bucket_name, key = parse_s3_url(s3_url)

    os.makedirs(input_dir, exist_ok=True)

    filename = os.path.basename(key)
    local_path = os.path.join(input_dir, filename)

    s3_client = boto3.client('s3')
    s3_client.download_file(bucket_name, key, local_path)

    return local_path


def table_description_extractor(before_table_text, table_content, after_table_text):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": f"""Given the following information, generate a highly detailed, fully descriptive, and table description in the form of first single paragraph that includes:
            
            - Every bit of information present in the table
            - All symbols, units, footnotes, and special formatting
            - Key numerical values, data trends, and relationships
            - Column and row headers with precise labels
            - Any equations, abbreviations, or annotations present
            - Structural and contextual importance of the table
            - Any relevant metadata that enhances understanding
            
            Ensure that the description preserves the full meaning of the table without omitting any details.
            Text Before Table:
            {before_table_text}
            
            Table:
            {table_content}
            
            Text After Table:
            {after_table_text}
"""
             }
        ]
    )

    return completion.choices[0].message.content


def extract_initials(initial_content):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": """You are an expert in structured data extraction and analysis. You will be provided with text content from a book or magazine related to firearms.
             Your task is to extract specific fields of information from this content in a structured format. Additionally, analyze and extract the origin type (e.g., book, magazine).
             For each provided text, identify, extract, and analyze the following fields as accurately as possible, ensuring that all information is clear, complete, and follows the given structure"""},
            {"role": "user", "content": f"{initial_content}"}
        ],
        response_format=InitialFieldsExtraction,
    )

    response = completion.choices[0].message.parsed

    return response


def get_file_object(local_file_path: str) -> UploadFile:
    """
    Creates an UploadFile-like object from a local file path.

    Args:
        local_file_path (str): The path to the local file.

    Returns:
        UploadFile: An UploadFile-like object for the file.
    """
    file_path = Path(local_file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"The file {local_file_path} does not exist.")

    # Open the file as a binary stream
    file_stream = file_path.open("rb")
    return UploadFile(filename=file_path.name, file=file_stream)

def is_s3_folder(s3_url):
    # Define common file extensions
    file_extensions = {'.txt', '.pdf', '.csv', '.jpg', '.png', '.json', '.zip',
                       '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.mp4',
                       '.mp3', '.wav', '.py', '.html', '.js', '.css'}

    # Validate and parse S3 URL
    match = re.match(r's3://([^/]+)/(.+)', s3_url)
    if not match:
        raise ValueError("Invalid S3 URL format. Expected 's3://bucket-name/key'.")

    key = match.group(2)

    # Check if the key ends with a '/'
    if key.endswith('/'):
        return True

    # Check if the key ends with a known file extension
    if any(key.lower().endswith(ext) for ext in file_extensions):
        return False

    # Default case: assume it's a folder if no extension matches
    return True


def get_filename_from_s3_url(s3_url):
    # Parse the URL
    parsed_url = urlparse(s3_url)
    # Get the path and extract the filename
    filename = parsed_url.path.split('/')[-1]
    return filename


def delete_files_in_directory(path):
    # Check if the path exists
    if os.path.exists(path):
        # Loop through all files and subdirectories in the specified path
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))  # Remove files
            for name in dirs:
                os.rmdir(os.path.join(root, name))  # Remove empty directories
        logging.info(f"All files and subfolders in {path} have been deleted.")
    else:
        logging.info(f"The specified path {path} does not exist.")

def format_chat_history(messages) -> str:
    return "\n".join(f"{msg.role}: {msg.content}" for msg in messages)

def convert_s3_to_http_url(s3_url: str) -> str:
    if not s3_url.startswith("s3://"):
        return s3_url  
    # Strip off "s3://"
    s3_stripped = s3_url.replace("s3://", "", 1)
    parts = s3_stripped.split("/", 1)  # split into bucket, the rest of the key
    bucket = parts[0]
    key = ""
    if len(parts) > 1:
        key = parts[1]

    encoded_key = quote(key, safe="")
    region = "us-east-1"  
    return f"https://{bucket}.s3.{region}.amazonaws.com/{encoded_key}"
