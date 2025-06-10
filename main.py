import boto3
import json
import os
import re
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
import time
import botocore.exceptions

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global configuration
AWS_REGION = "us-west-2"
MODEL_ID = "arn:aws:bedrock:us-west-2:240647218770:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0"



def initialize_bedrock_client(aws_region: str = AWS_REGION):
    """Initialize AWS Bedrock client"""
    try:
        bedrock_client = boto3.client(
            service_name='bedrock-runtime',
            region_name = aws_region,
            aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_session_token = os.getenv("AWS_SESSION_TOKEN")
        )
        logger.info("AWS Bedrock client initialized successfully")
        return bedrock_client
    except Exception as e:
        logger.error(f"Failed to initialize AWS Bedrock client: {str(e)}")
        raise


def read_text_file(file_path: str) -> str:
    """
    Read content from a text file
    
    Args:
        file_path: Path to the text file
        
    Returns:
        Content of the file as string
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
        logger.info(f"Successfully read file: {file_path}")
        return content
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        return ""


def extract_ad_numbers_with_claude(document_content: str, bedrock_client) -> List[str]:
    """
    Use Claude Sonnet via AWS Bedrock to extract AD number from document content
    Implements retry logic with exponential backoff on throttling or transient errors.
    Args:
        document_content: Content of the document
        bedrock_client: AWS Bedrock client instance
        
    Returns:
        Extracted AD numbers
    """
    prompt = f"""
    You are an aviation maintenance document analyzer. Please analyze the following document content and extract **all Airworthiness Directive (AD) numbers** from the document below.

    AD numbers can appear in various formats including:
    - EASA format: YYYY-NNNN (e.g., 2011-0015)
    - FAA format: YYYY-NN-NN (e.g., 2002-08-52)
    - Other formats: XX-XX-XX (e.g., 74-08-09)

   They may be introduced by phrases like:
    - "AD No.", "EASA AD No", "AIRWORTHINESS DIRECTIVE", "Supersedes AD", "Amendment", or inside parentheses.

    Respond with each AD number on a new line. One per line, no explanations.

    Example:
    2011-0015  
    2002-08-52  
    74-08-09  

    Document Content:
    {document_content[:3000]}...
    """
    # Prepare the request body for Claude
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 200,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    })
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try: 
            # Invoke Claude via Bedrock
            response = bedrock_client.invoke_model(
                body = body,
                modelId = MODEL_ID,
                accept="application/json",
                contentType="application/json"
            )
            
            # Parse response
            response_body = json.loads(response.get('body').read())
            ad_numbers = [line.strip() for line in response_body['content'][0]['text'].splitlines()]
            ad_numbers = [ad for ad in ad_numbers if is_valid_ad_number(ad)]

            logger.info(f"Claude extracted AD numbers: {ad_numbers}")
            return ad_numbers
            
        except botocore.exceptions.ClientError as e:
            error_msg = str(e)
            if "ThrottlingException" in error_msg or "Rate exceeded" in error_msg:
                wait_time = 2 ** attempt
                logger.warning(f"Claude throttled (attempt {attempt}/{max_attempts}). Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"Claude ClientError: {error_msg}")
                break
        except Exception as e:
            logger.error(f"Claude extraction failed: {str(e)}")
            break
    return []


def fallback_ad_extraction(document_content: str) -> List[str]:
    """
    Fallback method to extract AD number using regex patterns
    Improved to avoid false matches with dates
    
    Args:
        document_content: Content of the document
        
    Returns:
        Extracted list of AD number 
    """
    # Enhanced AD number patterns - ordered by specificity to avoid date conflicts
    patterns = [
        # Most specific patterns first (with clear AD context)
        r'(?:EASA\s+)?AD\s+No\.?\s*:?\s*(\d{4}-\d{4})',
        r'(?:EASA\s+)?AD\s+No\s*:?\s*(\d{4}-\d{4})',
        r'AD\s+No\.?\s*:?\s*(\d{4}-\d{2}-\d{2})',
        r'AD\s+(\d{4}-\d{4})',
        r'AD\s+(\d{4}-\d{2}-\d{2})',
        r'emergency\s+AD\s+(\d{4}-\d{2}-\d{2})',
        r'Amendment\s+\d+-\d+;\s+AD\s+(\d{4}-\d{2}-\d{2})',
        
        # Context-based patterns with AD keywords
        r'AD\s*(?:NUMBER|No\.?)\s*:?\s*(\d{2,4}-\d{2,4}(?:-\d{2})?)',
        r'AIRWORTHINESS\s+DIRECTIVE\s*(?:NUMBER|No\.?)?\s*:?\s*(\d{2,4}-\d{2,4}(?:-\d{2})?)',
        r'Docket\s+No\.\s+\d{4}-[A-Z]{2}-\d+-AD;\s+Amendment\s+\d+-\d+;\s+AD\s+(\d{4}-\d{2}-\d{2})',
        
        # Parenthetical AD references
        r'(\d{2}-\d{2}-\d{2})\s*\(AIRWORTHINESS\s+DIRECTIVE\)',
        r'(\d{4}-\d{2}-\d{2})\s*\(AIRWORTHINESS\s+DIRECTIVE\)',
        r'(\d{4}-\d{4})\s*\(AIRWORTHINESS\s+DIRECTIVE\)',
        
        # Perform/comply with AD patterns
        r'(?:PERFORM|COMPLY\s+WITH)\s+(\d{2}-\d{2}-\d{2})',
        r'(?:PERFORM|COMPLY\s+WITH)\s+(\d{4}-\d{2}-\d{2})',
        r'(?:PERFORM|COMPLY\s+WITH)\s+(\d{4}-\d{4})',

        r'\b(?:EASA\s+)?AD\s+No\.?\s*[:\-]?\s*(\d{4}-\d{4})',
        r'\bAD\s+No\.?\s*[:\-]?\s*(\d{4}-\d{2}-\d{2})',
        r'\bAD\s+(\d{4}-\d{4})',
        r'\bAD\s+(\d{4}-\d{2}-\d{2})',
        r'\bAD\s+(\d{2}-\d{2}-\d{2})',
        r'Supersedes\s+AD\s+(\d{4}-\d{2}-\d{2})',
        r'Supersedes\s+AD\s+(\d{2}-\d{2}-\d{2})'
    ]
    
    found = set()
    for pattern in patterns:
        matches = re.findall(pattern, document_content, re.IGNORECASE)
        for m in matches:
            m = m.strip()
            if is_valid_ad_number(m):
                found.add(m)

    ad_numbers = sorted(list(found))
    logger.info(f"Regex extracted AD numbers: {ad_numbers}")
    return ad_numbers


def is_valid_ad_number(ad_number: str) -> bool:
    """
    Validate if the extracted number is likely an AD number and not a date
    
    Args:
        ad_number: The extracted number string
        
    Returns:
        True if it's likely a valid AD number
    """
    parts = ad_number.strip().split('-')

    if len(parts) == 2:
        # Format: YYYY-NNNN (EASA)
        year_part, second_part = parts
        if len(year_part) == 4 and len(second_part) == 4:
            try:
                year = int(year_part)
                return 1980 <= year <= 2035
            except ValueError:
                return False

    elif len(parts) == 3:
        # Formats: FAA (YYYY-NN-NN or YY-NN-NN)
        year, month, day = parts
        try:
            year = int(year)
            month = int(month)
            day = int(day)
            if 1 <= month <= 12 and 1 <= day <= 31:
                if (len(parts[0]) == 4 and 1980 <= year <= 2035) or (len(parts[0]) == 2 and 70 <= year <= 99):
                    return True
        except ValueError:
            return False

    return False



def has_ad_context(document_content: str, ad_number: str) -> bool:
    """
    Check if the document has proper AD context around the number
    
    Args:
        document_content: Full document content
        ad_number: The extracted AD number
        
    Returns:
        True if AD context is found
    """
    # Look for AD-related keywords near the number
    ad_keywords = [
        'AD', 'AIRWORTHINESS DIRECTIVE', 'EASA', 'FAA',
        'MANDATORY', 'COMPLY', 'PERFORM', 'INSPECTION',
        'AMENDMENT', 'DOCKET', 'EMERGENCY AD'
    ]
    
    # Find the position of the AD number in the document
    ad_pos = document_content.upper().find(ad_number.upper())
    if ad_pos == -1:
        return False
    
    # Check context around the AD number (±200 characters)
    start_pos = max(0, ad_pos - 200)
    end_pos = min(len(document_content), ad_pos + len(ad_number) + 200)
    context = document_content[start_pos:end_pos].upper()
    
    # Check if any AD keywords are in the context
    for keyword in ad_keywords:
        if keyword in context:
            return True
    
    return False


def create_folder_structure(base_path: str, ad_numbers: List[str]) -> Dict[str, str]:
    """
    Create folder structure for organizing documents by AD numbers
    
    Args:
        base_path: Base directory path
        ad_numbers: List of unique AD numbers
        
    Returns:
        Dictionary mapping AD numbers to folder paths
    """
    folder_mapping = {}
    base_dir = Path(base_path)
    
    # Create base directory if it doesn't exist
    base_dir.mkdir(parents=True, exist_ok=True)
    
    for ad_number in ad_numbers:
        # Sanitize AD number for folder name (replace invalid characters)
        safe_ad_number = re.sub(r'[<>:"/\\|?*]', '_', ad_number)
        folder_path = base_dir / f"AD_{safe_ad_number}"
        folder_path.mkdir(parents=True, exist_ok=True)
        folder_mapping[ad_number] = str(folder_path)
        logger.info(f"Created folder: {folder_path}")
    
    return folder_mapping


def get_text_files(input_directory: str) -> List[Path]:
    """
    Get all text files from input directory
    
    Args:
        input_directory: Directory containing input text files
        
    Returns:
        List of Path objects for text files
    """
    input_path = Path(input_directory)
    
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_directory}")
        return []
    
    text_files = list(input_path.glob("*.txt"))
    if not text_files:
        logger.warning(f"No .txt files found in {input_directory}")
        return []
    
    logger.info(f"Found {len(text_files)} text files to process")
    return text_files


def process_single_file(file_path: Path, bedrock_client=None) -> Tuple[List[str], str]:
    """
    Process a single file to extract all AD number
    
    Args:
        file_path: Path to the file
        bedrock_client: AWS Bedrock client (optional)
        
    Returns:
        Tuple of (list of AD numbers, file content)
    """
    logger.info(f"Processing file: {file_path.name}")
    
    # Read file content
    content = read_text_file(str(file_path))
    if not content:
        logger.warning(f"Skipping empty or unreadable file: {file_path.name}")
        return [], content
    
    combined_ads = []
    claude_ad_numbers = []
    regex_ad_numbers = []
    
    # Try Claude extraction if client is available
    if bedrock_client:
        claude_ad_numbers = extract_ad_numbers_with_claude(content, bedrock_client)
        claude_ad_numbers = [ad for ad in claude_ad_numbers if is_valid_ad_number(ad)]
    
    regex_ad_numbers = fallback_ad_extraction(content)
    regex_ad_numbers = [ad for ad in regex_ad_numbers if has_ad_context(content, ad)]

    combined_ads = sorted(set(claude_ad_numbers + regex_ad_numbers))
    if not combined_ads:
        logger.warning(f"No AD numbers found in: {file_path.name}")
    
    return combined_ads, content


def categorize_files(text_files: List[Path], use_claude: bool = True) -> Tuple[Dict[str, List[Path]], List[Path]]:
    bedrock_client = None
    if use_claude:
        try:
            bedrock_client = initialize_bedrock_client()
        except Exception as e:
            logger.warning(f"Failed to initialize Bedrock. Falling back to regex. {e}")
            use_claude = False

    ad_document_mapping = {}
    unclassified_files = []

    for file_path in text_files:
        ad_numbers, _ = process_single_file(file_path, bedrock_client if use_claude else None)
        
        if ad_numbers:
            for ad in ad_numbers:
                ad_document_mapping.setdefault(ad, []).append(file_path)
        else:
            unclassified_files.append(file_path)

    return ad_document_mapping, unclassified_files


def copy_files_to_folders(ad_document_mapping: Dict[str, List[Path]], folder_mapping: Dict[str, str]):
    """
    Copy files to their appropriate AD folders
    
    Args:
        ad_document_mapping: Mapping of AD numbers to file lists
        folder_mapping: Mapping of AD numbers to folder paths
    """
    for ad_number, files in ad_document_mapping.items():
        target_folder = folder_mapping[ad_number]
        for file_path in files:
            target_path = Path(target_folder) / file_path.name
            shutil.copy2(file_path, target_path)
            logger.info(f"Copied {file_path.name} to {target_folder}")


def handle_unclassified_files(unclassified_files: List[Path], output_directory: str):
    """
    Handle files that couldn't be classified
    
    Args:
        unclassified_files: List of unclassified file paths
        output_directory: Base output directory
    """
    if unclassified_files:
        unclassified_folder = Path(output_directory) / "unclassified"
        unclassified_folder.mkdir(parents=True, exist_ok=True)
        
        for file_path in unclassified_files:
            target_path = unclassified_folder / file_path.name
            shutil.copy2(file_path, target_path)
            logger.info(f"Copied unclassified file {file_path.name} to unclassified folder")


def print_summary(text_files: List[Path], ad_document_mapping: Dict[str, List[Path]], 
                 unclassified_files: List[Path], use_claude: bool = True):
    """
    Print organization summary
    
    Args:
        text_files: List of all processed files
        ad_document_mapping: Mapping of AD numbers to files
        unclassified_files: List of unclassified files
        use_claude: Whether Claude was used
    """
    mode_text = "" if use_claude else " (REGEX-ONLY MODE)"
    
    logger.info("\n" + "="*50)
    logger.info(f"ORGANIZATION SUMMARY{mode_text}")
    logger.info("="*50)
    logger.info(f"Total files processed: {len(text_files)}")
    logger.info(f"Files organized by AD number: {sum(len(files) for files in ad_document_mapping.values())}")
    logger.info(f"Unclassified files: {len(unclassified_files)}")
    logger.info(f"Unique AD numbers found: {len(ad_document_mapping)}")
    
    for ad_number, files in ad_document_mapping.items():
        logger.info(f"  - AD {ad_number}: {len(files)} files")
    
    if unclassified_files:
        logger.info(f"\nUnclassified files:")
        for file_path in unclassified_files:
            logger.info(f"  - {file_path.name}")


def organize_documents(input_directory: str, output_directory: str = "organized_ads", use_claude: bool = True):
    """
    Main function to organize documents by AD numbers
    
    Args:
        input_directory: Directory containing input text files
        output_directory: Directory where organized files will be saved
        use_claude: Whether to use Claude for extraction (fallback to regex if False)
    """
    # Get all text files
    text_files = get_text_files(input_directory)
    if not text_files:
        return
    
    # Categorize files by AD numbers
    ad_document_mapping, unclassified_files = categorize_files(text_files, use_claude)
    
    # Create folder structure and copy files
    if ad_document_mapping:
        unique_ad_numbers = list(ad_document_mapping.keys())
        folder_mapping = create_folder_structure(output_directory, unique_ad_numbers)
        copy_files_to_folders(ad_document_mapping, folder_mapping)
    
    # Handle unclassified files
    handle_unclassified_files(unclassified_files, output_directory)
    
    # Print summary
    print_summary(text_files, ad_document_mapping, unclassified_files, use_claude)



def main():
    """
    Main function to run the AD Document Organizer
    """
    # Configuration - Update these paths as needed
    INPUT_DIRECTORY = "/Users/garvagarwal/Aero-AI/structured_output_folder"  # Change this to your input directory
    OUTPUT_DIRECTORY = "organized_ads_copy3"   # Change this to your desired output directory
    
    # Get input directory from user
    # user_input_dir = input(f"Enter your input directory path (press Enter for default '{INPUT_DIRECTORY}'): ").strip()
    # if user_input_dir:
    #     INPUT_DIRECTORY = user_input_dir
    
    # user_output_dir = input(f"Enter your output directory path (press Enter for default '{OUTPUT_DIRECTORY}'): ").strip()
    # if user_output_dir:
    #     OUTPUT_DIRECTORY = user_output_dir
    
    print(f"Input directory: {INPUT_DIRECTORY}")
    print(f"Output directory: {OUTPUT_DIRECTORY}")
    
    try:
        # Try to organize documents with Claude
        print("Attempting to organize documents using Claude AI...")
        organize_documents(INPUT_DIRECTORY, OUTPUT_DIRECTORY, use_claude=True)
        logger.info("Document organization completed successfully!")
        
    except Exception as e:
        logger.error(f"Error with Claude-based extraction: {str(e)}")
        
        # If AWS Bedrock fails, offer to run with fallback only
        fallback_only = input("Would you like to try running with regex-based extraction only? (y/n): ").lower().strip()
        if fallback_only == 'y':
            try:
                print("Running with regex-based extraction...")
                organize_documents(INPUT_DIRECTORY, OUTPUT_DIRECTORY, use_claude=False)
                logger.info("Document organization completed with fallback method!")
            except Exception as e2:
                logger.error(f"Error with fallback method: {str(e2)}")


if __name__ == "__main__":
    main()