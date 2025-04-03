# index.py

import os
import json
import pickle
import logging
import shutil
import tempfile
from datetime import datetime

# aryn library
from aryn_sdk.partition import partition_file, table_elem_to_dataframe
from aryn_sdk.config import ArynConfig

# llama index
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import ServerlessSpec

from src.utils.utils import (
    pc,
    index_name,
    aryn_json_output_dir,
    list_files_from_s3_url,
    upload_to_s3,
    main_directory_path,
    logging,
    processed_docs_folder_s3_url,
    download_file_from_s3,
    table_description_extractor,
    aryn_ignored_elements,
    get_file_object,
    processed_docs_folder,
    extract_initials,
    is_s3_folder,
    delete_s3_object,
    index_logs_file_url,
    get_filename_from_s3_url,
    delete_files_in_directory,
    aryn_cache_dir_path,
    upsert_logs_file_url,
    convert_s3_to_http_url
)

from src.utils.token_counter import count_tokens

# sanitize for metadata
def sanitize_metadata(metadata):
    sanitized_metadata = {}
    for key, value in metadata.items():
        if value is None:
            # Assign default values based on expected type
            if key in ['layout_width', 'layout_height']:
                sanitized_metadata[key] = 0  # Default numerical value
            elif key in ['title', 'author', 'manufacturer', 'date_published', 'issue_date', 'website_url', 'origin_type']:
                sanitized_metadata[key] = ""
            else:
                sanitized_metadata[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            sanitized_metadata[key] = value
        elif isinstance(value, list):
            sanitized_metadata[key] = [str(item) for item in value]
        else:
            # Convert any other types to string
            sanitized_metadata[key] = str(value)
    return sanitized_metadata


def upload_and_index(title, source_type, manufacturer, website_url, file=None, s3_url=None):
    # download existing index logs
    index_logs_local_path = download_file_from_s3(index_logs_file_url, 'index_logs')
    try:
        with open(index_logs_local_path, 'r') as index_log_file:
            index_logs_data = json.load(index_log_file)
    except (json.JSONDecodeError, FileNotFoundError):
        logging.warning("index_logs.json is empty or corrupted. Initializing new logs.")
        index_logs_data = []

    # download existing upsert logs
    upsert_logs_local_path = download_file_from_s3(upsert_logs_file_url, 'upsert_logs')
    try:
        with open(upsert_logs_local_path, 'r') as upsert_log_file:
            upsert_logs_data = json.load(upsert_log_file)
    except (json.JSONDecodeError, FileNotFoundError):
        logging.warning("upsert_logs.json is empty or corrupted. Initializing new upsert logs.")
        upsert_logs_data = []

    # Handle file uploads
    if file:
        object_name = f"{main_directory_path}/{source_type}/{file.filename}"
        s3_url = upload_to_s3(file, object_name)
        s3_url_all = [s3_url]
    else:
        if is_s3_folder(s3_url):
            s3_url_all = list_files_from_s3_url(s3_url)
        else:
            s3_url_all = [s3_url]

    logging.info(f"All input S3 URLs: {s3_url_all}")

    # Check Pinecone indexes
    index_list_response = pc.list_indexes()
    index_names = [idx_info['name'] for idx_info in index_list_response.get('indexes', [])]
    logging.info(f"Available indexes before creation: {index_names}")

    if index_name not in index_names:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        logging.info(f"Created new Pinecone index: {index_name}")
    else:
        logging.info(f"Using existing Pinecone index: {index_name}")

    pinecone_index = pc.Index(index_name)

    # S3 folder for storing processed serialized files
    aryn_serialized_s3_url = f"{processed_docs_folder_s3_url}{source_type}/"
    usage_info_list = []

    # PDFs already fully indexed
    all_indexed_filenames = [indexed_element['filename'] for indexed_element in index_logs_data]

    for current_s3_url in s3_url_all:
        pdf_filename = get_filename_from_s3_url(current_s3_url)

        # Check if we already indexed this PDF
        if pdf_filename in all_indexed_filenames:
            logging.info(f'{current_s3_url} is already indexed. Stopping pipeline.')
            return {
                "message": f"Document '{pdf_filename}' is already indexed.",
                "total_pages": 0
            }

        # If not indexed, proceed with pipeline
        upsert_files = [upsert_element['filename'] for upsert_element in upsert_logs_data]

        # Aryn pipeline
        aryn_serialized_s3_urls = run_aryn_pipeline(
            download_s3_url=current_s3_url,
            upload_s3_url=aryn_serialized_s3_url,
            upsert_files=upsert_files
        )

        if not aryn_serialized_s3_urls:
            logging.warning(f"No new serialized files generated for {current_s3_url}. Skipping indexing.")
            return {
                "message": f"No new serialized files generated for '{pdf_filename}'. Stopping pipeline.",
                "total_pages": 0
            }

        # Clean up local pipeline cache
        logging.info(f"Deleting local files from {aryn_cache_dir_path}")
        delete_files_in_directory(aryn_cache_dir_path)

        # Update index logs
        index_logs_data.append({
            'filename': pdf_filename,
            'time': datetime.now().isoformat(),
            's3_url': current_s3_url,
            'input_s3_url': s3_url
        })
        try:
            with open(index_logs_local_path, "w") as f:
                json.dump(index_logs_data, f, indent=4)
            index_logs_json_file = get_file_object(index_logs_local_path)
            upload_to_s3(index_logs_json_file, index_logs_json_file.filename)
        except Exception as e:
            logging.error(f"Failed to update and upload index_logs.json: {e}")
            return {
                "message": f"Failed to update index logs for '{pdf_filename}': {e}",
                "total_pages": 0
            }

        pdf_total_pages = 0  # We'll update this after extracting documents

        for serialized_s3_url in aryn_serialized_s3_urls:
            if not (serialized_s3_url.endswith(".json") or serialized_s3_url.endswith(".pickle")):
                logging.warning(f"Skipping unsupported file format: {serialized_s3_url}")
                continue

            local_serialized_path = download_file_from_s3(serialized_s3_url, aryn_json_output_dir)

            # Extract llama_index docs (captures total pages)
            documents, pdf_total_pages = extract_llama_index_docs(
                json_path=local_serialized_path,
                source_type=source_type,
                website_url=website_url,
                pdf_s3_url=current_s3_url
            )

            if not documents:
                logging.warning(f"No documents extracted from {local_serialized_path}. Skipping embedding.")
                continue

            # Count tokens for usage tracking
            total_prompt_tokens = 0
            for doc in documents:
                tokens = count_tokens(doc.text, model="gpt-3.5-turbo")
                total_prompt_tokens += tokens
            logging.debug(f"Manual Token Count for {local_serialized_path}: {total_prompt_tokens}")

            # Embed and index documents
            try:
                embed_model = OpenAIEmbedding()
                vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)

                index_obj = VectorStoreIndex.from_documents(
                    documents=documents,
                    storage_context=storage_context,
                    embed_model=embed_model
                )
            except Exception as e:
                logging.error(f"Embedding failed for {local_serialized_path}: {e}")
                continue

            try:
                serialized_file = get_file_object(local_serialized_path)
                serialized_object_name = f"{processed_docs_folder}/{source_type}/{serialized_file.filename}"
                modified_serialized_s3_url = upload_to_s3(serialized_file, serialized_object_name)
            except Exception as e:
                logging.error(f"Failed to upload modified serialized file for {local_serialized_path}: {e}")
                continue

            try:
                # Remove the old serialized file from S3 if desired
                delete_s3_object(serialized_s3_url)
                # Clean up local file
                os.remove(local_serialized_path)
            except Exception as e:
                logging.warning(f"Failed to clean up after uploading {serialized_s3_url}: {e}")

            serialized_filename = get_filename_from_s3_url(serialized_s3_url)
            upsert_logs_data.append({
                'filename': serialized_filename,
                'time': datetime.now().isoformat(),
                's3_url': modified_serialized_s3_url
            })
            try:
                with open(upsert_logs_local_path, "w") as f:
                    json.dump(upsert_logs_data, f, indent=4)
                upsert_logs_json_file = get_file_object(upsert_logs_local_path)
                upload_to_s3(upsert_logs_json_file, upsert_logs_json_file.filename)
            except Exception as e:
                logging.error(f"Failed to update and upload upsert_logs.json: {e}")

            embed_completion_tokens = 0
            embed_total_tokens = total_prompt_tokens + embed_completion_tokens
            logging.info(
                f"[EMBEDDING] Serialized File={local_serialized_path}, "
                f"prompt_tokens={total_prompt_tokens}, "
                f"completion_tokens={embed_completion_tokens}, "
                f"total_tokens={embed_total_tokens}"
            )
            usage_info_list.append({
                "file": local_serialized_path,
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": embed_completion_tokens,
                "total_tokens": embed_total_tokens
            })

            # Persist index
            try:
                persist_dir = f"storage/{index_name}"
                index_obj.storage_context.persist(persist_dir=persist_dir)
                logging.info(f"Index for {current_s3_url} persisted at {persist_dir}")
            except Exception as e:
                logging.error(f"Failed to persist index for {current_s3_url}: {e}")

        object_url = convert_s3_to_http_url(current_s3_url)

        #it means we've successfully indexed this file
        return {
            "message": f"Document '{pdf_filename}' has been successfully indexed.",
            "s3_url": object_url,
            "total_pages": pdf_total_pages
        }

    return {
        "message": "No valid files found to index or all were skipped.",
        "embedding_usage": usage_info_list,
        "total_pages": 0
    }


def run_aryn_pipeline(download_s3_url: str, upload_s3_url: str, upsert_files: list, aryn_api_key: str = None) -> list:
    local_pdf_path = download_file_from_s3(download_s3_url, "tmp_aryn_input_dir")
    if not os.path.exists(local_pdf_path):
        logging.error(f"Failed to download PDF from {download_s3_url}.")
        return []

    pdf_basename = os.path.basename(local_pdf_path)
    pdf_name_no_ext = os.path.splitext(pdf_basename)[0]

    temp_dir = tempfile.mkdtemp(prefix=f"aryn_{pdf_name_no_ext}_")
    final_serialized_path = os.path.join(temp_dir, f"{pdf_name_no_ext}.json")

    try:
        with open(local_pdf_path, "rb") as f:
            chunking_opts = {
                "strategy": "context_rich",
                "tokenizer": "openai_tokenizer",
                "tokenizer_options": {
                    "model_name": "gpt-3.5-turbo"
                },
                "merge_across_pages": True,
                "max_tokens": 512
            }

            # Configure Aryn API key
            if aryn_api_key:
                aryn_config = ArynConfig(aryn_api_key=aryn_api_key)
            else:
                aryn_config = None

            # Call partition_file
            data = partition_file(
                f,
                aryn_api_key=aryn_api_key,
                aryn_config=aryn_config,
                extract_table_structure=True,
                extract_images=False,
                use_ocr=True,
                chunking_options=chunking_opts
            )
        try:
            with open(final_serialized_path, "w", encoding="utf-8") as out_f:
                json.dump(data, out_f, indent=4)
            logging.info(f"Serialized Aryn output to JSON at {final_serialized_path}")
        except TypeError:
            # If data is not JSON-serializable, serialize using pickle
            final_pickle_path = os.path.join(temp_dir, f"{pdf_name_no_ext}.pickle")
            with open(final_pickle_path, "wb") as out_f:
                pickle.dump(data, out_f)
            logging.info(f"Serialized Aryn output to Pickle at {final_pickle_path}")
            final_serialized_path = final_pickle_path  # Update path to pickle

        try:
            serialized_file = get_file_object(final_serialized_path)
            if not serialized_file:
                logging.error(f"Failed to get file object for {final_serialized_path}.")
                return []

            serialized_filename = os.path.basename(final_serialized_path)
            upload_s3_bucket_path = upload_s3_url.replace("s3://", "")
            remote_serialized_key = os.path.join(upload_s3_bucket_path, serialized_filename)

            final_s3_url = upload_to_s3(serialized_file, remote_serialized_key)
            if not final_s3_url:
                logging.error(f"Failed to upload serialized file to S3 at {remote_serialized_key}.")
                return []

            logging.info(f"[Aryn Pipeline] New serialized file uploaded: {final_s3_url}")
            return [final_s3_url]

        except Exception as e:
            logging.error(f"Failed to upload serialized file to S3: {e}")
            return []

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        if os.path.exists(local_pdf_path):
            os.remove(local_pdf_path)


def extract_llama_index_docs(json_path, source_type, website_url, pdf_s3_url=None):
    """
    Returns a tuple: ([Document objects], total_page_count)
    """
    llama_index_docs = []
    pdf_total_pages = 0

    try:
        if json_path.endswith(".json"):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif json_path.endswith(".pickle"):
            with open(json_path, "rb") as f:
                data = pickle.load(f)
        else:
            logging.error(f"Unsupported file format for {json_path}. Skipping.")
            return [], 0
    except (json.JSONDecodeError, FileNotFoundError, pickle.UnpicklingError) as e:
        logging.error(f"Failed to load data from {json_path}: {e}")
        return [], 0

    elements = data.get("elements", [])
    if not elements:
        logging.warning(f"No 'elements' found in {json_path}.")
        return [], 0

    # Gather total page count by scanning all page_number properties
    page_numbers = []
    for elem in elements:
        page_num = elem.get('properties', {}).get('page_number')
        if page_num is not None:
            page_numbers.append(page_num)
    if page_numbers:
        pdf_total_pages = max(page_numbers)

    # Extract metadata from the first 5 pages for initial info
    initial_book_content = ""
    for idx, item in enumerate(elements):
        page_number = item.get('properties', {}).get('page_number', 1)
        if page_number > 5:
            break

        element_type = item.get('type', '').lower()
        if element_type in [elem.lower() for elem in aryn_ignored_elements]:
            continue

        if element_type == "table":
            # Extract table text
            try:
                dataframe = table_elem_to_dataframe(item)
                table_text = dataframe.to_string(index=False)
                initial_book_content += table_text + "\n"
            except Exception as e:
                logging.warning(f"Failed to convert table element to DataFrame at index {idx}: {e}")
                initial_book_content += "\n"
        else:
            text = item.get("text_representation") or ""
            initial_book_content += text + "\n"

    # Extract metadata from the first 5 pages
    book_initials_data = extract_initials(initial_book_content)
    logging.info(f"Extracted Information from initials: {book_initials_data}")
    title = book_initials_data.title
    author = book_initials_data.authors
    manufacturer = book_initials_data.manufacturers
    date_published = book_initials_data.date_published
    issue_date = book_initials_data.issue_date
    origin_type = book_initials_data.origin_type

    # Build Document objects
    for idx, item in enumerate(elements):
        metadata = item.get('metadata', {})
        properties = item.get('properties', {})

        # Fill in metadata
        if title:
            metadata['title'] = title
        if author:
            metadata['author'] = author
        if manufacturer:
            metadata['manufacturer'] = manufacturer
        if date_published:
            metadata['date_published'] = date_published
        if issue_date:
            metadata['issue_date'] = issue_date
        if website_url:
            metadata['website_url'] = website_url
        if origin_type:
            metadata['origin_type'] = origin_type

        bbox = metadata.get('bbox')
        if bbox:
            logging.info(f"BBOX: {bbox}")

        metadata['page_number'] = properties.get('page_number', 1)
        metadata['parent_id'] = properties.get('parent_id', '')
        metadata['filename'] = metadata.get('filename', os.path.basename(pdf_s3_url))
        metadata['type'] = item.get('type', '')
        metadata['element_id'] = metadata.get('element_id', '')
        metadata['bbox'] = item.get('bbox', [])

        data_source_url = metadata.get('data_source', {}).get('url', '')
        if not data_source_url and pdf_s3_url:
            data_source_url = pdf_s3_url
        metadata['s3_URL'] = data_source_url

        element_type = item.get('type', '').lower()
        text_representation = item.get("text_representation") or ""

        # Skip ignored elements
        if element_type in [elem.lower() for elem in aryn_ignored_elements]:
            continue

        # If table, we create a descriptive text about it
        if element_type == "table":
            before_table_text = ''
            after_table_text = ''

            # Look backward
            cur_idx = idx
            while len(before_table_text.split()) < 400:
                cur_idx -= 1
                if cur_idx < 0:
                    break
                prev_item = elements[cur_idx]
                prev_type = prev_item.get('type', '').lower()
                if prev_type in ["table"] + [elem.lower() for elem in aryn_ignored_elements]:
                    continue
                prev_text = prev_item.get("text_representation") or ""
                before_table_text = prev_text + "\n" + before_table_text

            # Look forward
            cur_idx = idx
            while len(after_table_text.split()) < 400 and cur_idx < len(elements) - 1:
                cur_idx += 1
                next_item = elements[cur_idx]
                next_type = next_item.get('type', '').lower()
                if next_type in ["table"] + [elem.lower() for elem in aryn_ignored_elements]:
                    continue
                next_text = next_item.get("text_representation") or ""
                after_table_text += next_text + "\n"

            # Extract table content
            try:
                dataframe = table_elem_to_dataframe(item)
                table_content = dataframe.to_string(index=False)
                logging.debug(f"Extracted table content for element index {idx}.")
            except Exception as e:
                logging.warning(f"Failed to convert table element to DataFrame at index {idx}: {e}")
                table_content = ""

            logging.info(f"Before Table Text: {before_table_text}")
            logging.info(f"Table Content: {table_content}")
            logging.info(f"After Table Text: {after_table_text}")
            try:
                descriptive_paragraph = table_description_extractor(
                    before_table_text,
                    table_content,
                    after_table_text
                )
            except Exception as e:
                logging.warning(f"table_description_extractor failed: {e}")
                descriptive_paragraph = ""

            text_representation = descriptive_paragraph + "\nTableContent:\n" + table_content
            item['text_representation'] = text_representation
        else:
            item['text_representation'] = text_representation

        # Sanitize and store metadata
        metadata = sanitize_metadata(metadata)
        item['metadata'] = metadata

        llama_index_docs.append(
            Document(text=item['text_representation'], metadata=metadata)
        )

    # Write back updated JSON (with any new or fixed metadata)
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        logging.error(f"Failed to write back updated JSON at {json_path}: {e}")

    # Return both the documents and the total page count
    return llama_index_docs, pdf_total_pages
