# Gunbrain AI

Gunbrain AI is a FastAPI-based application that enables querying and indexing of documents. The app allows users to upload files, perform internal and external queries, and retrieve information based on document content. It supports different types of prompts and sources to enhance the query functionality.

## Features

- **Upload and Index**: Upload documents and index them for future querying.
- **Internal Query**: Query the internal index to retrieve relevant information.
- **External Query**: Query an external source to gather additional data.

## Endpoints

### `/upload-and-index/` - Upload and Index Documents

Uploads a file and indexes it based on the source type provided. It requires the following parameters:

- `source_type` (Enum: `book`, `magazine`) - The type of the source.
- `file` (UploadFile) - The document to be uploaded.
- `title` (str) - The title of the document.
- `manufacturer` (str) - The manufacturer of the document.

#### Response

- `status` (str): Status message indicating the success or failure of the indexing process.

### `/internal-query` - Internal Query

Queries the internal index with a specified prompt type and query text. It requires the following parameters:

- `prompt_type` (Enum: `General`, `Information Audit`, `Serial Number Lookup`) - The type of the prompt.
- `query` (str) - The query text.

#### Response

- `response` (str): The answer to the query from the internal index.
- `citations` (list): List of citations related to the response.

### `/external-query/` - External Query

Queries an external source with a specified prompt type and query text. It requires the following parameters:

- `prompt_type` (Enum: `General`, `Information Audit`, `Serial Number Lookup`) - The type of the prompt.
- `query` (str) - The query text.

#### Response

- `response` (str): The answer to the query from the external source.
## Setting Environment Variables

To run the application, you need to set the following environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key.
- `PINECONE_API_KEY`: Your Pinecone API key.
- `ARYN_API_KEY`: Your ARYN API key.

You can set these variables in your environment or create a `.env` file in the root directory of the project and add them like so:

```env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
ARYN_API_KEY=your_aryn_api_key

## Installation

To run the Gunbrain AI app locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/gunbrain_ai.git
   cd gunbrain_ai
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
    source venv/bin/activate
    ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the FastAPI app:
   ```bash
   uvicorn main:app --reload
   ```
The app will be running at `http://127.0.0.1:8000`.

## Usage

- To upload and index a document, send a `POST` request to `/upload-and-index/` with the required parameters.
- To perform an internal query, send a `POST` request to `/internal-query` with the prompt type and query.
- To perform an external query, send a `POST` request to `/external-query/` with the prompt type and query.

