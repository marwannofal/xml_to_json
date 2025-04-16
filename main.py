from fastapi import FastAPI, Request, HTTPException, Header, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional, Union
from pydantic import BaseModel
import logging
import logging.handlers
import os
import json
import time
import xml.etree.ElementTree as ET
import uvicorn
import uuid
import shutil
import traceback
from datetime import datetime

# Setup directories
LOG_DIR = "logs"
UPLOAD_DIR = "uploads"
ARCHIVE_DIR = "archive"  # New directory for alternative storage
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(ARCHIVE_DIR, exist_ok=True)

# Configure logger
logger = logging.getLogger("xml_api")
logger.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [RequestID: %(request_id)s] - %(message)s')
console_handler.setFormatter(console_format)

# File handler with rotation
file_handler = logging.handlers.RotatingFileHandler(
    filename=f"{LOG_DIR}/xml_api.log",
    maxBytes=10485760,  # 10MB
    backupCount=10
)
file_handler.setLevel(logging.DEBUG)
file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [RequestID: %(request_id)s] - %(message)s')
file_handler.setFormatter(file_format)

# Custom log filter to add request ID
class RequestIDFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, "request_id"):
            record.request_id = "N/A"
        return True

# Add filter to handlers
console_handler.addFilter(RequestIDFilter())
file_handler.addFilter(RequestIDFilter())

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit per file

app = FastAPI(title="XML Request Handler API",
              description="API for handling XML requests with various metadata headers",
              version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define Pydantic models
class XMLData(BaseModel):
    xml: str
    filename: Optional[str] = None

class XMLDataList(BaseModel):
    files: List[XMLData]

class ProcessingResult(BaseModel):
    success: bool
    message: str
    timestamp: str
    xml_count: int
    details: Optional[Dict] = None

# Utility functions
def validate_xml_content(xml_content: str, log_extra: dict):
    """Validate XML content by attempting to parse it"""
    try:
        root = ET.fromstring(xml_content)
        return root
    except ET.ParseError as e:
        logger.error(f"XML parsing error: {str(e)}", extra=log_extra)
        raise HTTPException(status_code=400, detail=f"XML parsing error: {str(e)}")

def save_file_with_unique_name(target_dir: str, original_filename: str, log_extra: dict):
    """Save a file with a unique name if a file with the same name already exists"""
    safe_filename = os.path.basename(original_filename)
    file_path = os.path.join(target_dir, safe_filename)

    # Check if file already exists
    if os.path.exists(file_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_parts = os.path.splitext(safe_filename)
        safe_filename = f"{filename_parts[0]}_{timestamp}{filename_parts[1]}"
        file_path = os.path.join(target_dir, safe_filename)
        logger.info(f"File already exists, renamed to: {safe_filename}", extra=log_extra)

    return file_path, safe_filename

def save_xml_file(xml_content: str, target_dir: str, filename: Optional[str] = None, metadata: Optional[Dict] = None, log_extra: Optional[Dict] = None):
    """
    Save XML content to the specified directory

    Args:
        xml_content: The XML string to save
        target_dir: Directory to save the file in
        filename: Optional filename to use (will be generated if not provided)
        metadata: Optional metadata to include in the filename or logging
        log_extra: Optional logging extra data

    Returns:
        str: The path where the file was saved
    """
    try:
        if not log_extra:
            log_extra = {"request_id": "N/A"}

        # Generate a unique filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]

            # Use default filename from metadata if available
            if metadata and metadata.get("x_meta_default_filename"):
                base_filename = metadata.get("x_meta_default_filename")
                # Extract base name without extension
                if "." in base_filename:
                    base_name = base_filename.rsplit(".", 1)[0]
                else:
                    base_name = base_filename
                filename = f"{base_name}_{timestamp}_{unique_id}.xml"
            else:
                filename = f"xml_{timestamp}_{unique_id}.xml"

        # Ensure the filename has .xml extension
        if not filename.lower().endswith('.xml'):
            filename += '.xml'

        # Create complete file path and handle duplicates
        file_path, safe_filename = save_file_with_unique_name(target_dir, filename, log_extra)

        # Save XML content to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(xml_content)

        logger.info(f"XML file saved: {file_path}", extra=log_extra)
        return file_path

    except Exception as e:
        logger.error(f"Error saving XML file: {str(e)}", extra=log_extra)
        raise

@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Generate a unique request ID
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    # Start time
    start_time = time.time()

    # Set request_id for logging
    log_extra = {"request_id": request_id}

    logger.info(
        f"Request started: {request.method} {request.url.path}",
        extra=log_extra
    )

    try:
        # Process the request
        response = await call_next(request)

        # Calculate processing time
        process_time = time.time() - start_time
        logger.info(
            f"Request completed: {request.method} {request.url.path} - Status: {response.status_code} - Duration: {process_time:.3f}s",
            extra=log_extra
        )

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        return response

    except Exception as e:
        # Log the exception
        logger.error(
            f"Request failed: {request.method} {request.url.path} - Error: {str(e)}",
            extra=log_extra
        )
        logger.error(f"Exception details: {traceback.format_exc()}", extra=log_extra)

        # Return JSON error response
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "request_id": request_id}
        )

# Custom exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log_extra = {"request_id": getattr(request.state, "request_id", "N/A")}

    # Different handling based on exception type
    if isinstance(exc, HTTPException):
        logger.warning(
            f"HTTP Exception: {exc.status_code} - {exc.detail}",
            extra=log_extra
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "request_id": log_extra["request_id"]}
        )
    else:
        logger.error(
            f"Unhandled exception: {str(exc)}",
            extra=log_extra
        )
        logger.error(f"Exception traceback: {traceback.format_exc()}", extra=log_extra)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "request_id": log_extra["request_id"]}
        )


async def process_xml_request(
    request: Request,
    target_dir: str,
    meta_headers: Dict,
    files: Optional[List[UploadFile]] = None,
    log_extra: Optional[Dict] = None
) -> ProcessingResult:
    """
    Common function to process XML requests for different endpoints

    Args:
        request: The FastAPI request object
        target_dir: Directory to save XML files
        meta_headers: Metadata headers dictionary
        files: Optional list of uploaded files
        log_extra: Logging extra data dictionary

    Returns:
        ProcessingResult: The processing result
    """
    try:
        # Variables to store XML data
        xml_data = []
        xml_sources = []  # To track where each XML came from (file, body, etc.)

        # Check if files were uploaded
        if files:
            logger.info(f"Processing {len(files)} uploaded files", extra=log_extra)
            for i, file in enumerate(files):
                if not file:
                    continue

                # Validate file extension
                if not file.filename.lower().endswith('.xml'):
                    logger.warning(f"Skipping non-XML file: {file.filename}", extra=log_extra)
                    continue

                # Read file content
                file_content = await file.read()
                xml_str = file_content.decode("utf-8")
                xml_data.append(xml_str)
                xml_sources.append(f"file-{file.filename}")
                logger.debug(f"Processed file {i+1}/{len(files)}: {file.filename}", extra=log_extra)

        # Try to determine content type and process accordingly
        content_type = request.headers.get("content-type", "").lower()
        logger.debug(f"Content-Type: {content_type}", extra=log_extra)

        if not files or "multipart/form-data" not in content_type:
            # Process body content if no files or not multipart form
            if "application/json" in content_type:
                # Handle JSON payload
                logger.debug("Processing JSON payload", extra=log_extra)
                payload = await request.json()

                if isinstance(payload, dict):
                    if "xml" in payload:
                        # Single XML object
                        xml_data.append(payload["xml"])
                        xml_sources.append("json-body-single")
                    elif "files" in payload and isinstance(payload["files"], list):
                        # Multiple XML objects in files array
                        for item in payload["files"]:
                            if isinstance(item, dict) and "xml" in item:
                                xml_data.append(item["xml"])
                                xml_sources.append(f"json-body-multiple-{item.get('filename', 'unnamed')}")
                elif isinstance(payload, list):
                    # Multiple XML objects in array
                    for item in payload:
                        if isinstance(item, dict) and "xml" in item:
                            xml_data.append(item["xml"])
                            xml_sources.append(f"json-body-array-{item.get('filename', 'unnamed')}")

            elif "application/x-www-form-urlencoded" in content_type:
                # Handle form data
                logger.debug("Processing form-encoded payload", extra=log_extra)
                form_data = await request.form()

                if "xml" in form_data:
                    xml_data.append(form_data["xml"])
                    xml_sources.append("form-data-single")
                else:
                    # Check if multiple XML objects are present
                    xml_keys = [k for k in form_data.keys() if k.startswith("xml")]
                    for key in xml_keys:
                        xml_data.append(form_data[key])
                        xml_sources.append(f"form-data-{key}")

            else:
                # Default: Try to get raw body as XML
                logger.debug("Processing raw body as XML", extra=log_extra)
                body = await request.body()
                if body:
                    xml_content = body.decode("utf-8")

                    # Check if we have multiple XML objects or a single one
                    if xml_content.strip().startswith("<xml>") or xml_content.strip().startswith("<?xml"):
                        xml_data.append(xml_content)
                        xml_sources.append("raw-body-xml")
                    else:
                        # Attempt to parse as JSON
                        try:
                            json_data = json.loads(xml_content)
                            if isinstance(json_data, dict):
                                if "xml" in json_data:
                                    xml_data.append(json_data["xml"])
                                    xml_sources.append("raw-body-json-single")
                                elif "files" in json_data and isinstance(json_data["files"], list):
                                    for item in json_data["files"]:
                                        if isinstance(item, dict) and "xml" in item:
                                            xml_data.append(item["xml"])
                                            xml_sources.append(f"raw-body-json-multiple-{item.get('filename', 'unnamed')}")
                            elif isinstance(json_data, list):
                                for item in json_data:
                                    if isinstance(item, dict) and "xml" in item:
                                        xml_data.append(item["xml"])
                                        xml_sources.append(f"raw-body-json-array-{item.get('filename', 'unnamed')}")
                        except json.JSONDecodeError:
                            # If not JSON, treat as a single XML document
                            if not xml_data:  # Only add if we haven't added this content yet
                                xml_data = [xml_content]
                                xml_sources = ["raw-body"]

        # If no XML data found, return error
        if not xml_data:
            error_msg = "No XML data found in request"
            logger.error(error_msg, extra=log_extra)
            return ProcessingResult(
                success=False,
                message=error_msg,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                xml_count=0,
                details={"request_id": log_extra["request_id"]}
            )

        # Process all XML objects
        processed_results = []
        saved_files = []

        for i, (xml_content, source) in enumerate(zip(xml_data, xml_sources)):
            try:
                # Validate XML by parsing it
                root = validate_xml_content(xml_content, log_extra)

                # Log successful parsing
                logger.info(f"Successfully parsed XML document {i+1}/{len(xml_data)} from {source}", extra=log_extra)

                # Generate a specific filename for this XML document
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Use metadata to construct a meaningful filename
                filename_parts = []

                # Try to use default filename from headers
                if meta_headers.get("x_meta_default_filename"):
                    base_name = meta_headers["x_meta_default_filename"].rsplit(".", 1)[0] if "." in meta_headers["x_meta_default_filename"] else meta_headers["x_meta_default_filename"]
                    filename_parts.append(base_name)

                # Add game-specific identifiers if available
                if meta_headers.get("x_meta_game_id"):
                    filename_parts.append(f"game{meta_headers['x_meta_game_id']}")

                if meta_headers.get("x_meta_home_team_id") and meta_headers.get("x_meta_away_team_id"):
                    filename_parts.append(f"match_{meta_headers['x_meta_home_team_id']}vs{meta_headers['x_meta_away_team_id']}")

                # Add sequential index for multiple files in one request
                if len(xml_data) > 1:
                    filename_parts.append(f"part{i+1}")

                # Add source information
                if source.startswith("file-"):
                    # Extract original filename from source
                    orig_filename = source[5:]
                    if not filename_parts:
                        base_name = orig_filename.rsplit(".", 1)[0] if "." in orig_filename else orig_filename
                        filename_parts.append(base_name)

                # Combine parts and add timestamp
                if filename_parts:
                    custom_filename = f"{'_'.join(filename_parts)}_{timestamp}.xml"
                else:
                    custom_filename = f"xml_{timestamp}_{i+1}.xml"

                # Save the file
                file_path = save_xml_file(xml_content, target_dir, custom_filename, meta_headers, log_extra)
                saved_files.append(file_path)

                # Add processing result
                processed_results.append({
                    "index": i,
                    "source": source,
                    "root_tag": root.tag,
                    "child_count": len(list(root)),
                    "saved_path": file_path
                })

            except ET.ParseError as e:
                error_msg = f"XML parsing error in document {i+1} from {source}: {str(e)}"
                logger.error(error_msg, extra=log_extra)
                return ProcessingResult(
                    success=False,
                    message=error_msg,
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    xml_count=len(xml_data),
                    details={"error_index": i, "request_id": log_extra["request_id"]}
                )
            except Exception as e:
                error_msg = f"Error processing XML document {i+1} from {source}: {str(e)}"
                logger.error(error_msg, extra=log_extra)
                return ProcessingResult(
                    success=False,
                    message=error_msg,
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    xml_count=len(xml_data),
                    details={"error_index": i, "exception": str(e), "request_id": log_extra["request_id"]}
                )

        # Create success response
        return ProcessingResult(
            success=True,
            message=f"Successfully processed and saved {len(xml_data)} XML document(s) to {os.path.basename(target_dir)}",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            xml_count=len(xml_data),
            details={
                "processed": processed_results,
                "headers": meta_headers,
                "saved_files": saved_files,
                "target_directory": target_dir,
                "request_id": log_extra["request_id"]
            }
        )

    except Exception as e:
        error_message = f"Error processing request: {str(e)}"
        logger.exception(error_message, extra=log_extra)
        return ProcessingResult(
            success=False,
            message=error_message,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            xml_count=0,
            details={"exception": str(e), "request_id": log_extra["request_id"]}
        )

@app.post("/api/xml", response_model=ProcessingResult)
async def process_xml(
    request: Request,
    x_meta_feed_type: Optional[str] = Header(None),
    x_meta_feed_parameters: Optional[str] = Header(None),
    x_meta_default_filename: Optional[str] = Header(None),
    x_meta_game_id: Optional[str] = Header(None),
    x_meta_competition_id: Optional[str] = Header(None),
    x_meta_season_id: Optional[str] = Header(None),
    x_meta_gamesystem_id: Optional[str] = Header(None),
    x_meta_matchday: Optional[str] = Header(None),
    x_meta_away_team_id: Optional[str] = Header(None),
    x_meta_home_team_id: Optional[str] = Header(None),
    x_meta_game_status: Optional[str] = Header(None),
    x_meta_language: Optional[str] = Header(None),
    x_meta_production_server: Optional[str] = Header(None),
    x_meta_production_server_timestamp: Optional[str] = Header(None),
    x_meta_production_server_module: Optional[str] = Header(None),
    x_meta_mime_type: Optional[str] = Header(None),
    encoding: Optional[str] = Header(None),
    files: Optional[List[UploadFile]] = File(None)
):
    """Process XML data and save to the default uploads directory"""
    log_extra = {"request_id": getattr(request.state, "request_id", "N/A")}

    # Collect metadata headers
    meta_headers = {
        "x_meta_feed_type": x_meta_feed_type,
        "x_meta_feed_parameters": x_meta_feed_parameters,
        "x_meta_default_filename": x_meta_default_filename,
        "x_meta_game_id": x_meta_game_id,
        "x_meta_competition_id": x_meta_competition_id,
        "x_meta_season_id": x_meta_season_id,
        "x_meta_gamesystem_id": x_meta_gamesystem_id,
        "x_meta_matchday": x_meta_matchday,
        "x_meta_away_team_id": x_meta_away_team_id,
        "x_meta_home_team_id": x_meta_home_team_id,
        "x_meta_game_status": x_meta_game_status,
        "x_meta_language": x_meta_language,
        "x_meta_production_server": x_meta_production_server,
        "x_meta_production_server_timestamp": x_meta_production_server_timestamp,
        "x_meta_production_server_module": x_meta_production_server_module,
        "x_meta_mime_type": x_meta_mime_type,
        "encoding": encoding
    }

    logger.info(f"Received request to /api/xml with headers: {json.dumps(meta_headers)}", extra=log_extra)

    try:
        # Process the request using the common function
        result = await process_xml_request(
            request=request,
            target_dir=UPLOAD_DIR,
            meta_headers=meta_headers,
            files=files,
            log_extra=log_extra
        )

        if result.success:
            logger.info(f"Request to /api/xml completed successfully, processed {result.xml_count} XML documents", extra=log_extra)
            return result
        else:
            logger.error(f"Request to /api/xml failed: {result.message}", extra=log_extra)
            return JSONResponse(status_code=400, content=result.dict())

    except HTTPException as http_ex:
        # Re-raise HTTP exceptions
        logger.error(f"HTTP exception: {http_ex.detail}", extra=log_extra)
        raise

    except Exception as e:
        # Log and handle general exceptions
        error_message = f"Error processing request: {str(e)}"
        logger.exception(error_message, extra=log_extra)
        return JSONResponse(
            status_code=500,
            content=ProcessingResult(
                success=False,
                message=error_message,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                xml_count=0,
                details={"exception": str(e), "request_id": log_extra["request_id"]}
            ).dict()
        )

@app.post("/api/xml_secondary", response_model=ProcessingResult)
async def process_xml_archive(
    request: Request,
    x_meta_feed_type: Optional[str] = Header(None),
    x_meta_feed_parameters: Optional[str] = Header(None),
    x_meta_default_filename: Optional[str] = Header(None),
    x_meta_game_id: Optional[str] = Header(None),
    x_meta_competition_id: Optional[str] = Header(None),
    x_meta_season_id: Optional[str] = Header(None),
    x_meta_gamesystem_id: Optional[str] = Header(None),
    x_meta_matchday: Optional[str] = Header(None),
    x_meta_away_team_id: Optional[str] = Header(None),
    x_meta_home_team_id: Optional[str] = Header(None),
    x_meta_game_status: Optional[str] = Header(None),
    x_meta_language: Optional[str] = Header(None),
    x_meta_production_server: Optional[str] = Header(None),
    x_meta_production_server_timestamp: Optional[str] = Header(None),
    x_meta_production_server_module: Optional[str] = Header(None),
    x_meta_mime_type: Optional[str] = Header(None),
    encoding: Optional[str] = Header(None),
    files: Optional[List[UploadFile]] = File(None)
):
    """Process XML data and save to the archive directory"""
    log_extra = {"request_id": getattr(request.state, "request_id", "N/A")}

    # Collect metadata headers
    meta_headers = {
        "x_meta_feed_type": x_meta_feed_type,
        "x_meta_feed_parameters": x_meta_feed_parameters,
        "x_meta_default_filename": x_meta_default_filename,
        "x_meta_game_id": x_meta_game_id,
        "x_meta_competition_id": x_meta_competition_id,
        "x_meta_season_id": x_meta_season_id,
        "x_meta_gamesystem_id": x_meta_gamesystem_id,
        "x_meta_matchday": x_meta_matchday,
        "x_meta_away_team_id": x_meta_away_team_id,
        "x_meta_home_team_id": x_meta_home_team_id,
        "x_meta_game_status": x_meta_game_status,
        "x_meta_language": x_meta_language,
        "x_meta_production_server": x_meta_production_server,
        "x_meta_production_server_timestamp": x_meta_production_server_timestamp,
        "x_meta_production_server_module": x_meta_production_server_module,
        "x_meta_mime_type": x_meta_mime_type,
        "encoding": encoding
    }

    logger.info(f"Received request to /api/archive/xml with headers: {json.dumps(meta_headers)}", extra=log_extra)

    try:
        # Process the request using the common function but target the archive directory
        result = await process_xml_request(
            request=request,
            target_dir=ARCHIVE_DIR,
            meta_headers=meta_headers,
            files=files,
            log_extra=log_extra
        )

        if result.success:
            logger.info(f"Request to /api/archive/xml completed successfully, processed {result.xml_count} XML documents", extra=log_extra)
            return result
        else:
            logger.error(f"Request to /api/archive/xml failed: {result.message}", extra=log_extra)
            return JSONResponse(status_code=400, content=result.dict())

    except HTTPException as http_ex:
        # Re-raise HTTP exceptions
        logger.error(f"HTTP exception: {http_ex.detail}", extra=log_extra)
        raise

    except Exception as e:
        # Log and handle general exceptions
        error_message = f"Error processing request: {str(e)}"
        logger.exception(error_message, extra=log_extra)
        return JSONResponse(
            status_code=500,
            content=ProcessingResult(
                success=False,
                message=error_message,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                xml_count=0,
                details={"exception": str(e), "request_id": log_extra["request_id"]}
            ).dict()
        )

@app.get("/health")
async def health_check(request: Request):
    """Health check endpoint"""
    log_extra = {"request_id": getattr(request.state, "request_id", "N/A")}
    logger.info("Health check endpoint accessed", extra=log_extra)

    return {
        "status": "healthy",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "uploads_dir": {
            "exists": os.path.exists(UPLOAD_DIR),
            "file_count": len([f for f in os.listdir(UPLOAD_DIR) if f.endswith('.xml')]) if os.path.exists(UPLOAD_DIR) else 0
        },
        "archive_dir": {
            "exists": os.path.exists(ARCHIVE_DIR),
            "file_count": len([f for f in os.listdir(ARCHIVE_DIR) if f.endswith('.xml')]) if os.path.exists(ARCHIVE_DIR) else 0
        },
        "logs_dir_exists": os.path.exists(LOG_DIR),
        "request_id": log_extra["request_id"]
    }

@app.get("/")
async def root(request: Request):
    """API root endpoint with basic information"""
    log_extra = {"request_id": getattr(request.state, "request_id", "N/A")}
    logger.info("Root endpoint accessed", extra=log_extra)

    return {
        "name": "XML Request Handler API",
        "version": "1.0.0",
        "endpoints": {
            "/api/xml": "POST - Process XML requests (supports file uploads and raw XML)",
            "/upload-xml-raw/": "POST - Legacy endpoint for uploading a single XML string",
            "/upload-xml-raw-multiple/": "POST - Legacy endpoint for uploading multiple XML strings",
            "/health": "GET - Health check"
        },
        "upload_directory": os.path.abspath(UPLOAD_DIR),
        "xml_files_stored": len([f for f in os.listdir(UPLOAD_DIR) if f.endswith('.xml')]) if os.path.exists(UPLOAD_DIR) else 0,
        "request_id": log_extra["request_id"]
    }

if __name__ == "__main__":
    logger.info("Starting XML Request Handler API")
    logger.info(f"Files will be saved to: {os.path.abspath(UPLOAD_DIR)}")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)