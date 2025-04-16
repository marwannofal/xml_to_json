from fastapi import FastAPI, Request, HTTPException, Header, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional
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

# New imports for XML-to-JSON conversion and MongoDB integration
import xmltodict  # for parsing XML into a dict
from pymongo import MongoClient as DB
from bson import ObjectId

# ------------------------------
# MongoDB setup
# ------------------------------
client = DB("mongodb://localhost:27017/")
db_mongo = client["xml_to_json"]
collection = db_mongo["xml_to_json"]

# Custom JSON encoder that converts ObjectId to strings
class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return super().default(o)

class CustomJSONResponse(JSONResponse):
    def render(self, content: any) -> bytes:
        return json.dumps(content, cls=JSONEncoder, indent=4).encode("utf-8")

# ------------------------------
# Setup directories and logging
# ------------------------------
LOG_DIR = "logs"
UPLOAD_DIR = "uploads"
ARCHIVE_DIR = "archive"  # New directory for alternative storage

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(ARCHIVE_DIR, exist_ok=True)

logger = logging.getLogger("xml_api")
logger.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_format = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [RequestID: %(request_id)s] - %(message)s'
)
console_handler.setFormatter(console_format)

# File handler with rotation
file_handler = logging.handlers.RotatingFileHandler(
    filename=f"{LOG_DIR}/xml_api.log",
    maxBytes=10485760,  # 10MB
    backupCount=10
)
file_handler.setLevel(logging.DEBUG)
file_format = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [RequestID: %(request_id)s] - %(message)s'
)
file_handler.setFormatter(file_format)

# Custom log filter to add request ID
class RequestIDFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, "request_id"):
            record.request_id = "N/A"
        return True

console_handler.addFilter(RequestIDFilter())
file_handler.addFilter(RequestIDFilter())
logger.addHandler(console_handler)
logger.addHandler(file_handler)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit per file

app = FastAPI(
    title="XML Request Handler API",
    description="API for handling XML requests with file, JSON, MongoDB, and JSON file storage",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# Pydantic models for responses
# ------------------------------
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

# ------------------------------
# Utility functions
# ------------------------------
def validate_xml_content(xml_content: str, log_extra: dict):
    """Validate XML content by attempting to parse it using xml.etree.ElementTree."""
    try:
        root = ET.fromstring(xml_content)
        return root
    except ET.ParseError as e:
        logger.error(f"XML parsing error: {str(e)}", extra=log_extra)
        raise HTTPException(status_code=400, detail=f"XML parsing error: {str(e)}")

def save_file_with_unique_name(target_dir: str, original_filename: str, log_extra: dict):
    """Save a file with a unique name if a file with the same name already exists."""
    safe_filename = os.path.basename(original_filename)
    file_path = os.path.join(target_dir, safe_filename)
    if os.path.exists(file_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_parts = os.path.splitext(safe_filename)
        safe_filename = f"{filename_parts[0]}_{timestamp}{filename_parts[1]}"
        file_path = os.path.join(target_dir, safe_filename)
        logger.info(f"File already exists, renamed to: {safe_filename}", extra=log_extra)
    return file_path, safe_filename

def save_xml_file(xml_content: str, target_dir: str, filename: Optional[str] = None,
                  metadata: Optional[Dict] = None, log_extra: Optional[Dict] = None):
    """
    Save XML content to a specified directory.
    Generates a unique filename if none is provided.
    """
    if not log_extra:
        log_extra = {"request_id": "N/A"}
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        if metadata and metadata.get("x_meta_default_filename"):
            base_filename = metadata.get("x_meta_default_filename")
            base_name = base_filename.rsplit(".", 1)[0] if "." in base_filename else base_filename
            filename = f"{base_name}_{timestamp}_{unique_id}.xml"
        else:
            filename = f"xml_{timestamp}_{unique_id}.xml"
    if not filename.lower().endswith('.xml'):
        filename += '.xml'
    file_path, safe_filename = save_file_with_unique_name(target_dir, filename, log_extra)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(xml_content)
    logger.info(f"XML file saved: {file_path}", extra=log_extra)
    return file_path

# ------------------------------
# Middleware for Logging
# ------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start_time = time.time()
    log_extra = {"request_id": request_id}
    logger.info(f"Request started: {request.method} {request.url.path}", extra=log_extra)
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(
            f"Request completed: {request.method} {request.url.path} - Status: {response.status_code} - Duration: {process_time:.3f}s",
            extra=log_extra
        )
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception as e:
        logger.error(
            f"Request failed: {request.method} {request.url.path} - Error: {str(e)}",
            extra=log_extra
        )
        logger.error(f"Exception details: {traceback.format_exc()}", extra=log_extra)
        return JSONResponse(status_code=500, content={"detail": "Internal server error", "request_id": request_id})

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log_extra = {"request_id": getattr(request.state, "request_id", "N/A")}
    if isinstance(exc, HTTPException):
        logger.warning(f"HTTP Exception: {exc.status_code} - {exc.detail}", extra=log_extra)
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail, "request_id": log_extra["request_id"]})
    else:
        logger.error(f"Unhandled exception: {str(exc)}", extra=log_extra)
        logger.error(f"Exception traceback: {traceback.format_exc()}", extra=log_extra)
        return JSONResponse(status_code=500, content={"detail": "Internal server error", "request_id": log_extra["request_id"]})

# ------------------------------
# Core XML processing function with MongoDB and JSON file storage
# ------------------------------
async def process_xml_request(
    request: Request,
    target_dir: str,
    meta_headers: Dict,
    files: Optional[List[UploadFile]] = None,
    log_extra: Optional[Dict] = None
) -> ProcessingResult:
    """
    Processes XML requests by reading XML from files or request bodies,
    validates and saves each XML file to disk, parses the XML into JSON using xmltodict,
    inserts the JSON into MongoDB, and saves the JSON to a file.
    """
    try:
        xml_data = []
        xml_sources = []  # Track the source for each XML document

        # Process file uploads if provided
        if files:
            logger.info(f"Processing {len(files)} uploaded files", extra=log_extra)
            for i, file in enumerate(files):
                if not file:
                    continue
                if not file.filename.lower().endswith('.xml'):
                    logger.warning(f"Skipping non-XML file: {file.filename}", extra=log_extra)
                    continue
                file_content = await file.read()
                xml_str = file_content.decode("utf-8")
                xml_data.append(xml_str)
                xml_sources.append(f"file-{file.filename}")
                logger.debug(f"Processed file {i+1}/{len(files)}: {file.filename}", extra=log_extra)

        # Process body content if not multipart or if no files provided.
        content_type = request.headers.get("content-type", "").lower()
        logger.debug(f"Content-Type: {content_type}", extra=log_extra)

        if not files or "multipart/form-data" not in content_type:
            if "application/json" in content_type:
                logger.debug("Processing JSON payload", extra=log_extra)
                payload = await request.json()
                if isinstance(payload, dict):
                    if "xml" in payload:
                        xml_data.append(payload["xml"])
                        xml_sources.append("json-body-single")
                    elif "files" in payload and isinstance(payload["files"], list):
                        for item in payload["files"]:
                            if isinstance(item, dict) and "xml" in item:
                                xml_data.append(item["xml"])
                                xml_sources.append(f"json-body-multiple-{item.get('filename', 'unnamed')}")
                elif isinstance(payload, list):
                    for item in payload:
                        if isinstance(item, dict) and "xml" in item:
                            xml_data.append(item["xml"])
                            xml_sources.append(f"json-body-array-{item.get('filename', 'unnamed')}")
            elif "application/x-www-form-urlencoded" in content_type:
                logger.debug("Processing form-encoded payload", extra=log_extra)
                form_data = await request.form()
                if "xml" in form_data:
                    xml_data.append(form_data["xml"])
                    xml_sources.append("form-data-single")
                else:
                    xml_keys = [k for k in form_data.keys() if k.startswith("xml")]
                    for key in xml_keys:
                        xml_data.append(form_data[key])
                        xml_sources.append(f"form-data-{key}")
            else:
                logger.debug("Processing raw body as XML", extra=log_extra)
                body = await request.body()
                if body:
                    xml_content = body.decode("utf-8")
                    if xml_content.strip().startswith("<xml>") or xml_content.strip().startswith("<?xml"):
                        xml_data.append(xml_content)
                        xml_sources.append("raw-body-xml")
                    else:
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
                            if not xml_data:
                                xml_data = [xml_content]
                                xml_sources = ["raw-body"]

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

        processed_results = []
        saved_files = []
        json_saved_files = []  # List to track JSON files saved

        for i, (xml_content, source) in enumerate(zip(xml_data, xml_sources)):
            try:
                # Validate XML format
                root = validate_xml_content(xml_content, log_extra)
                logger.info(f"Successfully parsed XML document {i+1}/{len(xml_data)} from {source}", extra=log_extra)

                # Parse XML to JSON using xmltodict and insert into MongoDB
                try:
                    xml_json = xmltodict.parse(xml_content)
                    mongo_result = collection.insert_one(xml_json)
                    mongo_id = str(mongo_result.inserted_id)
                    logger.info(f"Inserted XML JSON to MongoDB with _id: {mongo_id}", extra=log_extra)
                except Exception as e:
                    mongo_id = None
                    logger.error(f"MongoDB insertion error for document {i+1}: {str(e)}", extra=log_extra)

                # Determine a unique filename for saving; generate a custom filename based on headers and source
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename_parts = []
                if meta_headers.get("x_meta_default_filename"):
                    base_name = meta_headers["x_meta_default_filename"].rsplit(".", 1)[0]
                    filename_parts.append(base_name)
                if meta_headers.get("x_meta_game_id"):
                    filename_parts.append(f"game{meta_headers['x_meta_game_id']}")
                if meta_headers.get("x_meta_home_team_id") and meta_headers.get("x_meta_away_team_id"):
                    filename_parts.append(f"match_{meta_headers['x_meta_home_team_id']}vs{meta_headers['x_meta_away_team_id']}")
                if len(xml_data) > 1:
                    filename_parts.append(f"part{i+1}")
                if source.startswith("file-") and not filename_parts:
                    orig_filename = source[5:]
                    base_name = orig_filename.rsplit(".", 1)[0]
                    filename_parts.append(base_name)
                custom_filename = f"{'_'.join(filename_parts)}_{timestamp}.xml" if filename_parts else f"xml_{timestamp}_{i+1}.xml"

                # Save XML file
                file_path = save_xml_file(xml_content, target_dir, custom_filename, meta_headers, log_extra)
                saved_files.append(file_path)

                # Save parsed JSON to a separate JSON file.
                # Generate JSON filename by replacing the .xml extension with .json
                json_filename = custom_filename.replace('.xml', '.json')
                json_file_path = os.path.join(target_dir, json_filename)
                try:
                    with open(json_file_path, 'w', encoding='utf-8') as f:
                        json.dump(xml_json, f, indent=4, cls=JSONEncoder)
                    logger.info(f"JSON file saved: {json_file_path}", extra=log_extra)
                    json_saved_files.append(json_file_path)
                except Exception as e:
                    logger.error(f"Error saving JSON file for document {i+1}: {str(e)}", extra=log_extra)
                    json_file_path = None

                processed_results.append({
                    "index": i,
                    "source": source,
                    "root_tag": root.tag,
                    "child_count": len(list(root)),
                    "saved_path": file_path,
                    "json_saved_path": json_file_path,
                    "mongo_id": mongo_id
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

        return ProcessingResult(
            success=True,
            message=f"Successfully processed and saved {len(xml_data)} XML document(s) to {os.path.basename(target_dir)}",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            xml_count=len(xml_data),
            details={
                "processed": processed_results,
                "headers": meta_headers,
                "saved_files": saved_files,
                "json_saved_files": json_saved_files,
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

# ------------------------------
# API endpoints
# ------------------------------
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
    """Process XML data, save file, parsed JSON in MongoDB, and also save the JSON to a file"""
    log_extra = {"request_id": getattr(request.state, "request_id", "N/A")}
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
        logger.error(f"HTTP exception: {http_ex.detail}", extra=log_extra)
        raise
    except Exception as e:
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
    """Process XML data, save to archive directory, insert parsed JSON in MongoDB, and save JSON to a file"""
    log_extra = {"request_id": getattr(request.state, "request_id", "N/A")}
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
        logger.error(f"HTTP exception: {http_ex.detail}", extra=log_extra)
        raise
    except Exception as e:
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
    """Health check endpoint."""
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
    """API root endpoint with basic information."""
    log_extra = {"request_id": getattr(request.state, "request_id", "N/A")}
    logger.info("Root endpoint accessed", extra=log_extra)
    return {
        "name": "XML Request Handler API",
        "version": "1.0.0",
        "endpoints": {
            "/api/xml": "POST - Process XML requests (supports file uploads, raw XML, and JSON payloads)",
            "/api/xml_secondary": "POST - Process XML requests to archive (also supports file uploads, raw XML, and JSON payloads)",
            "/health": "GET - Health check"
        },
        "upload_directory": os.path.abspath(UPLOAD_DIR),
        "xml_files_stored": len([f for f in os.listdir(UPLOAD_DIR) if f.endswith('.xml')]) if os.path.exists(UPLOAD_DIR) else 0,
        "request_id": log_extra["request_id"]
    }

if __name__ == "__main__":
    logger.info("Starting XML Request Handler API")
    logger.info(f"Files will be saved to: {os.path.abspath(UPLOAD_DIR)}")
    uvicorn.run("test:app", host="127.0.0.1", port=8000, reload=True)
