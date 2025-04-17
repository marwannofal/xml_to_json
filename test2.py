from fastapi import FastAPI, Request, HTTPException, Header, File, UploadFile, APIRouter
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Union
import os, json, time, uuid, traceback, xml.etree.ElementTree as ET
import xmltodict
from pymongo import MongoClient, errors as mongo_errors
from bson import ObjectId
import logging, logging.handlers
from datetime import datetime
import uvicorn


"""Configuration / Utilities"""

class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return super().default(o)

class LoggerConfig:
    def __init__(self, name: str = "xml_api", log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        # console
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # file
        fh = logging.handlers.RotatingFileHandler(
            filename=f"{log_dir}/xml_api.log", maxBytes=10_485_760, backupCount=10
        )
        fh.setLevel(logging.DEBUG)
        fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [RequestID: %(request_id)s] - %(message)s')
        ch.setFormatter(fmt)
        fh.setFormatter(fmt)
        for h in (ch, fh):
            h.addFilter(self.RequestIDFilter())
            self.logger.addHandler(h)

    class RequestIDFilter(logging.Filter):
        def filter(self, record):
            if not hasattr(record, "request_id"):
                record.request_id = "N/A"
            return True

class MongoManager:
    def __init__(self, uri: str, db_name: str, logger: logging.Logger):
        self.logger = logger
        self.uri = uri
        self.db_name = db_name
        self.client = None
        self.db = None
        self.collection = None
        self.connect()
        
    def connect(self) -> bool:
        """Establish connection to MongoDB with retry logic and health check"""
        try:
            # Set a reasonable timeout for connection attempts
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=5000)
            # Validate connection is working with a ping
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            self.collection = self.db["xml_to_json"]
            self.logger.info("MongoDB connected successfully")
            return True
        except mongo_errors.ConnectionFailure as e:
            self.logger.error(f"MongoDB connection failed: {e}")
            self.client = None
            return False
        except Exception as e:
            self.logger.error(f"MongoDB setup error: {e}")
            self.client = None
            return False

    def is_connected(self) -> bool:
        """Check if MongoDB connection is active"""
        if not self.client:
            return False
        try:
            # Quick ping to verify connection is still alive
            self.client.admin.command('ping')
            return True
        except Exception:
            return False

    def insert(self, data: dict) -> Optional[str]:
        """Insert document with connection verification and retry"""
        if not self.is_connected() and not self.connect():
            self.logger.warning("MongoDB unavailable. Skipping database insertion.")
            return None

        try:
            res = self.collection.insert_one(data)
            self.logger.info(f"Inserted record to MongoDB: {res.inserted_id}")
            return str(res.inserted_id)
        except mongo_errors.AutoReconnect:
            # Try reconnecting once on connection issues
            self.logger.warning("MongoDB connection lost, attempting reconnect")
            if self.connect():
                try:
                    res = self.collection.insert_one(data)
                    return str(res.inserted_id)
                except Exception as e:
                    self.logger.error(f"MongoDB insertion retry failed: {e}")
            return None
        except Exception as e:
            self.logger.error(f"MongoDB insertion error: {e}")
            return None

class FileManager:
    def __init__(self, json_dir: str, logger: logging.Logger):
        self.json_dir = json_dir
        self.logger = logger
        os.makedirs(json_dir, exist_ok=True)

    def _get_timestamped_filename(self, prefix: str) -> str:
        """Generate a unique filename with current timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        return f"{prefix}_{timestamp}_{unique_id}.json"
    
    def save_json(self, data: dict, request_id: str, prefix: str = "doc") -> str:
        """Save JSON data to file with timestamp in filename"""
        filename = self._get_timestamped_filename(prefix)
        filepath = os.path.join(self.json_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, cls=JSONEncoder)
            self.logger.info(f"Saved JSON: {filepath}", extra={"request_id": request_id})
            return filepath
        except Exception as e:
            self.logger.error(f"Error saving JSON: {e}", extra={"request_id": request_id})
            raise IOError(f"Failed to save JSON: {e}")

"""Models"""

class XMLData(BaseModel):
    xml: str
    filename: Optional[str] = None

class ProcessingResult(BaseModel):
    success: bool
    message: str
    timestamp: str
    xml_count: int
    details: Optional[Dict] = None


"""Core Processor"""

class XMLProcessor:
    def __init__(self, file_manager: FileManager, mongo: MongoManager, logger: logging.Logger):
        self.fm = file_manager
        self.mongo = mongo
        self.logger = logger

    def validate(self, xml_content: str, request_id: str) -> ET.Element:
        try:
            return ET.fromstring(xml_content)
        except ET.ParseError as e:
            self.logger.error(f"ParseError: {e}", extra={"request_id": request_id})
            raise HTTPException(status_code=400, detail=f"Invalid XML: {str(e)}")

    async def process(self, xml_list: List[str], request_id: str) -> ProcessingResult:
        results = []
        saved_jsons = []
        
        for idx, xml_str in enumerate(xml_list):
            try:
                # Validate XML structure
                root = self.validate(xml_str, request_id)
                self.logger.info(f"XML validated (root={root.tag})", extra={"request_id": request_id})
                
                # Convert XML to JSON
                json_data = xmltodict.parse(xml_str)
                
                # Store in MongoDB if available
                mongo_id = self.mongo.insert(json_data)
                if not mongo_id:
                    self.logger.warning("MongoDB storage skipped", extra={"request_id": request_id})
                
                # Save JSON to file with timestamp in name
                prefix = f"doc_{idx+1}"
                json_path = self.fm.save_json(json_data, request_id, prefix)
                saved_jsons.append(json_path)
                
                results.append({
                    'index': idx,
                    'root': root.tag,
                    'mongo_id': mongo_id,
                    'json_path': json_path
                })
                
            except Exception as e:
                self.logger.error(f"Error processing XML document {idx}: {e}", extra={"request_id": request_id})
                results.append({
                    'index': idx,
                    'error': str(e),
                    'status': 'failed'
                })

        return ProcessingResult(
            success=True,
            message=f"Processed {len(xml_list)} document(s)",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            xml_count=len(xml_list),
            details={'processed': results, 'saved_jsons': saved_jsons}
        )

"""API Application"""

class XMLAPIApp:
    def __init__(self):
        # config dirs
        self.json_dir = 'uploads'
        # setup logger
        self.logger = LoggerConfig().logger
        # managers
        mongo_uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/")
        mongo_db = os.environ.get("MONGODB_DB", "xml_to_json")
        self.mongo = MongoManager(mongo_uri, mongo_db, self.logger)
        self.fm = FileManager(self.json_dir, self.logger)
        self.processor = XMLProcessor(self.fm, self.mongo, self.logger)
        # FastAPI
        self.app = FastAPI(
            title="XML to JSON Processor API",
            version="1.1.0",
            description="Processes XML data and stores as JSON with MongoDB integration"
        )
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
            allow_credentials=True
        )
        self.router = APIRouter()
        self._register_routes()
        self.app.include_router(self.router)
        self.app.middleware("http")(self.log_requests)
        self.app.add_exception_handler(Exception, self.global_exception_handler)

    async def log_requests(self, request: Request, call_next):
        rid = str(uuid.uuid4())
        request.state.request_id = rid
        start = time.time()
        self.logger.info(f"Start {request.method} {request.url.path}", extra={"request_id": rid})
        try:
            resp = await call_next(request)
            duration = time.time() - start
            self.logger.info(
                f"Completed {request.method} {request.url.path} - {resp.status_code} in {duration:.3f}s",
                extra={"request_id": rid}
            )
            resp.headers['X-Request-ID'] = rid
            return resp
        except Exception as e:
            self.logger.error(f"Error: {e}", extra={"request_id": rid})
            raise

    async def global_exception_handler(self, request: Request, exc: Exception):
        rid = getattr(request.state, 'request_id', 'N/A')
        if isinstance(exc, HTTPException):
            self.logger.warning(f"HTTPException: {exc.detail}", extra={"request_id": rid})
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail, "request_id": rid})
        self.logger.error(f"Unhandled: {exc}\n{traceback.format_exc()}", extra={"request_id": rid})
        return JSONResponse(
            status_code=500, 
            content={
                "detail": "Internal server error", 
                "request_id": rid,
                "error": str(exc)
            }
        )

    def _register_routes(self):
        @self.router.post("/api/xml", response_model=ProcessingResult)
        async def process_xml(
            request: Request,
            files: Optional[List[UploadFile]] = File(None)
        ):
            rid = request.state.request_id
            xmls = []
            
            # Handle XML from files or raw body
            if files:
                for f in files:
                    if f.filename.lower().endswith('.xml'):
                        content = await f.read()
                        xmls.append(content.decode('utf-8', errors='replace'))
            else:
                # Process raw XML body
                try:
                    body = await request.body()
                    txt = body.decode('utf-8', errors='replace')
                    if txt.strip().startswith('<'):
                        xmls.append(txt)
                except Exception as e:
                    self.logger.error(f"Error reading request body: {e}", extra={"request_id": rid})
                    raise HTTPException(status_code=400, detail="Invalid request body")
            
            # Validate we have XML to process
            if not xmls:
                raise HTTPException(status_code=400, detail="No valid XML content found")
                
            # Process the XML data
            return await self.processor.process(xmls, rid)

        @self.router.get("/health")
        async def health(request: Request):
            rid = request.state.request_id
            mongo_status = "connected" if self.mongo.is_connected() else "disconnected"
            json_count = len([f for f in os.listdir(self.json_dir) if f.endswith('.json')])
            
            return {
                "status": "healthy", 
                "mongo": mongo_status,
                "json_files": json_count,
                "request_id": rid,
                "timestamp": datetime.now().isoformat()
            }

        @self.router.get("/")
        async def root(request: Request):
            rid = request.state.request_id
            return {
                "name": "XML to JSON Processor API", 
                "version": "1.1.0", 
                "request_id": rid,
                "mongo_status": "connected" if self.mongo.is_connected() else "disconnected"
            }

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        uvicorn.run(self.app, host=host, port=port, reload=True)

"""Run the application"""
app = XMLAPIApp().app

if __name__ == "__main__":
    # Start server if script is run directly
    port = int(os.environ.get("PORT", 8000))
    XMLAPIApp().run(port=port)