from fastapi import FastAPI, Request, HTTPException, Header, File, UploadFile, APIRouter
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import os, json, time, uuid, traceback, xml.etree.ElementTree as ET
import xmltodict
from pymongo import MongoClient
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
        try:
            self.client = MongoClient(uri)
            self.db = self.client[db_name]
            self.collection = self.db["xml_to_json"]
            self.logger.info("MongoDB connected successfully.")
        except Exception as e:
            self.logger.warning(f"MongoDB connection failed: {e}")
            self.client = None  # Set to None if MongoDB isn't available

    def insert(self, data: dict) -> Optional[str]:
        if not self.client:
            self.logger.warning("MongoDB client is unavailable. Skipping database insertion.")
            return None  # Skip MongoDB insertion if the client is unavailable

        try:
            res = self.collection.insert_one(data)
            self.logger.info(f"Inserted record to MongoDB: {res.inserted_id}")
            return str(res.inserted_id)
        except Exception as e:
            self.logger.error(f"MongoDB insertion error: {e}")
            return None

class FileManager:
    def __init__(self, upload_dir: str, archive_dir: str, logger: logging.Logger):
        self.upload_dir = upload_dir
        self.archive_dir = archive_dir
        self.logger = logger
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(archive_dir, exist_ok=True)

    def _unique_filepath(self, target_dir: str, filename: str, request_id: str) -> str:
        base = os.path.basename(filename)
        path = os.path.join(target_dir, base)
        if os.path.exists(path):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            name, ext = os.path.splitext(base)
            path = os.path.join(target_dir, f"{name}_{ts}{ext}")
            self.logger.info(f"Filename exists, renamed to {path}", extra={"request_id": request_id})
        return path

    def save_xml(self, content: str, request_id: str, target: str, filename: Optional[str] = None) -> str:
        target_dir = self.upload_dir if target == "active" else self.archive_dir
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = filename or f"xml_{ts}_{uuid.uuid4().hex[:8]}.xml"
        if not fname.lower().endswith('.xml'):
            fname += '.xml'
        path = self._unique_filepath(target_dir, fname, request_id)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        self.logger.info(f"Saved XML: {path}", extra={"request_id": request_id})
        return path

    def save_json(self, data: dict, xml_path: str, request_id: str) -> Optional[str]:
        json_path = xml_path.replace('.xml', '.json')
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, cls=JSONEncoder)
            self.logger.info(f"Saved JSON: {json_path}", extra={"request_id": request_id})
            return json_path
        except Exception as e:
            self.logger.error(f"Error saving JSON: {e}", extra={"request_id": request_id})
            return None

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
            raise HTTPException(status_code=400, detail=str(e))

    async def process(self, xml_list: List[str], request_id: str, target: str) -> ProcessingResult:
        results = []
        saved_xmls = []
        saved_jsons = []
        for idx, xml_str in enumerate(xml_list):
            root = self.validate(xml_str, request_id)
            self.logger.info(f"XML validated (root={root.tag})", extra={"request_id": request_id})
            data = xmltodict.parse(xml_str)

            # Skip MongoDB insertion if MongoDB is unavailable
            mongo_id = None
            if self.mongo.client:
                mongo_id = self.mongo.insert(data)

            fname = f"doc_{idx+1}.xml"
            xml_path = self.fm.save_xml(xml_str, request_id, target, filename=fname)
            json_path = self.fm.save_json(data, xml_path, request_id)

            results.append({
                'index': idx,
                'root': root.tag,
                'mongo_id': mongo_id,
                'xml_path': xml_path,
                'json_path': json_path
            })
            saved_xmls.append(xml_path)
            saved_jsons.append(json_path)

        return ProcessingResult(
            success=True,
            message=f"Processed {len(xml_list)} document(s)",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            xml_count=len(xml_list),
            details={'processed': results, 'saved_xmls': saved_xmls, 'saved_jsons': saved_jsons}
        )

"""API Application"""

class XMLAPIApp:
    def __init__(self):
        # config dirs
        self.upload_dir = 'uploads'
        self.archive_dir = 'archive'
        # setup logger
        self.logger = LoggerConfig().logger
        # managers
        self.mongo = MongoManager("mongodb://localhost:27017/", "xml_to_json", self.logger)
        self.fm = FileManager(self.upload_dir, self.archive_dir, self.logger)
        self.processor = XMLProcessor(self.fm, self.mongo, self.logger)
        # FastAPI
        self.app = FastAPI(
            title="XML Request Handler API",
            version="1.0.0"
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
        self.logger.error(f"Unhandled: {exc}", extra={"request_id": rid})
        return JSONResponse(status_code=500, content={"detail": "Internal error", "request_id": rid})

    def _register_routes(self):
        @self.router.post("/api/xml", response_model=ProcessingResult)
        async def xml_endpoint(
            request: Request,
            files: Optional[List[UploadFile]] = File(None)
        ):
            rid = request.state.request_id
            xmls = []
            # handle files or raw body
            if files:
                for f in files:
                    if f.filename.lower().endswith('.xml'):
                        content = await f.read()
                        xmls.append(content.decode())
            else:
                body = await request.body()
                txt = body.decode()
                xmls = [txt] if txt.strip().startswith('<') else []
            if not xmls:
                raise HTTPException(status_code=400, detail="No XML found")
            return await self.processor.process(xmls, rid, target='active')

        @self.router.post("/api/xml_secondary", response_model=ProcessingResult)
        async def archive_endpoint(
            request: Request,
            files: Optional[List[UploadFile]] = File(None)
        ):
            rid = request.state.request_id
            xmls = []
            if files:
                for f in files:
                    if f.filename.lower().endswith('.xml'):
                        content = await f.read()
                        xmls.append(content.decode())
            else:
                body = await request.body()
                txt = body.decode()
                xmls = [txt] if txt.strip().startswith('<') else []
            if not xmls:
                raise HTTPException(status_code=400, detail="No XML found")
            return await self.processor.process(xmls, rid, target='archive')

        @self.router.get("/health")
        async def health(request: Request):
            rid = request.state.request_id
            count_up = len([f for f in os.listdir(self.upload_dir) if f.endswith('.xml')])
            count_arc = len([f for f in os.listdir(self.archive_dir) if f.endswith('.xml')])
            return {"status": "healthy", "uploads": count_up, "archive": count_arc, "request_id": rid}

        @self.router.get("/")
        async def root(request: Request):
            rid = request.state.request_id
            return {"name": "XML API", "version": "1.0.0", "request_id": rid}

    def run(self):
        uvicorn.run(self.app, host="0.0.0.0", port=8000, reload=True)

"""Run the application"""
app = XMLAPIApp().app
