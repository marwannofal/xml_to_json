from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import xmltodict as xml
import json
from pymongo import MongoClient as DB
from bson import ObjectId

# Custom JSON encoder that converts ObjectId to strings
class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return super().default(o)

class CustomJSONResponse(JSONResponse):
    def render(self, content: any) -> bytes:
        return json.dumps(content, cls=JSONEncoder, indent=4).encode("utf-8")

app = FastAPI()

# Connect to MongoDB
client = DB("mongodb://localhost:27017/")
db = client["xml_to_json"]
collection = db["xml_to_json"]

@app.post("/convert/")
async def convert_xml_to_json(file: UploadFile = File(...)):
    if file.content_type not in ["application/xml", "text/xml"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a valid XML file."
        )

    connects = await file.read()

    try:
        # Decode and parse the XML file into a Python dictionary
        xml_string = connects.decode('utf-8')
        python_dict = xml.parse(xml_string)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse XML: {str(e)}"
        )

    result = collection.insert_one(python_dict)

    response_data = {
        "inserted_id": str(result.inserted_id),
        "data": python_dict
    }

    # For debugging: print JSON string using our custom encoder
    debug_json_string = json.dumps(python_dict, indent=4, cls=JSONEncoder)
    print(debug_json_string)

    return CustomJSONResponse(content=response_data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
