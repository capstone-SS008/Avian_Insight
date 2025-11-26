from fastapi import FastAPI, UploadFile, File
from utils.image_inference import predict_bird_image
from utils.sound_inference import predict_bird_sound
from utils.separation_inference import separate_and_classify

app = FastAPI(title="Bird Classification API")

@app.get("/")
def root():
    return {"message": "Bird Classification Backend Running"}

# -------------------------
# 1. Image Classification
# -------------------------
@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    result = await predict_bird_image(file)
    print(result)
    return {"prediction": result}

# -------------------------
# 2. Sound Classification
# -------------------------
@app.post("/predict_sound")
async def predict_sound(file: UploadFile = File(...)):
    result = await predict_bird_sound(file)
    print(result)
    return {"prediction": result}

# ---------------------------------------
# 3. Bird Sound Separation + Classification
# ---------------------------------------
@app.post("/separater")
async def separate_and_identify_api(file: UploadFile = File(...)):
    print("Separation and Classification")
    result = await  separate_and_classify(file)
    print(result)
    return {"prediction": result}
