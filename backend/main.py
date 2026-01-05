from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# SOLO importar el router, no lifespan
from api.endpoints import router as api_router

app = FastAPI(
    title="NouxTrends API",
    description="Motor de predicciones rápidas para series temporales",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "¡NouxTrends API funcionando!"}

if __name__ == "__main__":
    import uvicorn
    # CAMBIA esta línea - pasar la app como string para reload
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)