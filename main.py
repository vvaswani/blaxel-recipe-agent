# app_gemini.py
import os
import json
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import uvicorn
import google.generativeai as genai
import blaxel.core

app = FastAPI(title="Recipe Agent (Gemini)")

# ---- Pydantic I/O models ----

class RecipeIngredient(BaseModel):
    item: str
    quantity: Optional[str] = None

class RecipeResponse(BaseModel):
    title: str
    description: str
    servings: int
    ingredients: List[RecipeIngredient]
    steps: List[str]
    tips: Optional[List[str]] = None

class RecipeRequest(BaseModel):
    ingredients: List[str] = Field(..., min_length=3, description="At least 3 ingredients")
    style: str
    servings: Optional[int] = Field(default=2, gt=0)

    @field_validator("ingredients")
    @classmethod
    def strip_ingredients(cls, v):
        cleaned = [s.strip() for s in v if s.strip()]
        if len(cleaned) < 3:
            raise ValueError("Provide at least 3 non-empty ingredients.")
        return cleaned

# ---- Gemini-compatible schema (keep it minimal) ----
RECIPE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "description": {"type": "string"},
        "servings": {"type": "integer"},
        "ingredients": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item": {"type": "string"},
                    "quantity": {"type": "string"}  # optional by omission from "required"
                },
                "required": ["item"]
            }
        },
        "steps": {
            "type": "array",
            "items": {"type": "string"}
        },
        "tips": {
            "type": "array",
            "items": {"type": "string"}  # optional
        }
    },
    "required": ["title", "description", "servings", "ingredients", "steps"]
}

def _configure_gemini():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY (or GOOGLE_API_KEY) is not set.")
    genai.configure(api_key=api_key)

def _build_model():
    system_instruction = (
        "You are a culinary assistant. Create a clear, complete recipe that uses ALL of the "
        "provided ingredients (you may add pantry basics like oil, salt, pepper). "
        "Write concise, professional steps suitable for home cooks. "
        "Return only valid JSON that matches the provided schema. "
        "Aim for at least 3 ingredients entries and 3 steps."
    )
    generation_config = {
        "temperature": 0.7,
        "response_mime_type": "application/json",
        "response_schema": RECIPE_JSON_SCHEMA,
    }
    return genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=system_instruction,
        generation_config=generation_config,
    )

@app.post("/recipe", response_model=RecipeResponse)
def create_recipe(payload: RecipeRequest):
    _configure_gemini()
    model = _build_model()

    user_prompt = (
        f"Style of cooking: {payload.style}\n"
        f"Servings: {payload.servings}\n"
        f"Must-use ingredients: {', '.join(payload.ingredients)}"
    )

    try:
        result = model.generate_content(user_prompt)
        raw = result.text  # JSON string
        data = json.loads(raw)

        # Optional: normalize empties to None
        for ing in data.get("ingredients", []):
            if isinstance(ing.get("quantity"), str) and not ing["quantity"].strip():
                ing["quantity"] = None
        if "tips" in data and isinstance(data["tips"], list) and not data["tips"]:
            data["tips"] = None

        # Server-side invariants that Gemini's schema can't enforce:
        if not isinstance(data.get("ingredients"), list) or len(data["ingredients"]) < 3:
            raise HTTPException(status_code=502, detail="Model returned fewer than 3 ingredients.")
        if not isinstance(data.get("steps"), list) or len(data["steps"]) < 3:
            raise HTTPException(status_code=502, detail="Model returned fewer than 3 steps.")

        return RecipeResponse(**data)

    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="Upstream returned invalid JSON.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini error: {str(e)}")

if __name__ == "__main__":
    host = os.getenv("BL_SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("BL_SERVER_PORT", "80"))

    uvicorn.run("main:app", host=host, port=port, reload=True)
