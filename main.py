import modal
import torch
import json
import os
import requests
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# 1. Initialize FastAPI with global CORS
web_app = FastAPI()
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app = modal.App("ayurparam-service")

# 2. Optimized Image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==1.24.3",
        "transformers==4.40.0",
        "torch==2.1.0",
        "accelerate==0.27.0",
        "fastapi[standard]==0.109.0",
        "huggingface_hub==0.20.0",
        "requests==2.31.0"
    )
)

volume = modal.Volume.from_name("ayurparam-models-final", create_if_missing=True)


@app.cls(
    image=image,
    gpu="T4",
    timeout=1200,
    min_containers=1,
    volumes={"/cache": volume},
    secrets=[
        modal.Secret.from_name("my-umls-secret"),
        modal.Secret.from_name("huggingface-secret")
    ]
)
class AyurEngine:
    @modal.enter()
    def setup(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        from huggingface_hub import login

        print("🔧 Starting setup...")
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token: login(token=hf_token)

        # ✅ TIER 1: ClinicalBERT NER
        self.ner = pipeline(
            "ner",
            model="emilyalsentzer/Bio_ClinicalBERT",
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1
        )

        # ✅ TIER 2: AyurParam LLM
        model_id = "bharatgenai/AyurParam"
        # Fix for 'untagged enum' error: use_fast=False
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir="/cache/models",
            trust_remote_code=True,
            use_fast=False
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir="/cache/models",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.umls_api_key = os.environ.get("UMLS_API_KEY", "")
        print("🚀 Systems Online!")

    def _convert_to_native(self, obj):
        import numpy as np
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)): return obj.item()
        if isinstance(obj, dict): return {k: self._convert_to_native(v) for k, v in obj.items()}
        if isinstance(obj, list): return [self._convert_to_native(i) for i in obj]
        return obj

    @modal.method()
    def get_treatment(self, user_input: str):
        import numpy as np  # Ensure numpy is available for conversion
        print(f"📥 Query: {user_input}")

        entities = []
        keyword = user_input
        try:
            raw_res = self.ner(user_input)
            # 🛑 CRITICAL FIX: Convert NumPy math types to standard Python numbers
            # This prevents the 'numpy.float32 is not iterable' 500 error
            for ent in raw_res:
                entities.append({
                    "word": str(ent["word"]),
                    "score": float(ent["score"]),  # Convert float32 to float
                    "entity_group": str(ent.get("entity_group", ""))
                })

            if entities:
                keyword = sorted(entities, key=lambda x: x['score'], reverse=True)[0]['word']
        except Exception as e:
            print(f"⚠️ NER Error: {e}")

        # Standard UMLS Mapping
        snomed_code = "N/A"
        if self.umls_api_key:
            try:
                url = "https://uts-ws.nlm.nih.gov/rest/search/current"
                params = {"string": keyword, "apiKey": self.umls_api_key, "sabs": "SNOMEDCT_US", "returnIdType": "code"}
                r = requests.get(url, params=params, timeout=10)
                if r.status_code == 200:
                    results = r.json().get("result", {}).get("results", [])
                    if results: snomed_code = results[0].get("ui", "N/A")
            except:
                pass

        treatment = self._generate_treatment(keyword, snomed_code)

        return {
            "input_text": user_input,
            "clinical_entities": entities if entities else [{"word": keyword, "score": 1.0}],
            "umls_cui": snomed_code,
            "snomed_code": snomed_code,
            "results": [{
                "ayurveda_term": treatment.get("condition_name", keyword),
                "snomed_code": snomed_code,
                "treatment_info": treatment
            }]
        }

    def _generate_treatment(self, condition, code):
        prompt = f"<user> Provide Ayurvedic treatment for {condition} (Code: {code}) in valid JSON. </user> <assistant>"
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
            output = self.model.generate(**inputs, max_new_tokens=1500)
            raw = self.tokenizer.decode(output[0], skip_special_tokens=True).split("<assistant>")[-1].strip()
            start, end = raw.find('{'), raw.rfind('}') + 1
            return json.loads(raw[start:end])
        except:
            return {"condition_name": condition, "brief_description": "Analysis complete."}


@web_app.post("/")
async def api_handler(request: Request):
    try:
        body = await request.json()
        user_text = body.get("text", "")
        engine = AyurEngine()
        # ✅ Use .remote() correctly without passing call_timeout as a function argument
        return engine.get_treatment.remote(user_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    return web_app
