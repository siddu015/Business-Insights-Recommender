#!/usr/bin/env python3

import os
import json
import ast
import re
from typing import List, Dict, Any, Tuple

import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from peft import PeftModel


DATA_DIR = "/Users/siddu/Developer/Business-Insights-Recommender"
ENGINEERED_CSV = os.path.join(DATA_DIR, "data/processed/engineered_business_data.csv")
MODEL_DIR = os.path.join(DATA_DIR, "models/flan_t5_lora")
MODEL_NAME = "google/flan-t5-small"


def detect_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_prompt(summary: Dict, few_shot_examples: List[Tuple[Dict[str, Any], Dict[str, Any]]] | None = None) -> str:
    header = (
        "Given the following business profile, produce personalized recommendations as structured JSON "
        "with keys: business_id, recommendations (3-5 items), confidence_score (0-100)."
    )
    skeleton = {
        "business_id": summary.get("business_id"),
        "recommendations": ["..."],
        "confidence_score": 70,
    }
    few_shot_str = ""
    if few_shot_examples:
        lines = ["Examples (follow this exact JSON style):"]
        for (ex_profile, ex_output) in few_shot_examples[:2]:
            lines.append(
                f"Profile: {json.dumps(ex_profile, ensure_ascii=False)}\nOutput: {json.dumps(ex_output, ensure_ascii=False)}"
            )
        few_shot_str = "\n" + "\n\n".join(lines) + "\n"

    return (
        f"{header} Return ONLY a single JSON object matching this schema, with no extra text: "
        f"{json.dumps(skeleton, ensure_ascii=False)}. "
        f"Use 3-5 concise, actionable recommendations. Do not repeat instructions.\n"
        f"{few_shot_str}"
        f"Business Profile:\n{json.dumps(summary, ensure_ascii=False)}"
    )


def _normalize_recommendation_lines(text: str) -> List[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    cleaned: List[str] = []
    for ln in lines:
        ln = re.sub(r"^[\-\*•\d\)\(\.\s]+", "", ln)
        lowered = ln.lower()
        if (
            ln
            and "business_id" not in lowered
            and "confidence" not in lowered
            and "recommendations (3-5" not in lowered
            and "keys:" not in lowered
            and "return only" not in lowered
            and "business profile" not in lowered
        ):
            cleaned.append(ln)
    unique: List[str] = []
    for rec in cleaned:
        if rec not in unique:
            unique.append(rec)
        if len(unique) >= 5:
            break
    return unique


def _fallback_recommendations(summary: Dict[str, Any]) -> List[str]:
    recs: List[str] = []
    digital_readiness = float(str(summary.get("digital_readiness", "0")).replace("%", ""))
    growth_potential = float(str(summary.get("revenue_growth_potential", "0")).replace("%", ""))
    competition = int(summary.get("competition_density", 0) or 0)
    industry_trend = str(summary.get("industry_trend", "")).lower()
    young_pct_str = str(summary.get("young_customers_pct", "0%"))
    try:
        young_pct = float(young_pct_str.rstrip("%"))
    except Exception:
        young_pct = 0.0

    if digital_readiness < 5:
        recs.append("Invest in website SEO and social profiles to improve digital readiness.")
    if competition >= 7:
        recs.append("Differentiate offering and run targeted promos to defend share in a dense market.")
    if young_pct >= 40:
        recs.append("Launch short-form social content and creator partnerships to reach younger customers.")
    if "declin" in industry_trend:
        recs.append("Explore adjacent products/services or new geographies to offset industry headwinds.")
    if growth_potential >= 5:
        recs.append("Allocate budget to high-ROAS channels and expand top-selling SKUs.")

    if not recs:
        recs = [
            "Audit funnel from awareness to conversion and fix top 2 drop-offs.",
            "Test 2 new acquisition channels and 1 retention lever this quarter.",
        ]
    return recs[:5]


def _coerce_schema(obj: Any, summary: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "business_id": summary.get("business_id"),
        "recommendations": [],
        "confidence_score": 0,
    }
    if isinstance(obj, dict):
        biz_id = obj.get("business_id") or obj.get("id") or out["business_id"]
        out["business_id"] = biz_id
        recs = obj.get("recommendations") or obj.get("recs") or obj.get("actions")
        if isinstance(recs, str):
            recs = _normalize_recommendation_lines(recs)
        elif isinstance(recs, list):
            recs = [str(x).strip() for x in recs if str(x).strip()]
        else:
            recs = []
        out["recommendations"] = recs[:5]

        conf = obj.get("confidence_score") or obj.get("confidence") or 0
        try:
            conf = int(float(conf))
        except Exception:
            conf = 0
        out["confidence_score"] = max(0, min(100, conf))

    return out


def _parse_model_json(text: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    print(f"\n{'='*80}")
    print(f"Business ID: {summary.get('business_id')}")
    print(f"Industry: {summary.get('industry')}")
    print(f"\nRaw model output (first 500 chars):\n{repr(text[:500])}")
    if len(text) > 500:
        print(f"... [truncated, total length: {len(text)} chars]")
    print(f"{'='*80}\n")
    

    text = text.strip()
    if text.startswith('"') and not text.startswith('{'):
        text = '{' + text
        print("Added missing opening brace")
    if not text.endswith('}') and '"confidence_score"' in text:
        text = text + '}'
        print("Added missing closing brace")
    
    try:
        obj = json.loads(text)
        result = _coerce_schema(obj, summary)
        print(f"✓ Parsed as JSON: {len(result['recommendations'])} recommendations")
        return result
    except Exception as e:
        print(f"✗ JSON parse failed: {e}")
    
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        print(f"Trying to extract JSON from braces: {candidate[:100]}...")
        try:
            obj = json.loads(candidate)
            result = _coerce_schema(obj, summary)
            print(f"✓ Parsed extracted JSON: {len(result['recommendations'])} recommendations")
            return result
        except Exception as e:
            print(f"✗ Extracted JSON parse failed: {e}")

            try:
                obj = ast.literal_eval(candidate)
                result = _coerce_schema(obj, summary)
                print(f"✓ Parsed as Python literal: {len(result['recommendations'])} recommendations")
                return result
            except Exception as e2:
                print(f"✗ Python literal parse failed: {e2}")
    
    rec_pattern = r'"recommendations":\s*\[(.*?)(?:\]|$)'
    match = re.search(rec_pattern, text, re.DOTALL)
    if match:
        recs_text = match.group(1)
        recs = re.findall(r'"([^"]{15,})"', recs_text)
        
        if len(recs) < 3:
            incomplete = re.findall(r'"([^"]{15,})', recs_text)
            recs.extend(incomplete)
        
        if recs:
            cleaned = []
            seen = set()
            for rec in recs:
                rec = rec.strip().rstrip('.,;:')
                if (rec and len(rec) > 15 and 
                    'business_id' not in rec.lower() and 
                    'confidence' not in rec.lower() and
                    'recommendations' not in rec.lower() and
                    rec not in seen):
                    cleaned.append(rec)
                    seen.add(rec)
                if len(cleaned) >= 5:
                    break
            
            if cleaned:
                print(f"✓ Extracted {len(cleaned)} recommendations from malformed JSON")
                return {
                    "business_id": summary.get("business_id"),
                    "recommendations": cleaned,
                    "confidence_score": 75,
                }

    recs = _normalize_recommendation_lines(text)
    if recs:
        print(f"✓ Extracted {len(recs)} recommendations from text lines")
        return {
            "business_id": summary.get("business_id"),
            "recommendations": recs[:5],
            "confidence_score": 50,
        }
    
    print(f"✗ Could not parse any recommendations from model output")
    return {
        "business_id": summary.get("business_id"),
        "recommendations": [],
        "confidence_score": 0,
    }


def _load_training_examples_by_industry(df: pd.DataFrame, labels_path: str) -> Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]]:
    id_to_row = {str(r.Business_ID): r for _, r in df.iterrows()}
    industry_to_examples: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]] = {}
    try:
        with open(labels_path, "r") as f:
            for line in f:
                ex = json.loads(line)
                biz_id = str(ex.get("business_id"))
                row = id_to_row.get(biz_id)
                if row is None:
                    continue
                prof = create_business_summary(row)
                out = {
                    "business_id": biz_id,
                    "recommendations": ex.get("recommendations", []),
                    "confidence_score": ex.get("confidence_score", 70),
                }
                industry = str(prof.get("industry", "Unknown"))
                industry_to_examples.setdefault(industry, []).append((prof, out))
    except Exception:
        pass
    return industry_to_examples


def create_business_summary(row: pd.Series) -> Dict:
    regulatory = str(row.get("Regulatory_Constraints", ""))
    ig_comments = str(row.get("IG_Comments", ""))
    if len(regulatory) > 150:
        regulatory = regulatory[:150] + "..."
    if len(ig_comments) > 150:
        ig_comments = ig_comments[:150] + "..."
    return {
        "business_id": row["Business_ID"],
        "industry": row["Industry"],
        "revenue": f"${int(row['Revenue_Last_12M']):,}",
        "location": row["Location"],
        "growth_stage": row["Growth_Stage"],
        "online_presence": row["Online_Presence"],
        "digital_payment": row["Digital_Payment_Adoption"],
        "young_customers_pct": f"{float(row['Pct_Young_Customers']):.1f}%",
        "market_spending_growth": f"{row['Consumer_Spending_Growth_YoY']}%",
        "competition_density": int(row["Competition_Density_Score"]),
        "industry_trend": row["Industry_Growth_Trend"],
        "revenue_growth_potential": f"{float(row['Revenue_Growth_Potential']):.1f}",
        "risk_score": f"{float(row['Risk_Score']):.1f}",
        "digital_readiness": f"{float(row['Digital_Readiness']):.1f}",
        "regulatory_constraints": regulatory,
        "customer_feedback": ig_comments,
        "main_challenges": row.get("Main_Challenges", ""),
    }


def generate_for_businesses(df: pd.DataFrame, tokenizer, model, device, limit: int = 5) -> List[Dict]:
    outputs: List[Dict] = []        
    subset = df.head(limit)
    print(f"\nGenerating recommendations for {len(subset)} businesses...\n")
    
    labels_jsonl = os.path.join(DATA_DIR, "data/processed/training_labels.jsonl")
    labels_json = os.path.join(DATA_DIR, "data/processed/training_labels.json")
    labels_path = labels_jsonl if os.path.exists(labels_jsonl) else labels_json
    industry_to_examples = _load_training_examples_by_industry(df, labels_path)
    
    for idx, (_, row) in enumerate(subset.iterrows(), 1):
        print(f"\n{'#'*80}")
        print(f"Processing {idx}/{len(subset)}")
        print(f"{'#'*80}")
        
        summary = create_business_summary(row)
        prompt = build_prompt(summary, few_shot_examples=None)
        
        print(f"Input prompt length: {len(prompt)} chars")
        
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        enc = {k: v.to(device) for k, v in enc.items()}
        
        print(f"Tokenized input length: {enc['input_ids'].shape[1]} tokens")
        
        with torch.no_grad():
            gen_ids = model.generate(
                **enc,
                max_new_tokens=300,
                num_beams=1,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                repetition_penalty=1.2,
            )
        
        print(f"Generated token IDs shape: {gen_ids.shape}")
        text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        print(f"Decoded text length: {len(text)} chars")
        obj = _parse_model_json(text, summary)
        outputs.append(obj)
    
    return outputs


def main():
    device = detect_device()
    print(f"Using device: {device}")

    tokenizer = T5TokenizerFast.from_pretrained(MODEL_DIR)
    base_model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model = PeftModel.from_pretrained(base_model, MODEL_DIR)
    model.to(device)
    model.eval()

    df = pd.read_csv(ENGINEERED_CSV)
    results = generate_for_businesses(df, tokenizer, model, device, limit=5)

    out_path = os.path.join(DATA_DIR, "data/outputs/inference_output.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote recommendations for {len(results)} businesses -> {out_path}")


if __name__ == "__main__":
    main()


