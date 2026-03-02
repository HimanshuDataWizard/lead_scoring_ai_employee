from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lead_scoring.inference import load_artifacts, predict_one
from api.main import LeadRequest
from pydantic import ValidationError


def main() -> None:
    model, metrics = load_artifacts()
    threshold = metrics.get("decision_threshold", 0.5)

    valid_payload = {
        "company_size": 120,
        "industry": "Pharma",
        "region": "Mumbai",
        "prev_purchases": 3,
        "response_time": 14,
        "last_contact": 21,
        "source": "Website",
        "deal_value": 15000.0,
    }

    print("[1] Valid payload")
    req = LeadRequest(**valid_payload)
    print("PASS", predict_one(model, req.model_dump(), threshold=threshold))

    print("[2] Unknown categories")
    unknown = dict(valid_payload)
    unknown.update({"industry": "FinTech", "region": "Pune", "source": "Partner"})
    print("PASS", predict_one(model, unknown, threshold=threshold))

    print("[3] Invalid payload")
    bad = dict(valid_payload)
    bad["company_size"] = -1
    try:
        LeadRequest(**bad)
        print("FAIL validation unexpectedly passed")
    except ValidationError:
        print("PASS validation failed as expected")


if __name__ == "__main__":
    main()
