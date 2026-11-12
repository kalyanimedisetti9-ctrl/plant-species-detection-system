import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml_model.prediction import predict_plant

for name in ['brinjal.jpeg', 'brinjal_nZ64tNt.jpeg', 'brinjal_q740lO5.jpeg', 'brinjal_urlKc0l.jpeg']:
    p = Path('media/uploaded_images') / name
    if p.exists():
        res = predict_plant(str(p))
        print(f"{name}: Predicted -> {res['predicted_species']} ({res['confidence']}%) | Top3: {[(x['species'], x['confidence']) for x in res['top_predictions'][:3]]}")
