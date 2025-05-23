import torch
import joblib

# Зареждане на TorchScript модела (.pt)
traced_model = torch.jit.load("skills_assessment.pt")

# Сериализация към .joblib
joblib.dump(traced_model, "skills_assessment.joblib")

print("[✓] Моделът е успешно запазен като skills_assessment.joblib")
