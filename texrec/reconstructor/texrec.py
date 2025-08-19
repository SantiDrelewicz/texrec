from .model import TexRecModel
from pathlib import Path
import requests
import torch


class TexRec:
    """
    TexRec is a class that extends TexRecModel to provide additional functionality
    for text recognition tasks.
    """

    def __init__(self, filename="texrec_model.pt", version="v0.1.0"):
        # URL del release en GitHub
        url = f"https://github.com/SantiDrelewicz/pitext/releases/download/{version}/{filename}"
        
        # Cache local (~/.cache/pitext/texrec_model.pt)
        cache_path = Path.home() / ".cache" / "texrec" / filename
        cache_path.parent.mkdir(parents=True, exist_ok=True)

         # Descargar si no está en caché
        if not cache_path.exists():
            print(f"Descargando modelo desde {url}...")
            r = requests.get(url)
            r.raise_for_status()
            with open(cache_path, "wb") as f:
                f.write(r.content)
            print("Descarga completada ✅")

        # Cargar el modelo
        state = torch.load(cache_path, map_location="cpu", )
        model = TexRecModel()  # tu clase definida en pitext
        model.load_state_dict(state)
        model.eval()
        self.__model = model


    def __call__(self, text: str) -> str:
        """
        Calls the TexRecModel's __call__ method to perform plane text reconstruction.
        """
        return self.__model.predict()