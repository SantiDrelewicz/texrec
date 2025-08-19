from .model import TexRecModel


class TexRec:
    """
    TexRec is a class that extends TexRecModel to provide additional functionality
    for text recognition tasks.
    """

    def __init__(self):
        # Additional initialization can be added here if needed
        self.__model = TexRecModel()
        self.__model.load_model("_texrec_model.pt")

    def __call__(self, text: str) -> str:
        """
        Calls the TexRecModel's __call__ method to perform plane text reconstruction.
        """
        return self.__model.predict()