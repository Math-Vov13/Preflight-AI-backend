import datetime, os

def generate_s3_object_name(file_name: str, content_type: str) -> str:
    """Génère un nom d'objet S3 unique en combinant le nom de fichier, la date et l'heure actuelles."""
    base_name, ext = os.path.splitext(os.path.basename(file_name))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    object_name = f"{str.split(content_type, '/')[0]}/{base_name}-{timestamp}{ext}"
    return object_name.replace(" ", "_").lower()