import os
from pathlib import Path
import requests
import pandas as pd
from tqdm.auto import tqdm
from io import BytesIO
plants = pd.read_excel("data/Australian layers/AusTraits.xlsx")
plant_names = plants.species.str.replace("_", " ")
from huggingface_hub import login

os.environ['HF_TOKEN'] = 'REDACTED_HF_TOKEN'


PROMPT = """
TASK
You will look only for FRUITS of the plant species in the image and output their COLOUR(S). 
Ignore flowers, leaves, stems, bark, background, containers, and artifacts.

SPECIES TEXT
Flower description (to ignore): {flower_description}
Fruit description (to match): {fruit_description}

COLOUR LABEL SET
black, blue, brown, green, red, white, yellow, purple, orange, grey

OUTPUT REQUIREMENTS
- Output only ONE line in this exact format (no extra words):
  <label>            e.g., "red"
  or
  <label>, <label>   e.g., "green, red"  (dominant first; max 2 colours)
- If no fruit is visible, or fruit is too small/hidden/ambiguous to be confident, output exactly: None

RULES
1) Detect candidate fruit regions; ignore anything matching only the flower description.
2) The region must be consistent with the provided fruit description (shape/arrangement/texture/position). If not, output None.
3) Determine the perceived surface colour(s) of the visible fruit(s).
   - If multiple fruits: choose the dominant overall colour; include a second colour only if it clearly covers >=25% of visible fruit area.
   - Ripening/bi-colour fruit: order by

"""

FLOWER_PROMPT = """Looking at this image, determine if there are flowers present returning 'True' or 'False'. Flowers for this species ({species}) are described as:
{flower_description}
"""
species_list = plant_names.tolist()

flower_descriptions = [
    "Golden cylindrical flower spikes made of dense, tiny spherical florets", # Acacia
]
fruit_descriptions = pd.read_excel("australian_plant_fruit_descriptions.xlsx", index_col=0).fruit_description.to_dict()
# iNaturalist API helpers
INAT_BASE = "https://api.inaturalist.org/v1"

def fetch_observations_for_species(species_name, per_page=50, max_pages=4, photos_only=True):
    """Fetch a list of observations for a given species_name from iNaturalist."""
    observations = []
    for page in range(1, max_pages + 1):
        params = {
            "taxon_name": species_name,
            "page": page,
            "per_page": per_page,
            "order_by": "created_at",
            "order": "desc",
            "photos": "true" if photos_only else "false",
        }
        r = requests.get(f"{INAT_BASE}/observations", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if not data.get("results"):
            break
        observations.extend(data["results"])
        if len(data["results"]) < per_page:
            break
    return observations

def observation_likely_has_fruit(obs):
    """Attempt to detect whether the observation has a fruit annotation or mention.
    This is a heuristic: checks controlled annotations and free-text fields for 'fruit' keywords.
    """
    return True
    # check annotations
    if obs.get("description"):
        if "fruit" in obs["description"].lower():
            return True
    if obs.get("description"):
        if "fruits" in str(obs["comments"]).lower():
            return True
    
    if any("fruit" in tag.lower() for tag in obs.get("tags", [])):
        print('++++ found')
        return True
    return False

def download_photo(photo):
    """Download photo bytes from observation_photos item returned by iNaturalist.
    photo is expected to contain a 'photo' dict with a 'url' key (e.g. .../medium.jpg).
    """
    url = photo.get("photo", {}).get("url")
    if not url:
        return None
    # prefer 'original' size if available by replacing size token
    for sz in ("medium", "small", "square", "original"):
        if f"/{sz}." in url:
            # try to request original
            url_orig = url.replace(f"/{sz}.", "/original.")
            try:
                r = requests.get(url_orig, timeout=30)
                if r.status_code == 200:
                    return r.content
            except Exception:
                pass
            break
    # fallback to the provided url
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.content



import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoModelForVision2Seq

class FruitColorVLM:
    """
    Analyze fruit color using a lightweight vision-language model.
    Using MobileVLM or similar lightweight VLM for local inference.
    """
    
    def __init__(self, model_name: str = "moondream/moondream3-preview", prompt: str = PROMPT, device: str = None):
        """
        Initialize the VLM for fruit color analysis.
        
        Args:
            model_name: VLM model name (Phi-3.5-vision is lightweight and efficient)
            device: Device to run inference on
        """
        # TODO: Improve the pipeline by segmenting the image
        # e.g. https://arxiv.org/pdf/2501.04001
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.prompt = prompt
        
    def _init_model(self):
        """Lazy initialization of the vision-language model."""
        if self.model is not None:
            return
        
        print(f"Loading {self.model_name} vision-language model...")

        self.model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            trust_remote_code=True,
            device_map="cuda", # "cuda" on Nvidia GPUs
        ).to(torch.bfloat16)
        self.settings = {"temperature": 0.5, "max_tokens": 768, "top_p": 0.3}
        
        
        
        print(f"VLM loaded on {self.device}")

    def image_has_fruit(self, imag: Image, genus: str) -> bool:
        """
        Determine if the image likely contains fruit using VLM.
        
        Args:
            image_path: Path to the image file
        Returns:
            True if fruit is likely present, False otherwise
        """
        self._init_model()
        
        try:
            prom

            
    
    def get_fruit_color(self, img, genus: str):
        """
        Extract fruit color from image using VLM.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict with 'color' and 'raw_response'
        """
        self._init_model()
        
        try:
            image = Image.open(BytesIO(img)).convert('RGB')

            prompt = self.prompt.format(
                fruit_description=fruit_descriptions[genus],
                flower_description=", ".join(flower_descriptions)
            )
            
            answer = self.model.query(
                image,
                prompt,
                settings=self.settings
            )    
            if isinstance(answer, dict):
                answer = answer['answer']
            # Clean up the response (remove prompt if echoed)
            if prompt in answer:
                answer = answer.replace(prompt, "").strip()
            
            # Extract color from response
            color = self._parse_color(answer)
            
            return {
                'color': color,
                'raw_response': answer
            }
            
        except Exception as e:
            print(f"Error: {e}")
            return {
                'color': 'unknown',
                'raw_response': f'Error: {str(e)}'
            }
    
    def _parse_color(self, response: str) -> str:
        """
        Parse color from VLM response.
        
        Args:
            response: Raw text response from VLM
            
        Returns:
            Extracted color string
        """
        if response is None or response.strip() == "" or response.lower() == "none":
            return 'None'
            
        response_lower = response.lower()
        
        # Common fruit colors
        colors = [
            'red', 'green', 'yellow', 'orange', 'purple', 'blue',
            'brown', 'black', 'white', 'pink', 'burgundy', 'crimson',
            'golden', 'amber', 'violet', 'indigo', 'maroon', 'tan'
        ]
        
        # Check for color words
        found_colors = [c for c in colors if c in response_lower]
        
        if found_colors:
            return found_colors[0]  # Return first found color
        
        # Return full response if no specific color found
        return 'None'


img_path = Path("australian_imgs")
img_path.mkdir(exist_ok=True, parents=True)

rows = []
seen_photos = set()
vlm = FruitColorVLM()
per_species_limit = 500  # max observations per species to fetch

for i, species in tqdm(enumerate(species_list), total=len(species_list)):
    obs = fetch_observations_for_species(species, per_page=50, max_pages=(per_species_limit // 50) or 1)

    for o in tqdm(obs, desc=f"Observations for {species}", leave=False):
        if not observation_likely_has_fruit(o):
            # still consider passing through VLM if you want stricter filtering,
            # but this keeps API calls down by discarding obvious non-fruit obs.
            continue
        for op in o.get("observation_photos", []):
            photo_dict = op.get("photo") or op
            photo_id = photo_dict.get("id") or op.get("id")
            if not photo_id or photo_id in seen_photos:
                continue
            seen_photos.add(photo_id)
            try:
                img = download_photo(op)
            except Exception as e:
                # skip problematic downloads
                continue
            vlm_answer = vlm.get_fruit_color(img, genus=species.split(" ")[0])
            
            rows.append({
                "image_id": photo_id,
                "species": species,
                "tags": o.get("tags", []),
            } | vlm_answer)

            if not vlm_answer.get("color") in (None, "None", "unknown"):
                # store image locally in a subfolder for species with detected fruit color
                species_folder = img_path / species.replace(" ", "_")
                species_folder.mkdir(exist_ok=True, parents=True)
                img_file = species_folder / f"{photo_id}.jpg"
                if not img_file.exists():
                    with open(img_file, "wb") as f:
                        f.write(img)
                        
    if (i + 1) % 1 == 0:
        # save intermediate results every 10 species
        df = pd.DataFrame(rows)
        df.to_csv("australian_plant_fruit_colors_partial.csv", index=False)

df = pd.DataFrame(rows)
df.to_csv("australian_plant_fruit_colors.csv", index=False)