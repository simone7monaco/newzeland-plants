import base64
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from pathlib import Path
import requests
import pandas as pd
from tqdm.auto import tqdm
from io import BytesIO
from typing import Dict, Optional, Tuple, Union
from enum import Enum
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

plants = pd.read_excel("data/Australian layers/AusTraits.xlsx")
plant_names = plants.species.str.replace("_", " ")

class ImageContent(Enum):
    """Enum for detected image content types"""
    FLOWER = "flower"
    FRUIT = "fruit"
    NEITHER = "neither"
    UNCERTAIN = "uncertain"

# Agent prompts for different stages
FLOWER_DETECTION_PROMPT = """
TASK: Detect if there are FLOWERS visible in this image.

SPECIES: {species}
FLOWER DESCRIPTION: {flower_description}

OUTPUT REQUIREMENTS:
- Output ONLY one word: "Yes" or "No"
- "Yes" if you can clearly see flowers matching the description
- "No" if no flowers are visible or uncertain

RULES:
1) Only consider structures matching the flower description
2) Ignore fruits, buds, leaves, stems, and background
3) Be conservative - if uncertain, output "No"
"""

FRUIT_DETECTION_PROMPT = """
TASK: Detect if there are FRUITS visible in this image.

SPECIES: {species}
FRUIT DESCRIPTION: {fruit_description}

OUTPUT REQUIREMENTS:
- Output ONLY one word: "Yes" or "No"
- "Yes" if you can clearly see fruits matching the description
- "No" if no fruits are visible or uncertain

RULES:
1) Only consider structures matching the fruit description
2) Ignore flowers, leaves, stems, bark, and background
3) Fruits must be clearly visible (not tiny/hidden/ambiguous)
4) Be conservative - if uncertain, output "No"
"""

FRUIT_COLOR_EXTRACTION_PROMPT = """
TASK: Extract the COLOUR(S) of FRUITS visible in this image.

SPECIES: {species}
FRUIT DESCRIPTION: {fruit_description}

COLOUR LABEL SET:
black, blue, brown, green, red, white, yellow, purple, orange, grey

OUTPUT REQUIREMENTS:
- Output ONLY colour label(s) in this exact format:
  <label>            e.g., "red"
  or
  <label>, <label>   e.g., "green, red"  (dominant first; max 2 colours)

RULES:
1) Focus only on the visible fruit surfaces
2) Report dominant colour first
3) Include second colour only if it covers >=25% of visible fruit area
4) For ripening/bi-colour fruits: order by dominance
5) Ignore background, containers, and non-fruit elements
"""

IMAGE_QUALITY_PROMPT = """
TASK: Assess if this image is suitable for botanical analysis.

OUTPUT REQUIREMENTS:
- Output ONLY one word: "Good" or "Poor"
- "Good" if image is clear, well-lit, and shows plant structures in detail
- "Poor" if image is blurry, dark, distant, or shows minimal plant detail

RULES:
1) Check focus/sharpness
2) Check lighting and visibility
3) Check if plant structures are identifiable
4) Be strict - when in doubt, output "Poor"
"""

species_list = plant_names.tolist()
plant_descriptions = pd.read_csv("australian_plant_descriptions.csv", index_col=0)

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

def download_photo(photo):
    """Download photo and return as base64 string for efficient reuse.
    
    Returns:
        str: Base64-encoded image data, or None if download fails
    """
    url = photo.get("photo", {}).get("url")
    if not url:
        return None
    
    for sz in ("medium", "small", "square", "original"):
        if f"/{sz}." in url:
            # try to request original
            url_orig = url.replace(f"/{sz}.", "/original.")
            try:
                r = requests.get(url_orig, timeout=30)
                r.raise_for_status()
                if r.status_code == 200:
                    return base64.b64encode(r.content).decode('utf-8')
            except Exception:
                pass
            break
    return base64.b64encode(r.content).decode('utf-8')
    

import torch
from PIL import Image
from transformers import AutoModelForCausalLM

class AgenticFruitResearchPipeline:
    """
    Multi-agent pipeline for fruit research:
    1. Quality assessment agent
    2. Flower detection agent
    3. Fruit detection agent
    4. Color extraction agent
    """
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                #  model_name: str = "moondream/moondream3-preview", 
                 device: str = None, 
                 max_image_size: Tuple[int, int] = (800, 800)):
        """Initialize the agentic pipeline.
        
        Args:
            model_name: VLM model name
            device: Device to run inference on
            max_image_size: Maximum image dimensions (width, height) for resizing
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.model = None
        self.settings = {"temperature": 0.3, "max_tokens": 512, "top_p": 0.3}
        self.max_image_size = max_image_size
        self._init_model()
        
    def _init_model(self):
        """Lazy initialization of the vision-language model."""
        if self.model is not None:
            return
        
        print(f"Loading {self.model_name} vision-language model...")
        if 'Qwen2.5' in self.model_name:
            # except Exception as e:
            # print("Falling back to auto device map due to error:", e)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                dtype=torch.bfloat16,
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(self.model_name)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                "vikhyatk/moondream2",
                trust_remote_code=True,
                device_map="cuda",
            ).to(torch.bfloat16)
            self.processor = None
            self.model.eval()
        print(f"VLM loaded on {self.device}")
    
    def _query_vlm(self, image_base64: str, prompt: str) -> str:
        """Query the VLM with a base64-encoded image and prompt.
        
        Args:
            image_base64: Base64-encoded image string
            prompt: Text prompt for the VLM
            
        Returns:
            str: VLM response
        """
        if 'Qwen2.5' in self.model_name:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"data:image;base64,{image_base64}"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            # Preparation for inference
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Inference: Generation of the output (reduced peak memory)
            with torch.no_grad():
                if torch.cuda.is_available():
                    try:
                        # mixed precision context reduces activation memory on CUDA
                        with torch.amp.autocast(device_type="cuda"):
                            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
                    except Exception as e:
                        print('🚨🚨', e)
                        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
                else:
                    generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            answer = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        else:
            # For moondream, convert base64 to PIL Image
            img_bytes = base64.b64decode(image_base64)
            image = Image.open(BytesIO(img_bytes)).convert('RGB')
            answer = self.model.query(image, prompt, settings=self.settings)
            if isinstance(answer, dict):
                answer = answer.get('answer', '')
        return answer.strip()
    
    def agent_quality_check(self, image_base64: str, species: str, 
                            description: str = "") -> Tuple[bool, str]:
        """
        Agent 1: Quality assessment
        Args:
            image_base64: Base64-encoded image
        Returns: (is_good_quality, raw_response)
        """
        # check image size only
        img_bytes = base64.b64decode(image_base64)
        image = Image.open(BytesIO(img_bytes)).convert('RGB')
        is_good_quality = image.size[0] >= 256 and image.size[1] >= 256
        return is_good_quality, ""
        prompt = IMAGE_QUALITY_PROMPT
        response = self._query_vlm(image_base64, prompt)
        is_good = response.lower().startswith('good')
        return is_good, response
    
    def agent_detect_flower(self, image_base64: str, species: str, 
                            flower_description: str) -> Tuple[bool, str]:
        """
        Agent 2: Flower detection
        Args:
            image_base64: Base64-encoded image
        Returns: (has_flower, raw_response)
        """
        prompt = FLOWER_DETECTION_PROMPT.format(
            species=species,
            flower_description=flower_description
        )
        response = self._query_vlm(image_base64, prompt)
        doesnt_have_flower = response.lower().startswith('no')
        return doesnt_have_flower, response
    
    def agent_detect_fruit(self, image_base64: str, species: str,
                           fruit_description: str) -> Tuple[bool, str]:
        """
        Agent 3: Fruit detection
        Args:
            image_base64: Base64-encoded image
        Returns: (has_fruit, raw_response)
        """
        prompt = FRUIT_DETECTION_PROMPT.format(
            species=species,
            fruit_description=fruit_description
        )
        response = self._query_vlm(image_base64, prompt)
        has_fruit = response.lower().startswith('yes')
        
        return has_fruit, response
    
    def agent_extract_color(self, image_base64: str, species: str,
                            fruit_description: str) -> Tuple[str, str]:
        """
        Agent 4: Color extraction
        Args:
            image_base64: Base64-encoded image
        Returns: (color, raw_response)
        """
        prompt = FRUIT_COLOR_EXTRACTION_PROMPT.format(
            species=species,
            fruit_description=fruit_description
        )
        response = self._query_vlm(image_base64, prompt)
        color = self._parse_color(response)
        return color, response
    
    def _parse_color(self, response: str) -> str:
        """Parse color from VLM response."""
        if not response or response.lower() == "none":
            return 'None'
        
        response_lower = response.lower()
        colors = [
            'red', 'green', 'yellow', 'orange', 'purple', 'blue',
            'brown', 'black', 'white', 'pink', 'grey', 'gray'
        ]
        
        found_colors = [c for c in colors if c in response_lower]
        if found_colors:
            # Return up to 2 colors, comma-separated if multiple
            return ', '.join(found_colors[:2])
        
        return 'None'
    
    def process_image(self, image_base64: str, species: str, genus: str) -> Dict:
        """
        Run the full agentic pipeline on a base64-encoded image.
        
        Pipeline stages:
        0. Quality assessment (size only)
        ## 1. Leaf shape analysis (con immagini associate alla forma della foglia?)
        1. Flower detection
        2. Fruit detection (if no flower)
            a. Color extraction (if fruit detected)
        
        Args:
            image_base64: Base64-encoded image string
            species: Species name
            genus: Genus name
            
        Returns: Dict with results from each agent
        """
        agents = {
            'quality_check': self.agent_quality_check,
            'flower_detection': self.agent_detect_flower,
            'fruit_detection': self.agent_detect_fruit,
            'color': self.agent_extract_color,
        }
        try:
            # Get plant descriptions
            fruit_description, flower_description = plant_descriptions.xs(genus)
            
            # result = {
            #     'species': species,
            #     'genus': genus,
            #     'quality_check': None,
            #     'flower_detection': None,
            #     'fruit_detection': None,
            #     'color_extraction': None,
            #     'pipeline_stage': None,
            #     'raw_responses': {}
            # }
            result = {'species': species, 'genus': genus, 'raw_responses': {}}

            for stage, agent_name in enumerate(agents.keys()):
                response, raw_response = agents[agent_name](image_base64, species, 
                                                           flower_description if 'flower' in agent_name else fruit_description)
                result[agent_name] = response
                result['raw_responses'][agent_name] = raw_response
                if isinstance(response, bool):
                    #0: is_good_quality, 1: dont_have_flower, 2: have_fruit
                    if not response:
                        result['pipeline_stage'] = f'rejected_at_stage_{stage}'
                        return result
                    
            result['pipeline_stage'] = 'completed'
            return result
            
        except Exception as e:
            raise RuntimeError(f"Pipeline failed at stage {stage} for species {species}: {e}") from e

# Main processing loop
img_path = Path("australian_imgs")
img_path.mkdir(exist_ok=True, parents=True)

for child in img_path.iterdir():
    if child.is_file():
        child.unlink()
    else:
        import shutil
        shutil.rmtree(child)

rows = []
seen_photos = set()
pipeline = AgenticFruitResearchPipeline()
per_species_limit = 500

for i, species in tqdm(enumerate(species_list), total=len(species_list)):
    obs = fetch_observations_for_species(species, per_page=50, max_pages=(per_species_limit // 50) or 1)
    genus = species.split(" ")[0]

    for o in tqdm(obs, desc=f"Processing {species}", leave=False):
        saved_imgs = 0
        for op in o.get("observation_photos", []):
            photo_dict = op.get("photo") or op
            photo_id = photo_dict.get("id") or op.get("id")
            
            if not photo_id or photo_id in seen_photos:
                continue
            seen_photos.add(photo_id)
            
            # Download image as base64 (single download, reused for all queries)
            try:
                img_base64 = download_photo(op)
                if not img_base64:
                    continue
            except Exception as e:
                continue
            
            # Run agentic pipeline on base64 image
            result = pipeline.process_image(img_base64, species, genus)
            
            # Add metadata
            result['image_id'] = photo_id
            result['tags'] = o.get("tags", [])
            rows.append(result)
            
            # Only save image if fruit detected
            should_keep = result.get('pipeline_stage') == 'completed'
            
            if should_keep:
                # Decode base64, resize, and save only when needed
                species_folder = img_path / species.replace(" ", "_")
                species_folder.mkdir(exist_ok=True, parents=True)
                img_file = species_folder / f"{photo_id}.jpg"
                
                try:
                    img_bytes = base64.b64decode(img_base64)
                    pil_img = Image.open(BytesIO(img_bytes)).convert('RGB')
                    pil_img.thumbnail((600, 600), Image.Resampling.LANCZOS)
                    pil_img.save(img_file, "JPEG", quality=85, optimize=True)
                    result['image_path'] = str(img_file)
                    saved_imgs += 1
                    if saved_imgs >= 7:
                        break
                except Exception as e:
                    result['image_path'] = None
    
    # Save intermediate results
    if (i + 1) % 1 == 0:
        df = pd.DataFrame(rows)
        df.to_csv("australian_plant_fruit_colors_partial.csv", index=False)

# Final save
df = pd.DataFrame(rows)
df.to_csv("australian_plant_fruit_colors.csv", index=False)

# Print pipeline statistics
print("\n=== Pipeline Statistics ===")
print(f"Total images processed: {len(df)}")
print(f"\nBy pipeline stage:")
print(df['pipeline_stage'].value_counts())
print(f"\nFruit colors detected:")
print(df[df['pipeline_stage'] == 'completed']['color'].value_counts())