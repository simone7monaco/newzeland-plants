from pathlib import Path
import pandas as pd
from pyinaturalist import get_taxa, pprint, Observation
import pyinaturalist as pi
import requests
from tqdm import tqdm

get_id = lambda spec: next(iter(get_taxa(q=spec)['results']), None)
nz_id = pi.get_places_autocomplete(q='new zealand')['results'][0]['id']

species = pd.read_excel('plant_info.xlsx').set_index('SpeciesName')
species['Features'] = species['Features'].fillna('')
species = species[species.Features != '']

download_dir = Path('photos')
download_large_dir = Path('photos_large')
download_dir.mkdir(exist_ok=True)
download_large_dir.mkdir(exist_ok=True)

def get_photos(sp):
	taxon_id = get_id(sp)
	if taxon_id is None:
		print(f"Couldn't find {sp}")
		return

	res_dir = download_dir / sp
	res_large_dir = download_large_dir / sp
	res_dir.mkdir(exist_ok=True, parents=True)
	res_large_dir.mkdir(exist_ok=True, parents=True)
	if len(list(res_dir.iterdir())) >= 500:
		return

	taxon_id = taxon_id['id']
	res = pi.get_observations(photos='true', quality_grade='research', order='desc', 
						   reviewed='true', taxon_id=taxon_id, page=1, per_page=1000, 
						   sort='preferred', place_id=nz_id)
	res = res['results']
	if len(res) == 0:
		print(f"Couldn't find {sp}")
		return


	for obs in res:
		obs = Observation.from_json(obs)
		if obs.photos is None:
			continue
		for pid, photo in enumerate(obs.photos):
			photo_large = photo.large_url
			photo_square = photo.url
			if photo_square is None:
				continue
			response = requests.get(photo_square)
			if response.status_code != 200:
				continue
			with open(res_dir / f"{obs.id}_{pid}.{photo.ext}", 'wb') as f:
				f.write(response.content)

			response = requests.get(photo_large)
			if response.status_code != 200:
				continue
			with open(res_large_dir / f"{obs.id}_{pid}.{photo.ext}", 'wb') as f:
				f.write(response.content)

	if len(list(res_dir.iterdir())) == 0:
		res_dir.rmdir()
		res_large_dir.rmdir()
		return
	return sp

# for sp in tqdm(species.index, total=len(species), desc='Downloading photos'):
# 	get_photos(sp)

# parallelize
from multiprocessing import Pool
from tqdm import tqdm

with Pool(16) as p:
	# p.map(get_photos, species.index)
	res = list(tqdm(p.imap(get_photos, species.index), total=len(species)))

