import regex as re
import numpy as np
import pandas as pd
from pathlib import Path

root = Path(__file__).parent.parent


species = pd.read_excel(root/"Felci"/"fern_descriptions.xlsx").set_index('Species')
species['Features'] = species.Etymology.fillna('') + ' ' + species['Vernacular name'].fillna('')

extra_features_df = pd.read_excel(root / "Words before and after traits_v2.xlsx", sheet_name="FernPrecedingWords")
extra_features_df = extra_features_df[
    :extra_features_df[extra_features_df.Rhizome.str.startswith('Sentences that talk a', na=False)].index[0]
    ].map(lambda s:s.lower().strip() if type(s) == str else s)

extra_features = extra_features_df.to_dict('list')
extra_features = {k.title().replace(' ', ''): [x for x in v if str(x) != 'nan'] for k, v in extra_features.items()}

unit = '[m|c|d|µ]?m'
number = r"(\d+\.?\d*)"
full_regex = rf"(({number}\s?-\s?)?{number})?({number}\s?-\s?)?{number}\s*{unit}\s(wide)?(long)?"
full_regex = full_regex + r'(?=[\s\.,;])'


def string_preprocessing(s):
	s = s.replace('\xa0', ' ').replace('×', 'x').replace('–', '-').replace('·', '.') # remove non-breaking space and replace multiplication sign with x
	s = re.sub(r'(?<=xcluding)\s+[\w-]+', ' ', s) # remove each word following "excluding" (Mericarps (excluding style) 2.5-3.0 mm should point to "Mericarps")
	s = re.sub(fr'-?\(-?{number}-?\)-?', '', s) # remove all parentesis surrounding a number and the number inside (e.g. (-1.5) --> **)
	s = re.sub(r'(?<=\d)\s+(?=[cmd]?m)', '', s) # remove all spaces before measures (mm, cm, dm, m, these strings only if padded by a space)
	# s = re.sub('m long;?', 'm-long', s) # remove space between measure and "long" (e.g. 2 mm long --> 2 mm-long)
	# s = re.sub('m wide;?', 'm-wide', s) # remove space between measure and "wide" (e.g. 2 mm wide --> 2 mm-wide)
	s = re.sub(r'\s*-\s*', '-', s) # remove spaces around hyphens
	s = re.sub(r'(?<=\d)\s*\.(?=\d)', '.', s) # remove spaces before dot if followed and preceded by a number
	s = re.sub(r'(?<=\s)\.(?=\d)', '0.', s) # add a 0 before a dot if it is preceded by a space and a "not number" and followed by a number (e.g. foo .5 --> foo 0.5)
	# s = re.sub(r'(?<=[\dm])\s*x\s*(?=\d+)', 'x', s) # remove spaces around x in formulas
	s = re.sub(rf'(;\s*)({full_regex})', r' \2', s) # point to any ';' preceding a measure (full_regex) and remove it, without removing the measure
	return s

tmp = []
anomalies = set() # species with anomalies

def extract_features(i, feats:list):
	features = {}
	for feat in feats:
		if len(feat) < 1:
			continue
		feat = feat.replace(',', ' ')
		measures = re.finditer(full_regex, feat)
		for measure in measures:
			found = None
			for key, values in extra_features.items():

				feat = feat[:-1] if feat[-1] in ['.', ';'] else feat # remove any of .; at the end of the sentence
				matched_word = list(re.finditer( r'\b('+ '|'.join([w for w in set(values)]) + r')\b', feat.lower()))
				
				if any(matched_word):	
					if key != 'Stature': # TODO: Caso up to 3m, petiole 2mm???
						matched_word = [w for w in matched_word if w.span()[0] < measure.span()[0]]
						if not any(matched_word):
							continue # if not stature and measure appears before the word, skip
					matched_word = sorted(matched_word, key=lambda word: word.span()[1] - measure.span()[0])[0]

					this_distance = abs(matched_word.span()[1] - measure.span()[0])
					# this_distance = abs(word_match_position - measure_position)
					if found:
						if (any([w in feat.lower() for w in ['achene', 'cypsela']]) and {key, found[0]} == {'FruitSize', 'SeedSize'}) or\
						   (any([w in feat.lower() for w in ['stigma-style']]) and {key, found[0]} == {'StigmaSize', 'StyleSize'}) or\
						   (any([w in feat.lower() for w in ['floret']]) and {key, found[0]} == {'RayFloretsSize', 'DiskFloretSize'}):
							pass
							# print(f'OK>> Multiple features found ({found}, {key}) in "{feat}"')
						else:
							if this_distance >= found[1]:
								continue
							else:
								# print(found, key, matched_word, measure)
								features[found[0]].remove(found[2])
							anomalies.add(i)
					found = (key, this_distance, measure.group().strip())
					
					if key in features:
						features[key].append(measure.group().strip())
					else:
						features[key] = [measure.group().strip()]
	return pd.Series(features)

def get_unlabelled_measures(processed_features):
	for i, (species_name, feature) in enumerate(processed_features.iterrows()):
		features_text = species.loc[species_name, 'Features'].replace('\xa0', ' ').replace('×', 'x').replace('–', '-')
		if features_text == '': continue
		
		measures_in_text = re.finditer(r'\d[\d\.\s\(\)x-]*[cmd]?m\s?(?![\d-\(\)])', features_text)
		for match in measures_in_text:
			which_feature = feature[feature.apply(lambda x: isinstance(x, str) and string_preprocessing(match.group()).strip() in x.split('; '))]
			if which_feature.empty:
				anomalies.add(species_name)