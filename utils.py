import regex as re
import numpy as np
import pandas as pd
from pathlib import Path


species = pd.read_excel(Path(__file__).parent / "plant_info.xlsx").set_index('SpeciesName')
species['Features'] = species['Features'].fillna('')

extra_features_df = pd.read_excel("Words before and after traits_v1.xlsx", sheet_name="Words", skiprows=1)
extra_features_df = extra_features_df[:extra_features_df[extra_features_df.Stature.eq('Following words')].index[0]].applymap(lambda s:s.lower() if type(s) == str else s)

extra_features = extra_features_df.to_dict('list')
extra_features = {k.title().replace(' ', ''): [x for x in v if str(x) != 'nan'] for k, v in extra_features.items()}
all_words = extra_features_df.to_numpy().flatten()
# duplicated words
duplicate_words = []
for word in set(all_words):
	if (all_words == word).sum() > 1:
		duplicate_words.append(word)

words_to_remove = []
# remove all words starting with petal, flower, fruit, seed with a letter after different from s. Select the entire word up to the first space or punctuation (,;.) excluded
reg_exp = r'(petal|flower|fruit|seed)[a-rt-z]+[^\s;,.\)]'
for i, row in species.Features.items():
	# print(row)
	for word in re.finditer(reg_exp, row):
		words_to_remove.append(word.group())

words_to_remove = list(set(words_to_remove))


unit = '[m|c|d]?m'
# 150–400x100–300mm
number = r"(\d+\.?\d*)"
# full_regex = rf"(({number}-)?{number}{unit}?x)?{number}(-{number})?{unit}" ## Supposed to be correct
"""anomaies:
	0.05-0.35-1mx1.5-3-6mm
"""
full_regex = rf"({number}(-{number})?(-{number})?{unit}?x)?({number}(-{number})?(-{number})?{unit}?x)?{number}(-{number})?(-{number})?(-{number})?{unit}"


def string_preprocessing(s):
	s = s.replace('\xa0', ' ').replace('×', 'x').replace('–', '-') # remove non-breaking space and replace multiplication sign with x
	s = re.sub(fr'-?\(-?{number}-?\)-?', '', s) # remove all parentesis surrounding a number (e.g. (-1.5))
	s = s.replace('--', '-').replace('-.', '-').replace('..', '.')
	s = s.replace('(', '').replace(')', '')
	s = re.sub(r'\s(c|ca|o)\.', ' ', s) # remove all ' c.'
	s = re.sub(r'\s+(?=[cmd]?m)', '', s) # remove all spaces before measures (mm, cm, dm, m, these strings only if padded by a space)
	s = re.sub(r'\s*-\s*', '-', s) # remove spaces around hyphens
	s = re.sub(r'(?<=\d)\s*\.(?=\d)', '.', s) # remove spaces before dot if followed and preceded by a number
	s = re.sub(r'(?<=\s)\.(?=\d)', '0.', s) # add a 0 before a dot if it is preceded by a space and a "not number" and followed by a number (e.g. foo .5 --> foo 0.5)
	s = re.sub(r'(?<=[\dm])\s*x\s*(?=\d+)', 'x', s) # remove spaces around x in formulas
	# now all measures are supposed to have no spaces between number and unit and spaces around them
	s = re.sub(rf'(?<=\d)({unit})(?!\s*x)', r'\1 ', s) # fix situation in which a measure is not followed by a space, in the case, add that space
	s = re.sub(r'(?<![\d\sx\.-])(\d)', r' \1', s) # fix the situation in which a measure (the whole number and measure) is not preceded by a space. In the case, add a space before the measure
	s = re.sub(r'(?<=\d\.\d+)(\.\d?)', '', s) # fix the error in which there is a doubled dot in a number (e.g. 1.5.2), in the case, remove the second dot and the eventual numbers after it
	s = re.sub(r'(?<=[a-ln-z])-(?=\d)', ' ', s) # remove all '-' preceded by a letter (different from m) and followed by a number (e.g. to-250mm --> to 250mm)
	s = re.sub(r'(?<![a-z])(l|I)(?=[\s\.-]|\d)', '1', s) # replace all 'l' or "I" characters which should be '1' (e.g. l.5 --> 1.5). This should be followed by a space, a dot, a hyphen, or a number and not preceded by a letter
	s = re.sub('|'.join(words_to_remove), '', s, flags=re.IGNORECASE) # remove words to remove
	return s


tmp = []
anomalies = set() # species with anomalies

def extract_features(i, feats:list, wordmeasure_distance=100): # TODO: automatic wordmeasure_distance
	features = {}
	for feat in feats:
		if len(feat) < 1:
			continue
		# measure is a number and a unit of measurement (e.g. 1.5 mm)
		feat = feat.replace(',', ' ')
		measures = re.finditer(full_regex, feat)
		for measure in measures:
			found = None
			measure_position = np.where(np.array(feat.lower().split()) == measure.group())[0][0]
			for key, values in extra_features.items():
				# remove any of .; after one of the measures (cm, mm, dm, m), keeping the measure
				feat = re.sub(rf'{unit}[\.;]', lambda x: x.group()[:-1], feat)
				if any([w == wf for w in set(values) for wf in feat.lower().split()]):
					if 'calyx' in feat.lower() and key != 'CalyxSize' or 'petiole' in feat.lower() and key != 'PetioleSize' or 'anther' in feat.lower() and key != 'AntherSize' or 'pedicel' in feat.lower() and key != 'PedicelSize':
						continue
					word_match = [w.lower() for w in feat.split() if w.lower() in set(values)][0]
					word_match_position = np.where(np.array(feat.lower().split()) == word_match)[0][0]
					
					if found=='InflorescenceSize' and key == 'FlowerSize': # if inflorencence was already found, skip flower (e.g., "flower stem" associated with inflorescence only)
						continue
					if found=='FlowerSize' and key in ['StamenSize']: # avoid matching cases such as "Flowers large, white, about 8mm across, 4-petalled with 6 __stamens__"
						continue # TODO: can be avoided if considering only keywords before the measure?
					if found == 'FruitSize' and not any([w.lower() in feat.lower() for w in ['achene', 'cypsela']]):
						continue

					if abs(word_match_position - measure_position) > wordmeasure_distance:
						continue # TODO: remove abs to enforce keyword before measure (not in Stature, other exceptions?)
					if found:
						if (any([w.lower() in feat.lower() for w in ['achene', 'cypsela']]) and {key, found[0]} == {'FruitSize', 'SeedSize'}) or\
							{key, found[0]} == {'StigmaSize', 'StyleSize'}:
							pass
							# print(f'OK>> Multiple features found ({found}, {key}) in "{feat}"')
						else:
							this_distance = abs(word_match_position - measure_position)
							if this_distance > found[1]:
								continue
							else:
								features[found[0]].remove(found[2])
							tmp.append(f'({i}) Multiple features found ({found[0]}, {key}) in "{feat}"')
							anomalies.add(i)
					found = (key, abs(word_match_position - measure_position), measure.group())
					measure = measure.group()
					# print(key,[w.lower() for w in set(values)|set([key]) if w.lower() in feat.lower()], measure)
					
					if key in features:
						features[key].append(measure)
					else:
						features[key] = [measure]
			# primary features, if key is in petiole, anther, pedicel, calix
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