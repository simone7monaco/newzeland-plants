import streamlit as st
from pathlib import Path
import seaborn as sns
import json

from annotated_text import annotated_text
from utils import *
import hmac

processed_path = Path(__file__).parent / "processed_features.csv"
species = pd.read_excel(Path(__file__).parent / "plant_info.xlsx").set_index('SpeciesName')
species['Features'] = species['Features'].fillna('')

with open(Path(__file__).parent / "acceptances.json") as f:
    acceptances = json.load(f)

def get_color(score, fixed_color=True):
    from colorsys import rgb_to_hsv, hsv_to_rgb
    if fixed_color: return colors[score.argmax()]
    h = colors[score.argmax()].lstrip("#")

    c = rgb_to_hsv(*tuple(int(h[i:i+2], 16) for i in (0, 2, 4)))
    c = [c[0], c[1], 255-((255 - c[2]) * (score.max()))]
    c = [int(ch) for ch in hsv_to_rgb(*c)]

    return '#%02x%02x%02x' % tuple(c)

def refresh_processed_features():
    processed_features = species.Features.apply(string_preprocessing).str.split(r'(?<!\sc)[.;]\s|;').reset_index().apply(lambda x: extract_features(x.SpeciesName, x.Features), axis=1)
    processed_features = processed_features.applymap(lambda x: '; '.join(x) if not isinstance(x, float) else x)
    processed_features.index = species.index

    processed_features.to_csv(processed_path)
    get_unlabelled_measures(processed_features)

    with open(Path(__file__).parent / "acceptances.json", "w") as f:
        json.dump({sp: '' for sp in anomalies}, f) # TODO: is this available from utils?
    return processed_features

def update_acceptance(toggles):
    for k, v in toggles.items():
        if v:
            acceptances[k] = 'correct'
        else:
            acceptances[k] = ''
            # if acceptances.get(k, '') == 'correct':
                # acceptances.pop(k)
    with open(Path(__file__).parent / "acceptances.json", "w") as f:
        json.dump(acceptances, f)


def main():
    st.title("Extracted features control")
    st.write("This page is used to control the extracted features from the plant_info.xlsx file")

    if st.button("Refresh", type="primary"):
        refresh_processed_features()
    
    # data = st.file_uploader("Carica il file", type=["doc", "docx"])
    # now the file to take is plant_info.xlsx

    if processed_path.exists():
        processed_features = pd.read_csv(processed_path, index_col=0)
    else:
        processed_features = refresh_processed_features()

    with open(Path(__file__).parent / "acceptances.json") as f:
        acceptances = json.load(f)
        

    get_color = lambda i: sns.color_palette("husl", len(processed_features.columns)).as_hex()[i]

    features_colors = {feat: get_color(i) for i, feat in enumerate(processed_features.columns)} | {'null': '#808080'}

    page_length = 25
    processed_features_anom = processed_features[processed_features.index.isin(acceptances.keys())]
    page_number = st.selectbox("Page number", range(0, len(processed_features_anom), page_length), index=0)

    processed_view = processed_features_anom.iloc[page_number:page_number+page_length]
    st.text(f"Showing species from {page_number} to {page_number+page_length}")
    toggles = {}
    bar = st.progress(0)
    for i, (species_name, feature) in enumerate(processed_view.iterrows()):
        bar.progress(i/len(processed_view))
        features_text = species.loc[species_name, 'Features'].replace('\xa0', ' ').replace('×', 'x').replace('–', '-')
        if features_text == '': continue
        # print in the center of the page a bold title with the shown species (ids from ** to **)
        

        # TODO: create 3 tabs, one with all the remaining texts, one with the accepted, one with the errors
        if acceptances.get(species_name, '') == 'correct':
            st.header(f":green[{species_name}]")
            continue

        st.header(f"{species_name}")
        toggles[species_name] = st.toggle('Mark as correct', key=f"{species_name}_toggle", value=True if acceptances.get(species_name, '') == 'correct' else False) 

        
        

        measures_in_text = re.finditer(r'\d[\d\.\s\(\)x-]*[cmd]?m\s?(?![\d-\(\)])', features_text)
        
        detected_features = []
        for match in measures_in_text:
            which_feature = feature[feature.apply(lambda x: isinstance(x, str) and string_preprocessing(match.group()).strip() in x.split('; '))]
            if which_feature.empty:
                detected_features.append(('null', match.group(), match.start(), match.end()))
            else:
                for feat in which_feature.index:
                    detected_features.append((feat, match.group(), match.start(), match.end()))
        detected_features = sorted(detected_features, key=lambda x: x[2])

        coloured_text = []
        last_found = 0
        for feat, raw, start, end in detected_features:
            coloured_text.append(features_text[last_found:start])
            coloured_text.append((features_text[start:end], feat, features_colors[feat]))
            last_found = end

        if last_found != len(features_text):
            coloured_text.append(features_text[last_found:])
        
        annotated_text(*coloured_text)
        # write in bold
        st.markdown(f"**Extracted features**")
        annotated_text(*[(meas, feat, features_colors[feat]) for meas, feat in zip(feature, feature.index) if isinstance(meas, str)])
    
    with st.sidebar:
        col1, col2 = st.columns(2)
        with col1:
            submit = st.button("Submit", type="primary")
        with col2:
            st.download_button(
                label="Download JSON",
                data=json.dumps(acceptances),
                file_name='acceptances.json',
                mime='application/json',
            )
        if submit:
            update_acceptance(toggles)
            st.success("Submitted!")
        
        st.header("Marked as wrong")
        for k, v in toggles.items():
            if v:
                st.write(k)
        
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the passward is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False

if __name__ == "__main__":
    if not check_password():
        st.stop()  # Do not continue if check_password is not True.

    main()
