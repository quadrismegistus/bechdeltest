from ..imports import *
from .texts import *

def genderfy(names):
    apikey='10f018f37deabadcf64c7e26bc869657'
    from genderize import Genderize
    genderize = Genderize(
        api_key=apikey,
        timeout=5.0
    )
    return genderize.get(names)


def get_text_names(text_id):
    df_cast = get_text_cast(text_id)
    cast_names = (list(df_cast.actor_name) + list(df_cast.char_name)) if len(df_cast) else []
    
    df_dial = get_text_dialogue(text_id)
    dial_names = list(df_dial.speaker) if len(df_dial) else []

    return Counter(
        str(name).title() for name in cast_names+dial_names
    )

def get_all_full_names():
    ofn=os.path.join(PATH_DATA, 'all_full_names.json')
    if not os.path.exists(ofn):
        names = Counter()
        for text_id in tqdm(df.index):
            names += get_text_names(text_id)
        with open(ofn,'w') as of: json.dump(dict(names.most_common()), of, indent=4)
    else:
        with open(ofn) as of: return json.load(of)


def check_real_names_stanza(names):
    import stanza
    nlp = stanza.Pipeline('en', verbose=False, processors='tokenize,mwt,ner')
    return {
        name:('PERSON' in {ent.type for ent in nlp(name).entities})
        for name in tqdm(names)
    }

    

def check_real_names_spacy(names):
    # python -m spacy download en_core_web_sm
    import spacy
    from tqdm import tqdm
    # Load English tokenizer, tagger, parser and NER
    nlp = spacy.load("en_core_web_sm")
    return {
        name:('PERSON' in {ent.label_ for ent in nlp(name).ents})
        for name in tqdm(names)
    }

    

def get_all_full_names_real():
    ofn = os.path.join(PATH_DATA,'all_full_names_real.json')
    if not os.path.exists(ofn):
        all_full_names = get_all_full_names()
        name_reald = check_real_names_spacy(all_full_names)
        with open(ofn, 'w') as of: json.dump(name_reald, of)
        return name_reald
    else:
        with open(ofn) as f: return json.load(f)


def get_all_first_names_real(force=False):
    ofn = os.path.join(PATH_DATA,'all_first_names_real.json')
    if force or not os.path.exists(ofn):
        all_full_names_real = get_all_full_names_real()
        all_first_names = Counter([
            strip_punct(name.strip()).split()[0]
            for name,isreal in all_full_names_real.items()
            if isreal
        ])
        all_first_names_real = check_real_names_spacy(all_first_names)
        with open(ofn, 'w') as of: json.dump(all_first_names_real, of)
        return all_first_names_real
    else:
        with open(ofn) as f: return json.load(f)



def get_all_first_names_real_genderized(force=False, min_prob=0):
    ofn = os.path.join(PATH_DATA,'all_first_names_real_genderized.json')
    if force or not os.path.exists(ofn):
        all_first_names_real = get_all_first_names_real()
        objs = list(all_first_names_real.keys())
        res = genderfy(objs)
        with open(ofn, 'w') as of: json.dump(res, of)
    
    with open(ofn) as f: ld = json.load(f)
    odf = pd.DataFrame(ld).fillna('').set_index('name')
    odf = odf.sort_values('probability',ascending=False)
    odf = odf[odf.probability != 0]
    odf = odf[odf.probability >= min_prob]

    odf['gender_dom'] = [
        'def_male' if gender=="male" and prob>=.9 else "not_def_male"
        for gender,prob in zip(odf.gender, odf.probability)
    ]
    odf['gender_def'] = [
        'def_'+gender if prob>=.9 else "not_def_"+gender
        for gender,prob in zip(odf.gender, odf.probability)
    ]

    return odf



def get_all_text_gender_ratios(force=False):
    ofn=os.path.join(PATH_DATA,'data.gender_ratios.csv')
    if force or not os.path.exists(ofn):
        text_ids = get_all_text_ids()
        res = pmap(compute_gender_ratios, text_ids, num_proc=8)
        resdf = pd.DataFrame(res).set_index('id')
        resdf.to_csv(ofn)
        return resdf
    else:
        return pd.read_csv(ofn).set_index('id')


def compute_gender_ratios(text_id):
    data = {
        'cast':get_text_cast(text_id),
        'dialogue':get_text_dialogue(text_id)
    }
    gdf = get_all_first_names_real_genderized()
    odx={'id':text_id}
    def dodiv(x,y): return x/y if y else np.nan
    for dftype,df in data.items():
        try:
            counts_dom = dict(df.gender_dom.value_counts())
            counts_def = dict(df.gender_def.value_counts())
            odx[f'num_{dftype}_def_male'] = counts_def.get('def_male',0)
            odx[f'num_{dftype}_not_def_male'] = counts_dom.get('not_def_male',0)
            odx[f'num_{dftype}_def_female'] = counts_def.get('def_female',0)
            odx[f'ratio_{dftype}__def_male/not_def_male'] = dodiv(counts_dom.get('def_male',0), counts_dom.get('not_def_male',0))
            odx[f'ratio_{dftype}__def_male/def_female'] = dodiv(counts_def.get('def_male',0), counts_def.get('def_female',0))
        except Exception as e:
            # print(f'!! {e}')
            pass
    return odx