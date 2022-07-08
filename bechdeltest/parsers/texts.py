from ..imports import *
from .names import *


def get_corpus_metadata(force=False):
    fn=PATH_CORPUS_METADATA
    if not os.path.exists(fn): return pd.DataFrame()
    df = pd.read_csv(fn, dtype='str').set_index('id')
    return df

def get_corpus_metadata_annotated(force=False):
    df = get_corpus_metadata().reset_index()
    
    resdf = get_bechdeltestdotcom_scores()
    resdf.columns = ['bechdeltest.com_'+col for col in resdf]
    df = df.merge(resdf, left_on='imdb_id', right_on='bechdeltest.com_imdbid')

    from .names import get_all_text_gender_ratios
    df_ratios = get_all_text_gender_ratios()
    df = df.merge(df_ratios, on='id')
    return df.set_index('id')

def get_bechdeltestdotcom_scores(force=False):
    df = get_corpus_metadata()
    # get bechdel scores
    ofn=os.path.join(PATH_DATA,'bechdeltestdotcom_scores.csv')
    if force or not os.path.exists(ofn):
        objs = list(df.imdb_id)
        res = pmap(do_get_bechdel_score, objs)
        res2 = [d for d in res if 'rating' in d]
        resdf = pd.DataFrame(res2)
        resdf.to_csv(ofn)
    else:
        resdf = pd.read_csv(ofn, dtype='str')
    return resdf

def get_all_text_ids():
    return list(get_corpus_metadata().index)

def get_text_cast(text_id):
    fn = os.path.join(PATH_CORPUS_TEXTS, text_id, text_id + '.cast.csv')
    if not os.path.exists(fn): return pd.DataFrame()
    odf = pd.read_csv(fn, dtype='str').set_index('order')
    odf['actor_fname'] = odf['actor_name'].apply(lambda x: strip_punct(str(x).strip()).split()[0].title() if strip_punct(str(x).strip()) else '')
    odf['char_fname'] = odf['char_name'].apply(lambda x: strip_punct(str(x).strip()).split()[0].title() if strip_punct(str(x).strip()) else '')

    from .names import get_all_first_names_real_genderized, get_all_first_names_real, get_all_full_names_real
    gdf1 = get_all_first_names_real_genderized()
    gdf2 = get_all_first_names_real_genderized()
    valid_fnames = get_all_first_names_real()
    valid_fnames = get_all_full_names_real()
    odf['actor_name_real'] = odf['actor_name'].apply(valid_fnames.get)
    odf['char_name_real'] = odf['char_name'].apply(valid_fnames.get)
    odf['actor_fname_real'] = odf['actor_fname'].apply(valid_fnames.get)
    odf['char_fname_real'] = odf['char_fname'].apply(valid_fnames.get)

    gdf1.columns = ['actor_'+col for col in gdf1]
    gdf2.columns = ['char_'+col for col in gdf2]
    odf = odf.merge(gdf1, left_on = 'actor_fname', right_on = 'name', how='left').fillna('')
    odf = odf.merge(gdf2, left_on = 'char_fname', right_on = 'name', how='left').fillna('')
    odf['gender']=[x if x else y for x,y in zip(odf.actor_gender, odf.char_gender)]
    odf['gender_dom']=[x if x else y for x,y in zip(odf.actor_gender_dom, odf.char_gender_dom)]
    odf['gender_def']=[x if x else y for x,y in zip(odf.actor_gender_def, odf.char_gender_def)]
    return odf

def get_text_dialogue(text_id):
    fn = os.path.join(PATH_CORPUS_TEXTS, text_id, text_id + '.dialogue.csv')
    if not os.path.exists(fn): return pd.DataFrame()
    odf = pd.read_csv(fn, dtype='str').set_index('line_num')

    odf['speaker_name'] = odf['speaker'].apply(lambda x: strip_punct(str(x).strip()).title() if strip_punct(str(x).strip()) else '')
    odf['speaker_fname'] = odf['speaker'].apply(lambda x: strip_punct(str(x).strip()).split()[0].title() if strip_punct(str(x).strip()) else '')
    from .names import get_all_first_names_real_genderized, get_all_first_names_real, get_all_full_names_real

    gdf = get_all_first_names_real_genderized()
    valid_fnames = get_all_first_names_real()
    odf = odf.merge(gdf, left_on = 'speaker_fname', right_on = 'name', how='left').fillna('')
    return odf




def do_get_bechdel_score(imdbid):
    try:
        return json.loads(gethtml(f'http://bechdeltest.com/api/v1/getMovieByImdbId?imdbid={imdbid}'))
    except Exception as e:
        print(f'!! {e}')
        return {}

# do_get_bechdel_score('0147800')