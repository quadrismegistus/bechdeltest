from lltk.imports import *
from .imports import *

GENDERD={
    'male':'CM',  # genderize.io uses "male" and "female" to mean cis male and female
    'female':'CF', 

    'transmale':'TM', # our own categories
    'transfemale':'FM'
}

def get_imdb_id(imdb_id):
    # get imdb id
    idx=str(imdb_id)
    if idx.startswith('tt'): idx=idx[2:]
    while len(idx)<7: idx='0'+idx    
    return idx


class BechdelTestText(BaseText):
    @property
    def imdb(self):
        return get_imdb_id(self.meta.get('imdbID'))

    def get_dialogue(self):
        return parse_script(self.txt)

    def get_cast_from_imdb(self, force=False):
        qkey='cast_'+self.imdb
        res = self.qdb.get(qkey) if not force else None
        if res is None:

            url=f'https://www.imdb.com/title/tt{self.imdb}/fullcredits'
            html=gethtml(url)
            # parse with beautiful soup
            import bs4
            dom = bs4.BeautifulSoup(html, 'lxml')

            # find the cast_list table
            table = dom.select_one('.cast_list')

            # Quickly assemble a table
            old=[]
            for row in table('tr'):
                cells = row('td')
                if len(cells)<4: continue

                actor_cell = cells[1]
                char_cell = cells[3]

                odx={}
                odx['actor_name'] = actor_cell.text.replace('\n',' ').strip()
                # odx['actor_fname'] = odx['actor_name'].split()[0]
                # odx['actor_lname'] = odx['actor_name'].split()[-1]
                

                actor_link = actor_cell.select_one('a')
                actor_url = actor_link.attrs['href']
                if not actor_url.startswith('/name/'): continue

                odx['actor_id'] = actor_url[len('/name/'):].replace('/','')
                odx['actor_url'] = 'https://www.imdb.com' + actor_url


                odx['char_name'] = char_cell.text.replace('\n',' ').strip()
                # odx['char_fname'] = odx['char_name'].split()[0]
                # odx['char_lname'] = odx['char_name'].split()[-1]

                char_link = char_cell.select_one('a')
                odx['char_id'] = ''
                if char_link:
                    char_url = char_link.attrs['href']
                    if '/characters/' in char_url:
                        # odx['char_id'] = char_url.split('/characters/',1)[-1].split('?')[0]
                        odx['char_url'] = 'https://www.imdb.com' + char_url

                for xk in ['actor_name','char_name']:
                    while '  ' in odx[xk]:
                        odx[xk]=odx[xk].replace('  ',' ')

                old.append(odx)

            res=pd.DataFrame(old)
            self.qdb.set(qkey, res)

        res = res[~res.char_name.str.lower().str.contains('uncredited')]
        res = res[~res.actor_name.str.lower().str.contains('uncredited')]

        return res

    def get_cast_from_script(self):
        df = self.get_dialogue()
        odf = df.groupby('speaker').agg(dict(speaker='count', num_words=sum))
        odf = odf.rename({'speaker':'num_speeches'},axis=1)
        odf = odf.sort_values('num_speeches',ascending=False)
        return odf.reset_index()

    def get_cast_combined(self,force=False):
        qkey='cast_combined_'+self.imdb
        res = self.qdb.get(qkey) if not force else None
        charcounts={'num_speeches':Counter(), 'num_words':Counter()}
        if res is None:
            from thefuzz import fuzz
            from thefuzz import process

            dfcast1=self.get_cast_from_imdb(force=force)
            dfcast2=self.get_cast_from_script()
            choices = list(dfcast1.char_name)
            scorefunc = fuzz.token_set_ratio

            def choose_among_speakers(x,minval=70):
                if not x: return ''
                res=process.extract(x,choices,scorer=scorefunc)
                if res:
                    topres = res[0]
                    if topres[1]>=70:
                        return topres[0]
                return ''

            resd=defaultdict(set)
            for i,row in dfcast2.iterrows():
                spkr=row.speaker
                char = choose_among_speakers(spkr)
                if char and spkr:
                    resd[char] |= {spkr}
                    charcounts['num_speeches'][char]+=row.num_speeches
                    charcounts['num_words'][char]+=row.num_words
            
            dfcast1['speakers'] = [
                '; '.join(resd.get(char_name,[]))
                for char_name in dfcast1.char_name
            ]
            for numk in ['num_speeches','num_words']:
                dfcast1[numk] = [
                    charcounts[numk].get(cname,0)
                    for cname in dfcast1.char_name
                ]
            
            res = dfcast1
            res['rank_castlist'] = pd.Series(list(range(len(dfcast1))))+1
            res['rank_speaking'] = dfcast1.num_words.rank(ascending=False, method='min').apply(int)
            res = res.sort_values('rank_castlist')
            self.qdb.set(qkey,res)
        return res

    def get_cast(self,min_gender_prob=.95,force=False,min_speeches=2):
        qkey='cast_'+self.id
        res = self.qdb.get(qkey) if not force else None
        if res is None:

            dfcast = self.get_cast_combined(force=force)
            names = []


            for namek in ['actor_name','char_name']:
                fnames = [
                    name.split()[0] if name.split() else ''
                    for name in dfcast[namek]
                ]
                fnames_gendered = self.corpus.genderfy(fnames)
                onamek=namek.split('_')[0] + '_gender'
                dfcast[onamek] = [
                    GENDERD.get(gd.get('gender'),'') if gd.get('probability',0)>=min_gender_prob else ''
                    for gd in fnames_gendered
                ]
            
            res = dfcast
            self.qdb.set(qkey,res)
        
        odf = res
        odf['actor_url'] = odf['actor_id'].apply(lambda x: f'https://www.imdb.com/name/{x}')
        odf['char_url'] = odf['actor_id'].apply(lambda x: f'https://www.imdb.com/title/tt{self.imdb}/characters/{x}')
        odf=res.query(f'num_speeches>={min_speeches}')

        dfactors = self.corpus.metadata_actors()
        if len(dfactors):
            odf = odf[[col for col in odf if col=='actor_id' or col not in set(dfactors.columns)]].merge(dfactors, on='actor_id', how='left')
        
        return odf
            
    
    def get_speaker2char(self,dfcast=None):
        if dfcast is None: dfcast=self.get_cast()
        odx={}
        for spkrs,char in zip(dfcast.speakers, dfcast.char_name):
            for spkr in spkrs.split('; '):
                odx[spkr]=char
        return odx

    def get_cast_dialogue(self):
        dfcast = self.get_cast()
        speaker2char = self.get_speaker2char(dfcast)
        dfdial = self.get_dialogue()
        dfdial['char_name']=dfdial.speaker.apply(lambda x: speaker2char.get(x,''))
        dfdial = dfdial.reset_index().merge(dfcast, on = 'char_name', suffixes=('','_char'), how='inner').fillna('').set_index('line_num').sort_index()
        return dfdial

    def iter_interactions(self):
        df=self.get_cast_dialogue()
        # df['speaker'] = df['char_name']
        speaker2char = self.get_speaker2char()
        for scene,dfscene in sorted(df.reset_index().groupby('scene_num')):
            speech,direction,narration=[],[],[]
            numsp=0
            numw=0
            last_speaker=''


            for j in range(1,len(dfscene)):
                row_i = dfscene.iloc[j-1]
                row_j = dfscene.iloc[j]

                speaker = row_i.speaker
                if row_i.speech: speech+=[row_i.speech]
                if row_i.direction: direction+=[row_i.direction]
                if row_i.narration: narration+=[row_i.narration]
                numw+=row_i.num_words
                numsp+=1


                if row_i.speaker != row_j.speaker or j==(len(dfscene)-1):
                    target = last_speaker if last_speaker else row_j.speaker
                    yield dict(
                        line_num=row_i.line_num,
                        source=speaker2char.get(speaker,speaker),
                        target=speaker2char.get(target,target),
                        speech=' || '.join(speech),
                        direction=' || '.join(direction),
                        narration=' || '.join(narration),
                        scene_num=scene,
                        scene_desc=row_i.scene_desc,
                        num_speeches=numsp,
                        num_words=numw,
                        # **{
                        #     k:v
                        #     for k,v in dict(row_i).items()
                        #     if k not in {'source','target','speech','direction','narration'}
                        # }
                    )
                    
                    speech,direction,narration=[],[],[]
                    numsp=0
                    numw=0
                    last_speaker = speaker

                

    def get_interactions(self):
        return pd.DataFrame(self.iter_interactions()).set_index(['scene_num','scene_desc','line_num','source','target'])
                    
        



class BechdelTest(BaseCorpus):
    NAME='BechdelTest'
    ID='bechdel_test'
    TEXT_CLASS = BechdelTestText

    MOVIE_SCRIPT_DB_PATH = os.path.expanduser('~/data/moviedb/Movie-Script-Database/scripts')
    MOVIE_SCRIPT_DB_METADATA_URL = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vTTnPHG0eNrPcuR1pNWrGEy05LLFBf0Lpty6SEeinTn0Mq5bFsNg592qVPxXzZPSgTWjtK944en-yaD/pub?gid=572829200&single=true&output=csv'
    MOVIE_SCRIPT_DB_METADATA_KEYS = ['subgenre','genrenote','id','correct_imdb','source','file_name','script_url']
    IMDB_META_KEYS = ['rating', 'votes', 'imdbID', 'title', 'year']
    URL_ACTOR_METADATA = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQfw71n4k_a3p2gSj-Jhy-X3px1MQMf-Kr9EmIEM5bsZIJKEZu1koTLUy0ZY87oq0MH-XJqR4AYaNpQ/pub?gid=1183738268&single=true&output=csv'
    URL_CHAR_METADATA = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vRuBd6kmJroxVXWOIrr1Ebf5vKqN2LKbWijSZXyDBDdcHLd2Rp5Qo18OA8YWJ7_lMnDwDdPCHQWzFEF/pub?gid=1934597839&single=true&output=csv'
    
    def genderfy(self,names,force=False):
        named={}
        with self.qdb as db:
            if not force:
                for name in names:
                    if name in db:
                        named[name]=db[name]

            names_needed = list(set(names) - set(named.keys()))

            res = genderfy(names)
            for d in res:
                db[d['name']] = named[d['name']] = d
        
        return [named.get(name) for name in names]





    def compile(self):
        ## Step 1: Download movie script db
        pass

        # Step 2: Put up metadata on google docs
        pass

        # Step 3: Get all texts with a corrected imdb
        dfmeta = pd.read_csv(self.MOVIE_SCRIPT_DB_METADATA_URL, dtype='str').fillna('')
        dfmeta = dfmeta[dfmeta.correct_imdb != '']
        dfmeta = dfmeta[self.MOVIE_SCRIPT_DB_METADATA_KEYS]

        dfimdb = pd.DataFrame(pmap( self.get_imdb_meta, list(dfmeta.correct_imdb), num_proc=1))
        dfmeta = dfmeta.merge(dfimdb, on='correct_imdb')
        dfmeta['id'] = [f"{''.join(xx for xx in x if xx.isalnum())}-{y}" for x,y in zip(dfmeta.title, dfmeta.year)]
        dfmeta = dfmeta.set_index('id')

        # save metadata
        dfmeta.to_csv(self.path_metadata)

        # save files?
        for idx, fn,src in zip(dfmeta.index, dfmeta.file_name, dfmeta.source):
            origfn = os.path.join(self.MOVIE_SCRIPT_DB_PATH, 'unprocessed', src, fn+'.txt')
            if os.path.exists(origfn):
                ofn = os.path.join(self.path_texts, idx, 'text.txt')
                ensure_dir_exists(ofn)
                shutil.copyfile(origfn, ofn)

        return dfmeta
    
    def get_imdb_meta(self,imdb_id):
        idx = get_imdb_id(imdb_id)

        res = self.qdb.get(idx)
        if res is not None: return res
                

        from imdb import Cinemagoer
        ia = Cinemagoer()
        movie = ia.get_movie(idx)

        imdb_data = {k:movie[k] for k in self.IMDB_META_KEYS}
        imdb_data['correct_imdb']='tt'+idx
        self.qdb.set(idx,imdb_data)
        return imdb_data


    def get_casts(self):
        allcasts = pd.concat(
            t.get_cast().assign(
                id=t.id,
                title=t.meta.get('title'),
                year=t.meta.get('year'),
            ) for t in self.iter_texts(progress=True)
        ).fillna('').set_index(
            [
                'id','title','year',
                'rank_castlist','rank_speaking',
                'actor_name','actor_gender','actor_url',
                'char_name','speakers','char_gender','char_url',
            ]
        ).sort_values(
            ['year','title','id','rank_castlist','rank_speaking']
        ).drop(['actor_id','char_id'],axis=1)
        return allcasts


    def metadata_actors(self,force=False):
        if force or self._dfactors is None:
            self._dfactors = pd.read_csv(self.URL_ACTOR_METADATA, dtype='str').fillna('')
            self._dfactors['actor_id'] = self._dfactors.actor_url.apply(lambda x: x.split('/')[-1])

        return self._dfactors













#### PARSE SCRIPTING FUNCTIONS
import subprocess
import glob
import os
import numpy as np
import argparse
import re
import time
import codecs

# PROCESS ARGUMENTS
def read_args():
    parser = argparse.ArgumentParser(description='Script that parses a movie script pdf/txt into its constituent classes')
    parser.add_argument("-i", "--input", help="Path to script PDF/TXT to be parsed", required=True)
    parser.add_argument("-o", "--output", help="Path to directory for saving output", required=True)
    parser.add_argument("-a", "--abridged", help="Print abridged version (on/off)", default='off')
    parser.add_argument("-t", "--tags", help="Print class label tags (on/off)", default='off')
    parser.add_argument("-c", "--char", help="Print char info file (on/off)", default='off')
    args = parser.parse_args()
    if args.abridged not in ['on', 'off']: raise AssertionError("Invalid value. Choose either off or on")
    if args.tags not in ['on', 'off']: raise AssertionError("Invalid value. Choose either off or on")
    if args.char not in ['on', 'off']: raise AssertionError("Invalid value. Choose either off or on")
    return os.path.abspath(args.input), os.path.abspath(args.output), args.abridged, args.tags, args.char


# READ FILE
def read_txt(file_path):
    fid = codecs.open(file_path, mode='r', encoding='utf-8')
    txt_file = fid.read().splitlines()
    fid.close()
    return txt_file


# PROCESS FILE
def read_file(file_orig):
    if file_orig.endswith(".pdf"):
        file_name = file_orig.replace('.pdf', '.txt')
        subprocess.call(['pdftotext', '-enc', 'UTF-8', '-layout', file_orig, file_name])
        script_orig = read_txt(file_name)
        subprocess.call('rm "' + file_name + '"', shell=True)
    elif file_orig.endswith(".txt"):
        script_orig = read_txt(file_orig)
    else:
        raise AssertionError("Movie script file format should be either pdf or txt")
    
    return script_orig


# DETECT SCENE BOUNDARIES:
# LOOK FOR ALL-CAPS LINES CONTAINING "INT." OR "EXT."
def get_scene_bound(script_noind, tag_vec, tag_set, bound_set):
    bound_ind = [i for i, x in enumerate(script_noind) if tag_vec[i] not in tag_set and \
                                                        x.isupper() and \
                                                        any([y in x.lower() for y in bound_set])]
    if len(bound_ind) > 0:
        for x in bound_ind:
            tag_vec[x] = 'S'
    
    return tag_vec, bound_ind


# DETECT TRANSITIONS:
# LOOK FOR ALL-CAPS LINES PRECEDED BY NEWLINE, FOLLOWED BY NEWLINE AND CONTAINING "CUT " OR "FADE "
def get_trans(script_noind, tag_vec, tag_set, trans_thresh, trans_set):
    re_func = re.compile('[^a-zA-Z ]')
    trans_ind = [i for i, x in enumerate(script_noind) if tag_vec[i] not in tag_set\
                                                        and len(re_func.sub('', x).split()) < trans_thresh\
                                                        and any([y in x.lower() for y in trans_set])]
    if len(trans_ind) > 0:
        for x in trans_ind:
            tag_vec[x] = 'T'
    
    return tag_vec, trans_ind


# DETECT METADATA:
# LOOK FOR CONTENT PRECEDING SPECIFIC PHRASES THAT INDICATE BEGINNING OF MOVIE
def get_meta(script_noind, tag_vec, tag_set, meta_thresh, meta_set, sent_thresh, bound_ind, trans_ind):
    re_func = re.compile('[^a-zA-Z ]')
    met_ind = [i for i, x in enumerate(script_noind) if tag_vec[i] not in tag_set\
                                                        and i != 0 and i != (len(script_noind) - 1)\
                                                        and len(x.split()) < meta_thresh\
                                                        and len(re_func.sub('', script_noind[i - 1]).split()) == 0\
                                                        and len(re_func.sub('', script_noind[i + 1]).split()) == 0\
                                                        and any([y in x for y in meta_set])]
    sent_ind = [i for i, x in enumerate(script_noind) if tag_vec[i] not in tag_set\
                                                        and i != 0 and i != (len(script_noind) - 1)\
                                                        and len(x.split()) > sent_thresh\
                                                        and len(script_noind[i - 1].split()) == 0\
                                                        and len(script_noind[i + 1].split()) > 0]
    meta_ind = sorted(met_ind + bound_ind + trans_ind + sent_ind)
    if len(meta_ind) > 0:
        for i, x in enumerate(script_noind[: meta_ind[0]]):
            if len(x.split()) > 0:
                tag_vec[i] = 'M'
    
    return tag_vec


# DECOMPOSE LINE WITH DIALOGUE AND DIALOGUE METADATA INTO INDIVIDUAL CLASSES
def separate_dial_meta(line_str):
    if '(' in line_str and ')' in line_str:
        bef_par_str = ' '.join(line_str.split('(')[0].split())
        in_par_str = ' '.join(line_str.split('(')[1].split(')')[0].split())
        rem_str = ')'.join(line_str.split(')')[1: ])
    else:
        bef_par_str = line_str
        in_par_str = ''
        rem_str = ''
    
    return bef_par_str, in_par_str, rem_str


# DETECT CHARACTER-DIALOGUE BLOCKS:
# CHARACTER IS ALL-CAPS LINE PRECEDED BY NEWLINE AND NOT FOLLOWED BY A NEWLINE
# DIALOGUE IS WHATEVER IMMEDIATELY FOLLOWS CHARACTER
# EITHER CHARACTER OR DIALOGUE MIGHT CONTAIN DILAOGUE METADATA; WILL BE DETECTED LATER
def get_char_dial(script_noind, tag_vec, tag_set, char_max_words):
    char_ind = [i for i, x in enumerate(script_noind) if tag_vec[i] not in tag_set and any([y.isupper() for y in x.split()])\
                                                        and i != 0 and i != (len(script_noind) - 1)\
                                                        # and len(script_noind[i - 1].split()) == 0\
                                                        and len(script_noind[i + 1].split()) > 0\
                                                        and len(x.split()) < char_max_words\
                                                        and any([separate_dial_meta(x)[y] for y in [0, 2]])]
    if char_ind[-1] < (len(script_noind) - 1):
        char_ind += [len(script_noind) - 1]
    else:
        char_ind += [len(script_noind)]
    
    for x in range(len(char_ind) - 1):
        tag_vec[char_ind[x]] = 'C'
        dial_flag = 1
        while dial_flag > 0:
            line_ind = char_ind[x] + dial_flag
            if len(script_noind[line_ind].split()) > 0 and line_ind < char_ind[x + 1]:
                dial_str, dial_meta_str, rem_str = separate_dial_meta(script_noind[line_ind])
                if dial_str != '' or rem_str != '':
                    tag_vec[line_ind] = 'D'
                else:
                    tag_vec[line_ind] = 'E'
                
                dial_flag += 1
            else:
                dial_flag = 0
    
    return tag_vec


# DETECT SCENE DESCRIPTION
# LOOK FOR REMAINING LINES THAT ARE NOT PAGE BREAKS
def get_scene_desc(script_noind, tag_vec, tag_set):
    desc_ind = [i for i, x in enumerate(script_noind) if tag_vec[i] not in tag_set and \
                                                            len(x.split()) > 0 and\
                                                            not x.strip('.').isdigit()]
    for x in desc_ind:
        tag_vec[x] = 'N'
    
    return tag_vec


# CHECK IF LINES CONTAIN START OF PARENTHESES
def par_start(line_set):
    return [i for i, x in enumerate(line_set) if '(' in x]


# CHECK IF LINES CONTAIN START OF PARENTHESES
def par_end(line_set):
    return [i for i, x in enumerate(line_set) if ')' in x]


# COMBINE MULTI-LINE CLASSES, SPLIT MULTI-CLASS LINES
def combine_tag_lines(tag_valid, script_valid):
    tag_final = []
    script_final = []
    changed_tags = [x for x in tag_valid]
    for i, x in enumerate(tag_valid):
        if x in ['M', 'T', 'S']:
            # APPEND METADATA, TRANSITION AND SCENE BOUNDARY LINES AS THEY ARE
            tag_final.append(x)
            script_final.append(script_valid[i])
        elif x in ['C', 'D', 'N']:
            # IF CHARACTER, DIALOGUE OR SCENE DESCRIPTION CONSIST OF MULTIPLE LINES, COMBINE THEM
            if i == 0 or x != tag_valid[i - 1]:
                # INITIALIZE IF FIRST OF MULTIPLE LINES
                to_combine = []
                comb_ind = []
            
            to_combine += script_valid[i].split()
            comb_ind.append(i)
            if i == (len(tag_valid) - 1) or x != tag_valid[i + 1]:
                combined_str = ' '.join(to_combine)
                if x == 'N':
                    # IF SCENE DESCRIPTION, WRITE AS IT IS
                    tag_final.append(x)
                    script_final.append(combined_str)
                else:
                    _, in_par, _ = separate_dial_meta(combined_str)
                    if in_par != '':
                        # FIND LINES CONTAINING DIALOGUE METADATA
                        comb_lines = [script_valid[j] for j in comb_ind]
                        dial_meta_ind = []
                        while len(par_start(comb_lines)) > 0 and len(par_end(comb_lines)) > 0:
                            start_ind = comb_ind[par_start(comb_lines)[0]]
                            end_ind = comb_ind[par_end(comb_lines)[0]]
                            dial_meta_ind.append([start_ind, end_ind])
                            comb_ind = [x for x in comb_ind if x > end_ind]
                            comb_lines = [script_valid[j] for j in comb_ind]
                        
                        # REPLACE OLD TAGS WITH DIALOGUE METADATA TAGS
                        for dial_ind in dial_meta_ind:
                            for change_ind in range(dial_ind[0], (dial_ind[1] + 1)):
                                changed_tags[change_ind] = 'E'
                        
                        # EXTRACT DIALOGUE METADATA
                        dial_meta_str = ''
                        char_dial_str = ''
                        while '(' in combined_str and ')' in combined_str:
                            before_par, in_par, combined_str = separate_dial_meta(combined_str)
                            char_dial_str += ' ' + before_par
                            dial_meta_str += ' ' + in_par
                        
                        char_dial_str += ' ' + combined_str
                        char_dial_str = ' '.join(char_dial_str.split())
                        dial_meta_str = ' '.join(dial_meta_str.split())
                        if x == 'C':
                            # IF CHARACTER, APPEND DIALOGUE METADATA
                            tag_final.append(x)
                            script_final.append(' '.join(char_dial_str.split()))
                            tag_final.append('E')
                            script_final.append(dial_meta_str)
                        elif x == 'D':
                            # IF DIALOGUE, PREPEND DIALOGUE METADATA
                            tag_final.append('E')
                            script_final.append(dial_meta_str)
                            tag_final.append(x)
                            script_final.append(' '.join(char_dial_str.split()))
                    else:
                        # IF NO DIALOGUE METADATA, WRITE AS IT IS
                        tag_final.append(x)
                        script_final.append(combined_str)
        elif x == 'E':
            # IF DIALOGUE METADATA LINE, WRITE WITHOUT PARENTHESIS
            split_1 = script_valid[i].split('(')
            split_2 = split_1[1].split(')')
            dial_met = split_2[0]
            tag_final.append('E')
            script_final.append(dial_met)
    
    return tag_final, script_final, changed_tags


# CHECK FOR UN-MERGED CLASSES
def find_same(tag_valid):
    same_ind_mat = np.empty((0, 2), dtype=int)
    if len(tag_valid) > 1:
        check_start = 0
        check_end = 1
        while check_start < (len(tag_valid) - 1):
            if tag_valid[check_start] != 'M' and tag_valid[check_start] == tag_valid[check_end]:
                while check_end < len(tag_valid) and tag_valid[check_start] == tag_valid[check_end]:
                    check_end += 1
                
                append_vec = np.array([[check_start, (check_end - 1)]], dtype=int)
                same_ind_mat = np.append(same_ind_mat, append_vec, axis=0)
                check_end += 1
                check_start = check_end - 1
            else:
                check_start += 1
                check_end += 1
    
    return same_ind_mat


# MERGE CONSECUTIVE IDENTICAL CLASSES
def merge_tag_lines(tag_final, script_final):
    merge_ind = find_same(tag_final)
    if merge_ind.shape[0] > 0:
        # PRE-MERGE DISSIMILAR
        tag_merged = tag_final[: merge_ind[0, 0]]
        script_merged = script_final[: merge_ind[0, 0]]
        for ind in range(merge_ind.shape[0] - 1):
            # CURRENT MERGE SIMILAR
            tag_merged += [tag_final[merge_ind[ind, 0]]]
            script_merged += [' '.join(script_final[merge_ind[ind, 0]: (merge_ind[ind, 1] + 1)])]
            # CURRENT MERGE DISSIMILAR
            tag_merged += tag_final[(merge_ind[ind, 1] + 1): merge_ind[(ind + 1), 0]]
            script_merged += script_final[(merge_ind[ind, 1] + 1): merge_ind[(ind + 1), 0]]
        
        # POST-MERGE SIMILAR
        tag_merged += [tag_final[merge_ind[-1, 0]]]
        script_merged += [' '.join(script_final[merge_ind[-1, 0]: (merge_ind[-1, 1] + 1)])]
        # POST-MERGE DISSIMILAR
        tag_merged += tag_final[(merge_ind[-1, 1] + 1): ]
        script_merged += script_final[(merge_ind[-1, 1] + 1): ]
    else:
        tag_merged = tag_final
        script_merged = script_final
    
    return tag_merged, script_merged


# CHECK FOR DIALOGUE METADATA PRECEDING DIALOGUE
def find_arrange(tag_valid):
    c_ind = [i for i,x in enumerate(tag_valid) if x == 'C']
    c_segs = []
    arrange_ind = []
    invalid_set = [['C', 'E', 'D'], ['C', 'D', 'E', 'D']]
    if len(c_ind) > 0:
        # BREAK UP INTO C-* BLOCKS
        if c_ind[0] != 0:
            c_segs.append(tag_valid[: c_ind[0]])
        
        for i in range((len(c_ind) - 1)):
            c_segs.append(tag_valid[c_ind[i]: c_ind[i + 1]])
        
        c_segs.append(tag_valid[c_ind[-1]: ])
        # RE-ARRANGE IN BLOCKS IF REQUIRED
        for i in range(len(c_segs)):
            inv_flag = 0
            if len(c_segs[i]) > 2:
                if any([c_segs[i][j: (j + len(invalid_set[0]))] == invalid_set[0] \
                        for j in range(len(c_segs[i]) - len(invalid_set[0]) + 1)]):
                    inv_flag = 1
            
            if inv_flag == 0 and len(c_segs[i]) > 3:
                if any([c_segs[i][j: (j + len(invalid_set[1]))] == invalid_set[1] \
                        for j in range(len(c_segs[i]) - len(invalid_set[1]) + 1)]):
                    inv_flag = 1
            
            if inv_flag == 1:
                arrange_ind.append(i)
    
    return c_segs, arrange_ind


# REARRANGE DIALOGUE METADATA TO ALWAYS APPEAR AFTER DIALOGUE
def rearrange_tag_lines(tag_merged, script_merged):
    tag_rear = []
    script_rear = []
    char_blocks, dial_met_ind = find_arrange(tag_merged)
    if len(dial_met_ind) > 0:
        last_ind = 0
        for ind in range(len(char_blocks)):
            if ind in dial_met_ind:
                # ADD CHARACTER
                tag_rear += ['C']
                script_rear.append(script_merged[last_ind])
                # ADD DIALOGUE
                if 'D' in char_blocks[ind]:
                    tag_rear += ['D']
                    script_rear.append(' '.join([script_merged[last_ind + i] \
                                        for i, x in enumerate(char_blocks[ind]) if x == 'D']))
                
                # ADD DIALOGUE METADATA
                if 'E' in char_blocks[ind]:
                    tag_rear += ['E']
                    script_rear.append(' '.join([script_merged[last_ind + i] \
                                        for i, x in enumerate(char_blocks[ind]) if x == 'E']))
                # ADD REMAINING
                tag_rear += [x for x in char_blocks[ind] if x not in ['C', 'D', 'E']]
                script_rear += [script_merged[last_ind + i] \
                                for i, x in enumerate(char_blocks[ind]) if x not in ['C', 'D', 'E']]
            else:
                tag_rear += char_blocks[ind]
                script_rear += script_merged[last_ind: (last_ind + len(char_blocks[ind]))]
            
            last_ind += len(char_blocks[ind])
    
    return tag_rear, script_rear


# PARSER FUNCTION
def parse(file_orig, save_dir, abr_flag, tag_flag, char_flag, save_name=None, abridged_name=None, tag_name=None):
    #------------------------------------------------------------------------------------
    # DEFINE
    tag_set = ['S', 'N', 'C', 'D', 'E', 'T', 'M']
    meta_set = ['BLACK', 'darkness']
    bound_set = ['int.', 'ext.', 'int ', 'ext ']
    trans_set = ['cut', 'fade', 'transition', 'dissolve']
    char_max_words = 5
    meta_thresh = 2
    sent_thresh = 5
    trans_thresh = 6
    # READ PDF/TEXT FILE
    script_orig = read_file(file_orig)
    # REMOVE INDENTS
    alnum_filter = re.compile('[\W_]+', re.UNICODE)
    script_noind = []
    for script_line in script_orig:
        if len(script_line.split()) > 0 and alnum_filter.sub('', script_line) != '':
            script_noind.append(' '.join(script_line.split()))
        else:
            script_noind.append('')
    
    num_lines = len(script_noind)
    tag_vec = np.array(['0' for x in range(num_lines)])
    #------------------------------------------------------------------------------------
    # DETECT SCENE BOUNDARIES
    tag_vec, bound_ind = get_scene_bound(script_noind, tag_vec, tag_set, bound_set)
    # DETECT TRANSITIONS
    tag_vec, trans_ind = get_trans(script_noind, tag_vec, tag_set, trans_thresh, trans_set)
    # DETECT METADATA
    tag_vec = get_meta(script_noind, tag_vec, tag_set, meta_thresh, meta_set, sent_thresh, bound_ind, trans_ind)
    # DETECT CHARACTER-DIALOGUE
    tag_vec = get_char_dial(script_noind, tag_vec, tag_set, char_max_words)
    # DETECT SCENE DESCRIPTION
    tag_vec = get_scene_desc(script_noind, tag_vec, tag_set)
    #------------------------------------------------------------------------------------
    # REMOVE UN-TAGGED LINES
    nz_ind_vec = np.where(tag_vec != '0')[0]
    tag_valid = []
    script_valid = []
    for i, x in enumerate(tag_vec):
        if x != '0':
            tag_valid.append(x)
            script_valid.append(script_noind[i])
    
    # UPDATE TAGS
    tag_valid, script_valid, changed_tags = combine_tag_lines(tag_valid, script_valid)
    for change_ind in range(len(nz_ind_vec)):
        if tag_vec[nz_ind_vec[change_ind]] == 'D':
            tag_vec[nz_ind_vec[change_ind]] = changed_tags[change_ind]
    
    # SAVE TAGS TO FILE
    if tag_flag == 'on':
        if tag_name is None:
            tag_name = os.path.join(save_dir, '.'.join(file_orig.split('/')[-1].split('.')[: -1]) + '_tags.txt')
        else:
            tag_name = os.path.join(save_dir, tag_name)
        
        np.savetxt(tag_name, tag_vec, fmt='%s', delimiter='\n')
    
    # FORMAT TAGS, LINES
    max_rev = 0
    while find_same(tag_valid).shape[0] > 0 or len(find_arrange(tag_valid)[1]) > 0:
        tag_valid, script_valid = merge_tag_lines(tag_valid, script_valid)
        tag_valid, script_valid = rearrange_tag_lines(tag_valid, script_valid)
        max_rev += 1
        if max_rev == 1000: raise AssertionError("Too many revisions. Something must be wrong.")
    
    #------------------------------------------------------------------------------------
    # WRITE PARSED SCRIPT TO FILE
    if save_name is None:
        save_name = os.path.join(save_dir, '.'.join(file_orig.split('/')[-1].split('.')[: -1]) + '_parsed.txt')
    else:
        save_name = os.path.join(save_dir, save_name)
    
    fid = open(save_name, 'w')
    for tag_ind in range(len(tag_valid)):
        _ = fid.write(tag_valid[tag_ind] + ': ' + script_valid[tag_ind] + '\n')
    
    fid.close()
    #------------------------------------------------------------------------------------
    # CREATE CHARACTER=>DIALOGUE ABRIDGED VERSION
    if abr_flag == 'on':
        fid = open(save_name, 'r')
        parsed_script = fid.read().splitlines()
        fid.close()
        if abridged_name is None:
            abridged_name = os.path.join(save_dir, '.'.join(file_orig.split('/')[-1].split('.')[: -1]) + '_abridged.txt')
        else:
            abridged_name = os.path.join(save_dir, abridged_name)
        
        abridged_ind = [i for i, x in enumerate(parsed_script) if x.startswith('C:') and \
                                                                parsed_script[i + 1].startswith('D:')]
        fid = open(abridged_name, 'w')
        for i in abridged_ind:
            char_str = ' '.join(parsed_script[i].split('C:')[1].split())
            dial_str = ' '.join(parsed_script[i + 1].split('D:')[1].split())
            _ = fid.write(''.join([char_str, '=>', dial_str, '\n']))
        
        fid.close()
    
    #------------------------------------------------------------------------------------
    # CREATE CHAR INFO FILE
    if char_flag == 'on':
        tag_str_vec = np.array(tag_valid)
        script_vec = np.array(script_valid)
        char_ind = np.where(tag_str_vec == 'C')[0]
        char_set = sorted(set(script_vec[char_ind]))
        charinfo_vec = []
        for char_id in char_set:
            spk_ind = list(set(np.where(script_vec == char_id)[0]) & set(char_ind))
            if len(spk_ind) > 0:
                num_lines = len([i for i in spk_ind if i != (len(tag_str_vec) - 1) and \
                                                    tag_str_vec[i + 1] == 'D'])
                charinfo_str = char_id + ': ' + str(num_lines) + '|'.join([' ', ' ', ' ', ' '])
                charinfo_vec.append(charinfo_str)
        
        charinfo_name = os.path.join(save_dir, '.'.join(file_orig.split('/')[-1].split('.')[: -1]) + '_charinfo.txt')
        np.savetxt(charinfo_name, charinfo_vec, fmt='%s', delimiter='\n')


# MAIN FUNCTION
if __name__ == "__main__":
    file_orig, save_dir, abr_flag, tag_flag, char_flag = read_args()
    parse(file_orig, save_dir, abr_flag, tag_flag, char_flag)





## RYAN
PARSE_SCRIPT_KEY=dict(
    S = 'Scene',
    N = 'Scene description',
    C = 'Character',
    D = 'Dialogue',
    E = 'Dialogue metadata',
    T = 'Transition',
    M = 'Metadata'
)


def parse_script_iter(script_orig):
    # DEFINE
    tag_set = ['S', 'N', 'C', 'D', 'E', 'T', 'M']
    meta_set = ['BLACK', 'darkness']
    bound_set = ['int.', 'ext.', 'int ', 'ext ']
    trans_set = ['cut', 'fade', 'transition', 'dissolve']
    char_max_words = 5
    meta_thresh = 2
    sent_thresh = 5
    trans_thresh = 6
    # REMOVE INDENTS
    alnum_filter = re.compile('[\W_]+', re.UNICODE)
    script_noind = []
    for script_line in script_orig.split('\n'):
        if len(script_line.split()) > 0 and alnum_filter.sub('', script_line) != '':
            script_noind.append(' '.join(script_line.split()))
        else:
            script_noind.append('')
    
    num_lines = len(script_noind)
    tag_vec = np.array(['0' for x in range(num_lines)])
    #------------------------------------------------------------------------------------
    # DETECT SCENE BOUNDARIES
    tag_vec, bound_ind = get_scene_bound(script_noind, tag_vec, tag_set, bound_set)
    # DETECT TRANSITIONS
    tag_vec, trans_ind = get_trans(script_noind, tag_vec, tag_set, trans_thresh, trans_set)
    # DETECT METADATA
    tag_vec = get_meta(script_noind, tag_vec, tag_set, meta_thresh, meta_set, sent_thresh, bound_ind, trans_ind)
    # DETECT CHARACTER-DIALOGUE
    tag_vec = get_char_dial(script_noind, tag_vec, tag_set, char_max_words)
    # DETECT SCENE DESCRIPTION
    tag_vec = get_scene_desc(script_noind, tag_vec, tag_set)
    #------------------------------------------------------------------------------------
    # REMOVE UN-TAGGED LINES
    nz_ind_vec = np.where(tag_vec != '0')[0]
    tag_valid = []
    script_valid = []
    for i, x in enumerate(tag_vec):
        if x != '0':
            tag_valid.append(x)
            script_valid.append(script_noind[i])
    
    # UPDATE TAGS
    tag_valid, script_valid, changed_tags = combine_tag_lines(tag_valid, script_valid)
    for change_ind in range(len(nz_ind_vec)):
        if tag_vec[nz_ind_vec[change_ind]] == 'D':
            tag_vec[nz_ind_vec[change_ind]] = changed_tags[change_ind]
    
    # # SAVE TAGS TO FILE
    # if tag_flag == 'on':
    # 	if tag_name is None:
    # 		tag_name = os.path.join(save_dir, '.'.join(file_orig.split('/')[-1].split('.')[: -1]) + '_tags.txt')
    # 	else:
    # 		tag_name = os.path.join(save_dir, tag_name)
        
    # 	np.savetxt(tag_name, tag_vec, fmt='%s', delimiter='\n')
    
    # FORMAT TAGS, LINES
    max_rev = 0
    while find_same(tag_valid).shape[0] > 0 or len(find_arrange(tag_valid)[1]) > 0:
        tag_valid, script_valid = merge_tag_lines(tag_valid, script_valid)
        tag_valid, script_valid = rearrange_tag_lines(tag_valid, script_valid)
        max_rev += 1
        if max_rev == 1000: raise AssertionError("Too many revisions. Something must be wrong.")
    
    #------------------------------------------------------------------------------------
    # RETURN
    scene_num=0
    line_num=0
    
    def reset_odx(): return dict((xk,'') for xk in PARSE_SCRIPT_KEY)
    odx = reset_odx()
    last_speaker = None

    for tag_ind in range(len(tag_valid)):
        tag=tag_valid[tag_ind]
        text=script_valid[tag_ind]
        
        if tag=='S':
            scene_num+=1
            odx=dict((xk,'') for xk in PARSE_SCRIPT_KEY)
        if not scene_num: continue

        odx[tag] = text

        if tag == 'D':
            line_num+=1
            speaker=odx.get('C','')
            speaker=speaker.replace(':', '')
            for xx in [':','O.S.','V.O.',"'S VOICE","'S COM VOICE"]:
                speaker=speaker.replace(xx,'')
            # speaker = ' '.join(x for x in speaker.split() if x==x.upper())
            speaker = noPunc(speaker.strip()).strip()
            if last_speaker and speaker in {'CONTINUED', "CONT'D", "CONTD"}:
                speaker=last_speaker
            last_speaker=speaker

            yield dict(
                speaker=speaker,
                speech=text,

                direction=odx.get('E',''),
                narration=odx.get('N'),
                scene_desc = odx['S'],
                scene_num=scene_num,
                line_num=line_num,
                num_words = len(text.strip().split())
            )
            odx = {**reset_odx(), **dict(S=odx['S'])}
    
def parse_script(script_orig):
    import pandas as pd
    return pd.DataFrame(parse_script_iter(script_orig)).set_index('line_num')

    


def fixname(x):
    while '  ' in x: x=x.replace('  ',' ')
    return x