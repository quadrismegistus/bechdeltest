from lltk.imports import *
from .imports import *

GENDERD={
    'male':'CM',  # genderize.io uses "male" and "female" to mean cis male and female
    'female':'CF', 

    'transmale':'TM', # our own categories
    'transfemale':'FM'
}





class BechdelTestText(BaseText):
    @property
    def imdb(self): return get_imdb_id(self.meta.get('imdbID'))
    @property
    def path_dialogue(self): return os.path.join(self.path,f'{self.id}.dialogue.csv')
    @property
    def path_cast(self): return os.path.join(self.path,f'{self.id}.cast.csv')

    def get_dialogue(self,force=False):
        dfdial = pd.DataFrame()
        if not force: dfdial = read_df_anno(self.path_dialogue)
        if dfdial is None or not len(dfdial):
            if self.txt: dfdial = self.get_dialogue_txt()
            if not len(dfdial): dfdial = self.get_dialogue_fandom()
            if len(dfdial):
                ensure_dir_exists(self.path_dialogue)
                dfdial=dfdial[dfdial.speaker!='']
                dfdial.to_csv(self.path_dialogue,index=False)
        
        dfdial=dfdial.set_index('line_num')
        dfdial['num_words'] = dfdial.speech.apply(lambda text: len(str(text).strip().split()))
        return dfdial
    dialogue=get_dialogue
        

    def get_dialogue_txt(self):
        return parse_script(self.txt)

    def get_dialogue_fandom(self):
        url = self.meta.get('script_url')
        if not url: return ''
        if 'fandom' not in url or not url.endswith('/Transcript'): return ''
        
        htm = self.qdb.get(url)
        if not htm:
            htm = gethtml(url)
            self.qdb.set(url,htm)
        
        import bs4
        dom = bs4.BeautifulSoup(htm).select_one('#mw-content-text')
        if not dom: return pd.DataFrame()
        
        sep=' || '
        o=[]
        scene_num=1
        for i,p in enumerate(dom('p')):
            directions = [tag.extract().text.strip() for tag in p('i')]
            speakers = [tag.extract().text.replace(':','').strip() for tag in p('b')]
            speech = str(p.text).strip()

            if any(['scene' in x and 'change' in x for x in directions]):
                scene_num+=1

            odx=dict(
                line_num=i+1,
                scene_num=scene_num,
                speaker=sep.join(x for x in speakers if x),
                speech=speech,
                direction=sep.join(x for x in directions if x),
                narration='',
                scene_desc='',

            )
            o.append(odx)
        odf=pd.DataFrame(o)

        return odf

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

        # res = res[~res.char_name.str.lower().str.contains('uncredited')]
        # res = res[~res.actor_name.str.lower().str.contains('uncredited')]

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

    def get_cast(self,min_gender_prob=.95,force=False,min_speeches=0):
        # qkey='cast_'+self.id
        # res = self.qdb.get(qkey) if not force else None
        ofn = self.path_cast
        if force or not os.path.exists(ofn):
            dfcast = self.get_cast_combined(force=force).fillna('')
            names = []


            for namek in ['actor_name','char_name']:
                fnames = [
                    name.split()[0] if name.split() else ''
                    for name in dfcast[namek]
                ]
                fnames_gendered = self.corpus.genderfy(fnames)
                onamek=namek.split('_')[0] + '_gender'
                dfcast[onamek] = dfcast[onamek+'_auto'] = [
                    GENDERD.get(gd.get('gender'),'') if gd.get('probability',0)>=min_gender_prob else ''
                    for gd in fnames_gendered
                ]

            odf = res = dfcast

            odf['actor_url'] = odf['actor_id'].apply(lambda x: f'https://www.imdb.com/name/{x}')
            odf['char_url'] = odf['actor_id'].apply(lambda x: f'https://www.imdb.com/title/tt{self.imdb}/characters/{x}')
            odf=res.query(f'num_speeches>={min_speeches}')
            odf['char_id'] = odf['actor_id']

            meta_actors = pd.concat([self.metadata_cast(), self.corpus.metadata_actors()]).fillna('').drop_duplicates('actor_id',keep='first')
            actor_id2feats = meta_actors.set_index('actor_id').T.to_dict()
            
            dfchars = pd.concat([self.metadata_cast(), self.corpus.metadata_chars().query(f'id == "{self.id}"')]).fillna('')
            char_id2feats={}
            if len(dfchars) and 'actor_id' in set(dfchars.columns):
                dfchars=dfchars.drop_duplicates('actor_id',keep='first')
                dfchars['char_id'] = dfchars['actor_id']
                char_id2feats=dfchars.set_index('char_id').T.to_dict()


            o=[]
            for row in odf.to_dict('records'):
                for k,v in char_id2feats.get(row['char_id'], {}).items(): 
                    if v and k.startswith('char_') or k.startswith('actor_') or k in {'speakers'}:
                        row[k]=v
                for k,v in actor_id2feats.get(row['actor_id'], {}).items():
                    if v and k.startswith('actor_'):
                        row[k]=v
                o.append(row)
            odf = pd.DataFrame(o)
            if 'id' in set(odf.columns): odf=odf.drop('id',axis=1)
            odf.to_csv(ofn,index=False)
            res = odf
        else:
            res = pd.read_csv(ofn).fillna('')
        
        return res
    cast = get_cast

    def metadata_cast(self):
        if self.id in self.corpus.get_urls():
            url = self.corpus.get_urls().get(self.id)
            df = pd.read_csv(url).fillna('')
            return df
        return pd.DataFrame()
            
    
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

    def iter_networks(self):

        # utility
        def recolor_network(g,source='',target=''):
            for node in g.nodes(): 
                if g.nodes[node]['actor_gender']=='CM':
                    g.nodes[node]['color']='green'
                elif g.nodes[node]['actor_gender']=='CF':
                    g.nodes[node]['color']='purple'
                else:
                    g.nodes[node]['color']='gray'
            
            if source and target and g.has_edge(source,target):
                for a,b,d in g.edges(data=True):
                    # if g.nodes[a]['actor_gender'] == 'CM' and g.nodes[b]['actor_gender']=='CM':
                        # g.edges[(a,b)]['color']='green'
                    if g.nodes[a]['actor_gender'] == 'CF' and g.nodes[b]['actor_gender']=='CF':
                        g.edges[(a,b)]['color']='purple'
                    else:
                        g.edges[(a,b)]['color']='black'
                
                g.edges[source,target]['color']='red'
            return g


        import networkx as nx
        G = nx.DiGraph()
        dfcast = self.get_cast()
        u2d={}

        def rename(x):
            x=x.split('/')[0].split('(')[0]
            while '  ' in x: x=x.replace('  ',' ')
            return x.strip()

        for d in dfcast.to_dict('records'):
            u=rename(d['char_name'])
            u2d[u]=d
        
        ld = list(self.iter_interactions())
        for t,d in enumerate(tqdm(ld)):
            u,v=rename(d['source']),rename(d['target'])
            d['weight'] = d['num_speeches']
            if u != v:
                if not G.has_node(u): G.add_node(u, **u2d[u])
                if not G.has_node(v): G.add_node(v, **u2d[v])
                if not G.has_edge(u,v):
                    G.add_edge(u,v,**d)
                else:
                    for kk,vv in d.items():
                        if type(vv)==str: vv = ' || '+vv
                        v0 = G.edges[(u,v)].get(kk)
                        G.edges[(u,v)][kk]=v0+vv if type(v0)==type(vv) and type(v0) in {str,int} else vv
                
                yield recolor_network(G,u,v)

            # if t>25: break

    def get_network(self):
        for g in self.iter_networks(): pass
        return g

    def draw_network(self,g=None,**opts):
        from lltk.model.networks import draw_nx
        if g is None: g=self.get_network()
        return draw_nx(g,**opts)

    def draw_networks(self,**opts):
        from lltk.model.networks import draw_nx_dynamic
        return draw_nx_dynamic(
            self.iter_networks(),
            ofn=f'fig.dynamic_graph.{self.id}.mp4',
            final_g=self.get_network(),
            **opts
        )

    def get_bechdel_scores(self,force=False):
        
        qkey='bechdelscores_'+self.id
        res = self.qdb.get(qkey) if not force else None
        if res is None:
            import bs4

            d = do_get_bechdel_score(self.imdb)
            id = d.get('id')
            if id:
                url=f'https://bechdeltest.com/view/{id}'
                htm=gethtml(url).replace('\r\n','\n').replace('\r','\n')
                dom=bs4.BeautifulSoup(htm,'lxml')
                if htm:
                    paras=list(dom('p'))
                    comments=list(dom.select('.comment'))
                    if paras and comments:
                        d['msg'] = paras[0].text.strip()
                        d['explanation'] = comments[0].text.strip()
                        d['comments'] = '\n----------\n'.join([x.text.strip() for x in comments[1:]]) if len(comments)>1 else ''
                        while '\n\n\n' in d['comments']: d['comments']=d['comments'].replace('\n\n\n','\n\n')

                        res = d
                        self.qdb.set(qkey,res)
        
        return res
        
    def get_ratios(self):
        dfcast = self.get_cast()
        dfcast['actor_not_CM'] = dfcast['actor_gender'].apply(lambda x: None if not x else x!='CM')
        dfcast['actor_not_W'] = dfcast['actor_race'].apply(lambda x: None if not x else x!='W')
        dfcast['num_chars']=1
        
        odf=dfcast.groupby('actor_not_CM').agg(
            dict(
                num_speeches=sum,
                num_words=sum,
                num_chars=sum,
                rank_castlist=np.median,
                rank_speaking=np.median
            )
        )
        odx={}
        if len(odf)==2:
            nums_CM = odf.loc[False]
            nums_notCM = odf.loc[True]
            ratios_CM_to_notCM = dict(nums_CM / nums_notCM)
            for k,v in nums_CM.items(): odx[k+'__gender_CM']=v
            for k,v in nums_notCM.items(): odx[k+'__gender_notCM']=v
            for k,v in ratios_CM_to_notCM.items():
                odx[k+'__gender_CM_to_notCM']=v
        
        odf=dfcast.groupby('actor_not_W').sum()
        if len(odf)==2:
            nums_CM = odf.loc[False]
            nums_notCM = odf.loc[True]
            ratios_CM_to_notCM = dict(nums_CM / nums_notCM)
            for k,v in nums_CM.items(): odx[k+'__race_W']=v
            for k,v in nums_notCM.items(): odx[k+'__race_notW']=v
            for k,v in ratios_CM_to_notCM.items():
                odx[k+'__race_W_to_notW']=v
        
        odx={k:v for k,v in sorted(odx.items()) if not k.startswith('actor_')}
        return odx

    def show_ratios(self):
        rd = self.get_ratios()
        
        o=f"""
### Number of characters

* **{rd.get("num_chars__gender_CM",0):.0f}** cismen to **{rd.get("num_chars__gender_notCM",0):.0f}** non-cismen (**{rd.get("num_chars__gender_CM_to_notCM",0):.1f}x**)
* **{rd.get("num_chars__race_W",0):.0f}** white people to **{rd.get("num_chars__race_notW",0):.0f}** people of color (**{rd.get("num_chars__race_W_to_notW",0):.1f}x**)

### Number of speeches

* **{rd.get("num_speeches__gender_CM",0):.0f}** speeches by cismen to **{rd.get("num_speeches__gender_notCM",0):.0f}** by non-cismen (**{rd.get("num_speeches__gender_CM_to_notCM",0):.1f}x**)
* **{rd.get("num_speeches__race_W",0):.0f}** speeches by white people to **{rd.get("num_speeches__race_notW",0):.0f}** by people of color (**{rd.get("num_speeches__race_W_to_notW",0):.1f}x**)

### Number of words spoken

* **{rd.get("num_words__gender_CM",0):,.0f}** words by cismen to **{rd.get("num_words__gender_notCM",0):,.0f}** by non-cismen (**{rd.get("num_words__gender_CM_to_notCM",0):.1f}x**)
* **{rd.get("num_words__race_W",0):,.0f}** words by white people to **{rd.get("num_words__race_notW",0):,.0f}** by people of color (**{rd.get("num_words__race_W_to_notW",0):.1f}x**)
"""
        printm(o)
            
        
        

    def show_bechdel_scores(self,max_comments=3):
        d = self.get_bechdel_scores()
        comments='\n\t* '+'\n\t* '.join('\n\t\t* '.join(y.strip() for y in x.split('\n') if y.strip() and not y.strip().startswith('Message posted on ')) for x in d.get("comments").split("----------")[:max_comments])
        o=[]
        o+=[f'* Rating: **{d.get("rating")}**']
        o+=[f'* Note: {d.get("msg")}']
        o+=[f'* Explanation: {d.get("explanation")}']
        if d.get('comments'): o+=[f'* Comments: {comments}']
        printm('\n'.join(o))

    def show_nonCM_dialogue(self):
        g=self.get_network()
        for a,b,d in g.edges(data=True):
            ag,bg=g.nodes[a]['actor_gender'],g.nodes[b]['actor_gender']
            if ag and bg and ag!='CM' and bg!='CM':
                printm(f'##### {a} ({ag}) --> {b} ({bg})')
                printm('* ' + d['speech'].replace('||','\n* '))
                print()

    def show_POC_dialogue(self):
        g=self.get_network()
        for a,b,d in g.edges(data=True):
            ag,bg=g.nodes[a]['actor_race'],g.nodes[b]['actor_race']
            if ag and bg and ag!='W' and bg!='W':
                printm(f'##### {a} ({ag}) --> {b} ({bg})')
                printm('* ' + d['speech'].replace('||','\n* '))
                print()


    def show_graph_info(self):
        
        printm(f'# {self.id}')

        printm(f'## Final social network')
        self.draw_network()

        printm(f'## Gender/race ratios')
        self.show_ratios()

        printm(f'## Bechdeltest.com score')
        self.show_bechdel_scores()

        printm(f'## Dialogue between non-cismen')
        self.show_nonCM_dialogue()

        printm(f'## Dialogue between people of  color')
        self.show_POC_dialogue()

        printm(f'## Cast')
        display(self.get_cast())
            
        
    def get_speeches(self):
        dial = self.get_dialogue()
        return list(dial.speech) if len(dial) else []

    def get_topics(self):
        if not self.corpus._topic_model_doc2topic: self.corpus.topic_model()
        if not self.corpus._topic_model_doc2topic: return {}
        topic2counts = Counter()
        for speech in self.get_speeches():
            topic = self.corpus._topic_model_doc2topic.get(speech)
            if topic is not None and topic>=0:
                topic2counts[topic]+=1
        return topic2counts

    def get_doc_topic_probs(self):
        if not self.corpus._topic_model_doc2topic: self.corpus.topic_model()
        if not self.corpus._topic_model_doc2topic: return {}
        
        dfdial = self.get_dialogue()
        if not len(dfdial): return pd.DataFrame()
        
        for speech in self.get_speeches():
            topic = self.corpus._topic_model_doc2topic.get(speech)
            if topic is not None and topic>=0:
                topic2counts[topic]+=1
        return topic2counts






































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
    URL_OF_URLS = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vS1GjWpm41GyN13hHQ_Gmm8BLxKxTcyueX2A60uW7jWe3rMievBLAd1goftP06uYGzmg6rSNZEKM-m5/pub?gid=0&single=true&output=csv'


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
            if os.path.exists(origfn) and not os.path.exists(ofn):
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

    def get_urls(self,force=False):
        if force or not self._urls:
            df=pd.read_csv(self.URL_OF_URLS)
            self._urls = dict(zip(df.id, df.url))
        return self._urls



    def metadata_actors(self,force=False):
        if force or self._dfactors is None:
            self._dfactors = pd.read_csv(self.URL_ACTOR_METADATA, dtype='str').fillna('')
            self._dfactors['actor_id'] = self._dfactors.actor_url.apply(lambda x: x.split('/')[-1])
            

        return self._dfactors
    
    def metadata_chars(self,force=False):
        if force or self._dfallchars is None:
            self._dfallchars = pd.read_csv(self.URL_CHAR_METADATA, dtype='str').fillna('')
            self._dfallchars['char_id'] = self._dfallchars.char_url.apply(lambda x: x.split('/')[-1])
        
        return self._dfallchars


    def get_speeches(self):
        return [dial for t in self.texts() for dial in t.dialogue().speech]

    @property
    def path_topic_model(self): return os.path.join(self.path_data, 'topic_model.pkl')

    def topic_model(self, docs = None, force=False, lim=None, embedding_model="all-MiniLM-L6-v2"):
        from bertopic import BERTopic
        if not force and self._topic_model: return self._topic_model
        if force or not os.path.exists(self.path_topic_model):
            if not docs: docs = self.get_speeches()
            docs = docs[:lim]
            topic_model = BERTopic(verbose=True, embedding_model=embedding_model, calculate_probabilities=True)
            topics, probs = topic_model.fit_transform(docs)
            topic_model.save(self.path_topic_model)
            with open(self.path_topic_model.replace('.pkl','.docs.pkl'),'wb') as of: pickle.dump(docs, of)
            with open(self.path_topic_model.replace('.pkl','.topics.pkl'),'wb') as of: pickle.dump(topics, of)
            with open(self.path_topic_model.replace('.pkl','.probs.pkl'),'wb') as of: pickle.dump(probs, of)
            
            self._topic_model = topic_model
            self._topic_model_docs = docs
            self._topic_model_topics = topics
            self._topic_model_probs = probs
        else:
            self._topic_model=BERTopic.load(self.path_topic_model)
            with open(self.path_topic_model.replace('.pkl','.docs.pkl'),'rb') as of: self._topic_model_docs=pickle.load(of)
            with open(self.path_topic_model.replace('.pkl','.topics.pkl'),'rb') as of: self._topic_model_topics=pickle.load(of)
            with open(self.path_topic_model.replace('.pkl','.probs.pkl'),'rb') as of: self._topic_model_probs=pickle.load(of)
        
        self._topic_model_doc2topic = {k:v for k,v in zip(self._topic_model_docs, self._topic_model_topics) if v>=0}
        self._topic_model_doc2probs = {k:dict(enumerate(v)) for k,v in zip(self._topic_model_docs, self._topic_model_probs)}
        return self._topic_model
        










def get_imdb_id(imdb_id):
    # get imdb id
    idx=str(imdb_id)
    if idx.startswith('tt'): idx=idx[2:]
    while len(idx)<7: idx='0'+idx    
    return idx


def get_all_speech_docs():
    C=BechdelTest()
    path_dials=f'{C.MOVIE_SCRIPT_DB_PATH}/parsed/dialogue'
    docs = []
    for fn in tqdm(os.listdir(path_dials)):
        with open(os.path.join(path_dials,fn)) as f:
            for ln in f:
                if '=>' in ln:
                    doc=ln.split('=>',1)[1].strip()
                    docs.append(doc)
    return docs















def fixname(x):
    while '  ' in x: x=x.replace('  ',' ')
    return x





def draw(id):
  clear_output(wait=True)
  t = C.textd[id]
  g = t.get_network()
  
  
  printm(f'# {id}')

  printm(f'## Final social network')
  draw_nx(g,show=True)

  printm(f'## Gender/race ratios')
  o=[]
  for k,v in t.get_ratios().items():
    o+=[f'* {k.replace("__", " (")+")"}: **{round(v,1):,}**']
  printm('\n'.join(o))

  printm(f'## Bechdeltest.com score')
  t.show_bechdel_scores()

  printm(f'## Dialogue between non-cismen')
  t.show_nonCM_dialogue()

  printm(f'## Dialogue between people of  color')
  t.show_POC_dialogue()

  printm(f'## Cast')
  display(t.get_cast())


def draw_dynamic(id):
    t = C.textd[id]
    fn,htm = t.draw_networks()
    return htm



