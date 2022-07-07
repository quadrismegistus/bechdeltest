from bechdeltest import *
from .mica import *

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
			for xx in [':','O.S.','V.O.',"'S VOICE","'S COM VOICE"]: speaker=speaker.replace(xx,'')
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

def parse_script_file(filename):
    with open(filename) as f: txt=f.read()
    return parse_script(txt)


