from pathlib import Path
from rapidfuzz.distance import Levenshtein
from errant.en.lancaster import LancasterStemmer
import spacy
import spacy.symbols as POS

# Load Hunspell word list
def load_word_list(path):
    with open(path) as word_list:
        return set([word.strip() for word in word_list])

# Load Universal Dependency POS Tags map file.
# https://universaldependencies.org/tagset-conversion/en-penn-uposf.html
def load_pos_map(path):
    map_dict = {}
    with open(path) as map_file:
        for line in map_file:
            line = line.strip().split("\t")
            # Change ADP to PREP for readability
            if line[1] == "ADP": map_dict[line[0]] = "PREP"
            # Change PROPN to NOUN; we don't need a prop noun tag
            elif line[1] == "PROPN": map_dict[line[0]] = "NOUN"
            # Change CCONJ to CONJ
            elif line[1] == "CCONJ": map_dict[line[0]] = "CONJ"
            # Otherwise
            else: map_dict[line[0]] = line[1].strip()
        # Add some spacy PTB tags not in the original mapping.
        map_dict['""'] = "PUNCT"
        map_dict["SP"] = "SPACE"
        map_dict["_SP"] = "SPACE"
        map_dict["BES"] = "VERB"
        map_dict["HVS"] = "VERB"
        map_dict["ADD"] = "X"
        map_dict["GW"] = "X"
        map_dict["NFP"] = "X"
        map_dict["XX"] = "X"
    return map_dict

# Classifier resources
base_dir = Path(__file__).resolve().parent
# Spacy
nlp = None
# Lancaster Stemmer
stemmer = LancasterStemmer()
# GB English word list (inc -ise and -ize)
# spell = load_word_list(base_dir/"resources"/"en_GB-large.txt")

# Part of speech map file
pos_map = load_pos_map(base_dir/"resources"/"en-ptb_map_fact")
# Open class coarse Spacy POS tags 
open_pos1 = {POS.ADJ, POS.ADV, POS.NOUN, POS.VERB}
# Open class coarse Spacy POS tags (strings)
open_pos2 = {"ADJ", "ADV", "NOUN", "VERB"}
# Rare POS tags that make uninformative error categories
rare_pos = {"INTJ", "X"}
# Contractions
conts = {"'d", "'ll", "'m", "n't", "'re", "'s", "'ve"}
# Special auxiliaries in contractions.
aux_conts = {"ca": "can", "sha": "shall", "wo": "will"}
# Some dep labels that map to pos tags.
dep_map = {
    "acomp": "ADJ",
    "amod": "ADJ",
    "advmod": "ADV",
    "det": "DET", 
    "prep": "PREP",
    "prt": "PART",
    "punct": "PUNCT"}

# Map the POS tags to factual error categories
pos_facte_map = {
    'ADJ':'Ent:AttrE',
    'NOUN':'Ent:ObjE',
    'PROPN':'Ent:ObjE',
    'VERB':'Pred:VerbE',
    'ADV':'CircE',
    'PREP':'CircE',
    'PART':'CircE',
    'DET':'TrivE',
    'PRON':'CorefE',
    'CONJ':'LinkE',
    'NUM':'NumE',
    'SYM':'NumE'
}
# Trivial POS for factual errors
trivials = {"PUNCT", "SPACE"}
# Modal Verb
prob_modals = {"ca", "can", "could", "may", "might", "should", "would", "shall", "sha", "need", "must", "dare", "ought"}
# Negation
negs = {"not", "n't"}
# Adjective possessive pronuon
adj_pos = {"my", "your", "our", "his", "her", "its", "their"}
# Special person number
person_num = {"person1", "person2", "person3", "1", "2", "3"}
# Common conjunction
conjs = {"before", "after", "because", "though", "although", "whereas", "but", "besides", "therefore", "unless", "since", "if", "yet", "and", "or", "nor", "so", "as", "then"}

# Input: An Edit object
# Output: The same Edit object with an updated error type
def classify(edit):
    # Nothing to nothing is a detected but not corrected edit
    if not edit.o_toks and not edit.c_toks:
        edit.type = "UNK"
    # Missing
    elif not edit.o_toks and edit.c_toks:
        op = "M:"
        cat = get_one_sided_type(edit.c_toks)
        edit.type = op+cat
    # Unnecessary
    elif edit.o_toks and not edit.c_toks:
        op = "U:"
        cat = get_one_sided_type(edit.o_toks)
        edit.type = op+cat
    # Replacement and special cases
    else:
        # Same to same is a detected but not corrected edit
        if edit.o_str == edit.c_str:
            edit.type = "UNK"
        # Special: Ignore case change at the end of multi token edits
        # E.g. [Doctor -> The doctor], [, since -> . Since]
        # Classify the edit as if the last token wasn't there
        elif edit.o_toks[-1].lower == edit.c_toks[-1].lower and \
                (len(edit.o_toks) > 1 or len(edit.c_toks) > 1):
            # Store a copy of the full orig and cor toks
            all_o_toks = edit.o_toks[:]
            all_c_toks = edit.c_toks[:]
            # Truncate the instance toks for classification
            edit.o_toks = edit.o_toks[:-1]
            edit.c_toks = edit.c_toks[:-1]
            # Classify the truncated edit
            edit = classify(edit)
            # Restore the full orig and cor toks
            edit.o_toks = all_o_toks
            edit.c_toks = all_c_toks
        # Replacement
        else:
            op = "R:"
            cat = get_two_sided_type(edit.o_toks, edit.c_toks)
            edit.type = op+cat
    return edit

# Input: Spacy tokens
# Output: A list of pos and dep tag strings
def get_edit_info(toks):
    pos = []
    dep = []
    for tok in toks:
        pos.append(pos_map[tok.tag_])
        dep.append(tok.dep_)
    return pos, dep

# Input: Spacy tokens
# Output: An error type string based on input tokens from orig or cor
# When one side of the edit is null, we can only use the other side
def get_one_sided_type(toks):
    # Special cases
    if len(toks) == 1:
        # Possessive noun suffixes; e.g. ' -> 's
        if toks[0].tag_ == "POS":
            return "Ent:AttrE"
        # Contractions. Rule must come after possessive
        # if toks[0].lower_ in conts:
        #     return "CONTR"
        
        # Infinitival "to" is treated as part of a verb form
        # if toks[0].lower_ == "to" and toks[0].pos == POS.PART and \
        #         toks[0].dep_ != "prep":
        #     return "VERB:FORM"
        if toks[0].lower_.strip() in negs:
            return "Pred:NegE"
        if toks[0].lower_.strip() == "'s":
            return "Ent:AttrE"
        if toks[0].lower_.strip() in adj_pos:
            return "CorefE"
        if toks[0].lower_.strip() in conjs:
            return "LinkE"

        # For DialogSum, # Person 1 #, # Person 2 #
        if toks[0].lower_.strip() == "#":
            return "TrivE" 
        
    # Extract pos tags and parse info from the toks
    pos_list, dep_list = get_edit_info(toks)
    # Auxiliary verbs
    if set(dep_list).issubset({"aux", "auxpass"}):
        toks_text = [x.lower_ for x in toks]
        if "will" in toks_text or "'ll" in toks_text:
            return "Pred:TensE"
        elif set(toks_text).issubset(set(prob_modals)):
            return "Pred:ModE"
        else:
            return "Pred:VerbE"
    # do n't
    if dep_list == ["aux", "neg"]:
        return "Pred:NegE"
            
    # POS-based tags. Ignores rare, uninformative categories
    if len(set(pos_list)) == 1 and pos_list[0] not in rare_pos:
        if pos_list[0] in trivials:
            return "TrivE"
        return pos_facte_map[pos_list[0]] # my_add
    # More POS-based tags using special dependency labels
    if len(set(dep_list)) == 1 and dep_list[0] in dep_map.keys():
        if dep_map[dep_list[0]] in trivials:
            return "TrivE"
        return pos_facte_map[dep_map[dep_list[0]]] # my_add
    # To-infinitives and phrasal verbs
    # if pos_list == ["PART", "VERB"]:
    #      return "CircE"
    
    # such as 4 : 30
    if set(pos_list) == {"Num", "SYM"}:
        return "NumE"


    if person_num_change(toks):
        return "Ent:ObjE"
    # Tricky cases

    if "VERB" in pos_list:
        return "Pred:VerbE"
    if "NOUN" in pos_list or "PROPN" in pos_list or "PRON" in pos_list:
        return "Ent:ObjE"
    if "ADJ" in pos_list:
        return "Ent:AttrE"
    if "ADV" in pos_list or "PREP" in pos_list:
        return "CircE"

    return "OthE"

# Input 1: Spacy orig tokens
# Input 2: Spacy cor tokens
# Output: An error type string based on orig AND cor
def get_two_sided_type(o_toks, c_toks):
    # Extract pos tags and parse info from the toks as lists
    o_pos, o_dep = get_edit_info(o_toks)
    c_pos, c_dep = get_edit_info(c_toks)

    # Orthography; i.e. whitespace and/or case errors.
    if only_orth_change(o_toks, c_toks):
        return "TrivE"

    # For DialogSum, # Person 1 # -> person1
    if only_well_change(o_toks, c_toks):
        return "TrivE"

    if person_num_change(o_toks, c_toks):
        return "Ent:ObjE"

    # Word Order; only matches exact reordering.
    # if exact_reordering(o_toks, c_toks):
    #     return "WO"

    # 1:1 replacements (very common)
    if len(o_toks) == len(c_toks) == 1:
        # 1. SPECIAL CASES
        # Possessive noun suffixes; e.g. ' -> 's
        if o_toks[0].tag_ == "POS" or c_toks[0].tag_ == "POS":
            return "Ent:AttrE"
        
        # Contraction. Rule must come after possessive.
        # if (o_toks[0].lower_ in conts or \
        #        c_toks[0].lower_ in conts) and \
        #         o_pos == c_pos:
        #     return "CONTR"
        
        # Special auxiliaries in contractions (1); e.g. ca -> can, wo -> will
        # Rule was broken in V1. Turned off this fix for compatibility.
        # if (o_toks[0].lower_ in aux_conts and \
        #         c_toks[0].lower_ == aux_conts[o_toks[0].lower_]) or \
        #         (c_toks[0].lower_ in aux_conts and \
        #        o_toks[0].lower_ == aux_conts[c_toks[0].lower_]):
        #     return "CONTR"
        
        # Special auxiliaries in contractions (2); e.g. ca -> could, wo -> should
        if o_toks[0].lower_ in aux_conts or \
                c_toks[0].lower_ in aux_conts:
            return "Pred:ModE"
        # Special: "was" and "were" are the only past tense SVA
        if {o_toks[0].lower_, c_toks[0].lower_} == {"was", "were"}:
            return "Pred:VerbE"
        
        # conjunctons, but -> because
        if {o_toks[0].lower_, c_toks[0].lower_}.issubset(conjs):
            return "LinkE"
        

        # 2. SPELLING AND INFLECTION
        # Only check alphabetical strings on the original side
        # Spelling errors take precedence over POS errors; this rule is ordered
        '''
        if o_toks[0].text.isalpha():
            # Check a GB English dict for both orig and lower case.
            # E.g. "cat" is in the dict, but "Cat" is not.
            if o_toks[0].text not in spell and \
                    o_toks[0].lower_ not in spell:
                # Check if both sides have a common lemma
                if o_toks[0].lemma == c_toks[0].lemma:
                    # Inflection; often count vs mass nouns or e.g. got vs getted
                    if o_pos == c_pos and o_pos[0] in {"NOUN", "VERB"}:
                        return o_pos[0]+":INFL"
                    # Unknown morphology; i.e. we cannot be more specific.
                    else:
                        return "MORPH"
                # Use string similarity to detect true spelling errors.
                else:
                    # Normalised Lev distance works better than Lev ratio
                    str_sim = Levenshtein.normalized_similarity(o_toks[0].lower_, c_toks[0].lower_)
                    # WARNING: THIS IS AN APPROXIMATION.
                    # Thresholds tuned manually on FCE_train + W&I_train
                    # str_sim > 0.55 is almost always a true spelling error
                    if str_sim > 0.55:
                        return "SPELL"
                    # Special scores for shorter sequences are usually SPELL
                    if str_sim == 0.5 or round(str_sim, 3) == 0.333:
                        # Short strings are more likely to be spell: eles -> else
                        if len(o_toks[0].text) <= 4 and len(c_toks[0].text) <= 4:
                            return "SPELL"
                    # The remainder are usually word choice: amounght -> number
                    # Classifying based on cor_pos alone is generally enough.
                    if c_pos[0] not in rare_pos:
                        return c_pos[0]
                    # Anything that remains is OTHER
                    else:
                        return "OTHER"
        '''
        # 3. MORPHOLOGY
        # Only ADJ, ADV, NOUN and VERB can have inflectional changes.
        if o_toks[0].lemma == c_toks[0].lemma and \
                o_pos[0] in open_pos2 and \
                c_pos[0] in open_pos2:
            # Same POS on both sides
            if o_pos == c_pos:
                # Adjective form; e.g. comparatives
                if o_pos[0] == "ADJ":
                    return "Ent:AttrE"
                # Noun number
                if o_pos[0] == "NOUN":
                    return "Ent:ObjE"
                # Verbs - various types
                if o_pos[0] == "VERB":
                    # NOTE: These rules are carefully ordered.
                    # Use the dep parse to find some form errors.
                    # Main verbs preceded by aux cannot be tense or SVA.


                    if preceded_by_aux(o_toks, c_toks):
                        return "Pred:VerbE"
                    # Use fine PTB tags to find various errors.
                    # FORM errors normally involve VBG or VBN.
                    if o_toks[0].tag_ in {"VBG", "VBN"} or \
                            c_toks[0].tag_ in {"VBG", "VBN"}:
                        return "Pred:VerbE"
                    # Of what's left, TENSE errors normally involved VBD.
                    if o_toks[0].tag_ == "VBD" or c_toks[0].tag_ == "VBD":
                        return "Pred:TensE"
                    # Of what's left, SVA errors normally involve VBZ.
                    if o_toks[0].tag_ == "VBZ" or c_toks[0].tag_ == "VBZ":
                        return "Pred:VerbE"
                    # Any remaining aux verbs are called TENSE.
                    if o_dep[0].startswith("aux") and \
                            c_dep[0].startswith("aux"):
                        o_toks_text = [x.lower_ for x in o_toks]
                        c_toks_text = [x.lower_ for x in c_toks]


                        if "will" in o_toks_text or "'ll" in o_toks_text:
                            return "Pred:TensE"
                        elif set(o_toks_text).issubset(set(prob_modals)) or \
                        set(c_toks_text).issubset(set(prob_modals)):
                            return "Pred:ModE"
                        else:
                            return "Pred:VerbE"
            # Use dep labels to find some more ADJ:FORM
            if set(o_dep+c_dep).issubset({"acomp", "amod"}):
                return "Ent:AttrE"
            # Adj to plural noun is usually noun number; e.g. musical -> musicals.
            if o_pos[0] == "ADJ" and c_toks[0].tag_ == "NNS":
                return "Ent:ObjE"
            # For remaining verb errors (rare), rely on c_pos
            if c_toks[0].tag_ in {"VBG", "VBN"}:
                return "Pred:VerbE"
            if c_toks[0].tag_ == "VBD":
                return "Pred:TensE"
            if c_toks[0].tag_ == "VBZ":
                return "Pred:VerbE"
            # Tricky cases that all have the same lemma.
            # else:
            #     return "OthE"
        # Derivational morphology.
        # if stemmer.stem(o_toks[0].text) == stemmer.stem(c_toks[0].text) and \
        #         o_pos[0] in open_pos2 and c_pos[0] in open_pos2:
        #     return "MORPH"

        # 4. GENERAL
        # Auxiliaries with different lemmas
        if o_dep[0].startswith("aux") and c_dep[0].startswith("aux"):
            return "Pred:ModE"
        
        # his -> her adj_pos for CorefE
        if o_toks[0].lower_ in adj_pos or c_toks[0].lower_ in adj_pos:
            return "CorefE"

        
        # POS-based tags. Some of these are context sensitive mispellings.
        if o_pos == c_pos and o_pos[0] not in rare_pos:
            if o_pos[0] in trivials:
                return "TrivE"
            return pos_facte_map[o_pos[0]]
        # Some dep labels map to POS-based tags.
        if o_dep == c_dep and o_dep[0] in dep_map.keys():
            if dep_map[o_dep[0]] in trivials:
                return "TrivE"
            return pos_facte_map[dep_map[o_dep[0]]]
        # Phrasal verb particles.
        if set(o_pos+c_pos) == {"PART", "PREP"} or \
                set(o_dep+c_dep) == {"prt", "prep"}:
            return "CircE"
        # Can use dep labels to resolve DET + PRON combinations.
        if set(o_pos+c_pos) == {"DET", "PRON"}:
            # DET cannot be a subject or object.
            # if c_dep[0] in {"nsubj", "nsubjpass", "dobj", "pobj"}:
            #     return "PRON"
            # "poss" indicates possessive determiner
            # if c_dep[0] == "poss":
            #     return "DET"
            return "CorefE"
        # NUM and DET are usually DET; e.g. a <-> one
        # if set(o_pos+c_pos) == {"NUM", "DET"}:
        #     return "TrivE"
        # Special: other <-> another
        # if {o_toks[0].lower_, c_toks[0].lower_} == {"other", "another"}:
        #     return "DET"
        # Special: your (sincerely) -> yours (sincerely)
        # if o_toks[0].lower_ == "your" and c_toks[0].lower_ == "yours":
        #     return "PRON"
        # Special: no <-> not; this is very context sensitive
        # if {o_toks[0].lower_, c_toks[0].lower_} == {"no", "not"}:
        #     return "OTHER"
            
        # 5. STRING SIMILARITY
        # These rules are quite language specific.
        '''
        if o_toks[0].text.isalpha() and c_toks[0].text.isalpha():
            # Normalised Lev distance works better than Lev ratio
            str_sim = Levenshtein.normalized_similarity(o_toks[0].lower_, c_toks[0].lower_)
            # WARNING: THIS IS AN APPROXIMATION.
            # Thresholds tuned manually on FCE_train + W&I_train
            # A. Short sequences are likely to be SPELL or function word errors
            if len(o_toks[0].text) == 1:
                # i -> in, a -> at
                if len(c_toks[0].text) == 2 and str_sim == 0.5:
                    return "SPELL"
            if len(o_toks[0].text) == 2:
                # in -> is, he -> the, to -> too
                if 2 <= len(c_toks[0].text) <= 3 and str_sim >= 0.5:
                    return "SPELL"
            if len(o_toks[0].text) == 3:
                # Special: the -> that (relative pronoun)
                if o_toks[0].lower_ == "the" and c_toks[0].lower_ == "that":
                    return "PRON"
                # Special: all -> everything
                if o_toks[0].lower_ == "all" and c_toks[0].lower_ == "everything":
                    return "PRON"
                # off -> of, too -> to, out -> our, now -> know
                if 2 <= len(c_toks[0].text) <= 4 and str_sim >= 0.5:
                    return "SPELL"
            # B. Longer sequences are also likely to include content word errors
            if len(o_toks[0].text) == 4:
                # Special: that <-> what
                if {o_toks[0].lower_, c_toks[0].lower_} == {"that", "what"}:
                    return "PRON"
                # Special: well <-> good
                if {o_toks[0].lower_, c_toks[0].lower_} == {"good", "well"} and \
                        c_pos[0] not in rare_pos:
                    return c_pos[0]
                # knew -> new, 
                if len(c_toks[0].text) == 3 and str_sim > 0.5:
                    return "SPELL"
                # then <-> than, form -> from
                if len(c_toks[0].text) == 4 and str_sim >= 0.5:
                    return "SPELL"
                # gong -> going, hole -> whole
                if len(c_toks[0].text) == 5 and str_sim == 0.8:
                    return "SPELL"
                # high -> height, west -> western
                if len(c_toks[0].text) > 5 and str_sim > 0.5 and \
                        c_pos[0] not in rare_pos:
                    return c_pos[0]
            if len(o_toks[0].text) == 5:
                # Special: after -> later
                if {o_toks[0].lower_, c_toks[0].lower_} == {"after", "later"} and \
                        c_pos[0] not in rare_pos:
                    return c_pos[0]
                # where -> were, found -> fund
                if len(c_toks[0].text) == 4 and str_sim == 0.8:
                    return "SPELL"
                # thing <-> think, quite -> quiet, their <-> there
                if len(c_toks[0].text) == 5 and str_sim >= 0.6:
                    return "SPELL"
                # house -> domestic, human -> people
                if len(c_toks[0].text) > 5 and c_pos[0] not in rare_pos:
                    return c_pos[0]
            # C. Longest sequences include MORPH errors
            if len(o_toks[0].text) > 5 and len(c_toks[0].text) > 5:
                # Special: therefor -> therefore
                if o_toks[0].lower_ == "therefor" and c_toks[0].lower_ == "therefore":
                    return "SPELL"
                # Special: though <-> thought
                if {o_toks[0].lower_, c_toks[0].lower_} == {"though", "thought"}:
                    return "SPELL"
                # Morphology errors: stress -> stressed, health -> healthy
                if (o_toks[0].text.startswith(c_toks[0].text) or \
                        c_toks[0].text.startswith(o_toks[0].text)) and \
                        str_sim >= 0.66:
                    return "MORPH"
                # Spelling errors: exiting -> exciting, wether -> whether
                if str_sim > 0.8:
                    return "SPELL"
                # Content word errors: learning -> studying, transport -> travel
                if str_sim < 0.55 and c_pos[0] not in rare_pos:
                    return c_pos[0]
                # NOTE: Errors between 0.55 and 0.8 are a mix of SPELL, MORPH and POS
        # Tricky cases
        else:
            return "OTHER"
        '''

    # Multi-token replacements (uncommon)
    # All auxiliaries
    # Exclude "will" from ModE
    
    if set(o_dep+c_dep).issubset({"aux", "auxpass"}):
        return "Pred:ModE"
    # All same POS
    if len(set(o_pos+c_pos)) == 1:
        # Final verbs with the same lemma are tense; e.g. eat -> has eaten
        
        o_toks_text = [x.lower_ for x in o_toks]
        c_toks_text = [x.lower_ for x in c_toks]
        if o_pos[0] == "VERB" and \
                o_toks[-1].lemma == c_toks[-1].lemma:

            if set(o_toks_text).symmetric_difference(set(c_toks_text)).intersection(prob_modals):
                return "Pred:ModE"
            
            if o_toks[-1].tag_ == "VBD" or c_toks[-1].tag_ == "VBD":
                return "Pred:TensE"

            
        # POS-based tags.
        elif o_pos[0] not in rare_pos:
            if o_pos[0] in trivials:
                return "TrivE"
            return pos_facte_map[o_pos[0]]




    # All same special dep labels.
    if len(set(o_dep+c_dep)) == 1 and \
            o_dep[0] in dep_map.keys():
        return pos_facte_map[dep_map[o_dep[0]]]
    # Infinitives, gerunds, phrasal verbs.
    # if set(o_pos+c_pos) == {"PART", "VERB"}:
        # Final verbs with the same lemma are form; e.g. to eat -> eating
        # if o_toks[-1].lemma == c_toks[-1].lemma:
        #     return "VERB:FORM"
        # Remaining edits are often verb; e.g. to eat -> consuming, look at -> see
        # else:
        #     return "VERB"
    #     return "CircE"
    # Possessive nouns; e.g. friends -> friend 's
    if (o_pos == ["NOUN", "PART"] or c_pos == ["NOUN", "PART"]) and \
            o_toks[0].lemma == c_toks[0].lemma:
        return "Ent:AttrE"
    # Adjective forms with "most" and "more"; e.g. more free -> freer
    if (o_toks[0].lower_ in {"most", "more"} or \
            c_toks[0].lower_ in {"most", "more"}) and \
            o_toks[-1].lemma == c_toks[-1].lemma and \
            len(o_toks) <= 2 and len(c_toks) <= 2:
        return "Ent:AttrE"
    
    # The same POS of the origin
    # is -> xxxxxx
    if len(set(o_pos)) == 1 and o_pos[0] not in rare_pos:
        if o_pos[0] in trivials:
            return "TrivE"
        return pos_facte_map[o_pos[0]] # my_add

    # 4 : 30 -> 5 : 30
    if set(o_pos) == {"Num", "SYM"} or set(c_pos) == {"Num", "SYM"}:
        return "NumE"

    o_toks_text = [x.lower_ for x in o_toks]
    c_toks_text = [x.lower_ for x in c_toks]
    # will -> wo n't, shall -> sha n't
    if ( o_toks_text == ["wo", "n't"] and c_toks_text == ["will"] ) or \
    ( c_toks_text == ["wo", "n't"] and o_toks_text == ["will"] ):
        return "Pred:NegE"
    if ( o_toks_text == ["sha", "n't"] and c_toks_text == ["shall"] ) or \
    ( c_toks_text == ["sha", "n't"] and o_toks_text == ["shall"] ):
        return "Pred:NegE"
    
    if o_toks[-1].lemma == c_toks[-1].lemma:
        # wants -> does not want
        if ( o_pos == ["AUX", "PART", "VERB"] and c_pos == ["VERB"] ) or \
        ( c_pos == ["AUX", "PART", "VERB"] and o_pos == ["VERB"] ):
            return "Pred:NegE"




    # Tricky cases.

    if "VERB" in o_pos or "VERB" in c_pos:
        return "Pred:VerbE"
    if "NOUN" in o_pos or "PROPN" in o_pos or "PRON" in o_pos \
    or "NOUN" in c_pos or "PROPN" in c_pos or "PRON" in c_pos:
        return "Ent:ObjE"
    if "ADV" in o_pos or "PREP" in o_pos \
    or "ADV" in c_pos or "PREP" in c_pos:
        return "CircE"

    return "OthE"



# Input 1: Spacy orig tokens
# Input 2: Spacy cor tokens
# Output: Boolean; the difference between orig and cor is only whitespace or case
def only_orth_change(o_toks, c_toks):
    o_join = "".join([o.lower_ for o in o_toks])
    c_join = "".join([c.lower_ for c in c_toks])
    if o_join == c_join:
        return True
    return False

# Input 1: Spacy orig tokens
# Input 2: Spacy cor tokens
# Output: Boolean; the difference between orig and cor is only whitespace or case or #
def only_well_change(o_toks, c_toks):
    o_join = "".join([o.lower_ for o in o_toks])
    o_join = o_join.replace("#","")
    c_join = "".join([c.lower_ for c in c_toks])
    c_join = c_join.replace("#","")
    if o_join == c_join:
        return True
    return False

def person_num_change(o_toks, c_toks=None):
    o_join = "".join([o.lower_ for o in o_toks])
    o_join = o_join.replace("#","")
    if not c_toks:
        c_join = ""
    else:
        c_join = "".join([c.lower_ for c in c_toks])
        c_join = c_join.replace("#","")

    if o_join != c_join and set([o_join, c_join]).issubset(person_num):
        return True

    if o_join[:-1] == c_join[:-1] and o_join[:-1] == "person" and \
    o_join[-1].isdigit() and c_join[-1].isdigit():
        return True
    
    

    return False


# Input 1: Spacy orig tokens
# Input 2: Spacy cor tokens
# Output: Boolean; the tokens are exactly the same but in a different order
def exact_reordering(o_toks, c_toks):
    # Sorting lets us keep duplicates.
    o_set = sorted([o.lower_ for o in o_toks])
    c_set = sorted([c.lower_ for c in c_toks])
    if o_set == c_set:
        return True
    return False

# Input 1: An original text spacy token. 
# Input 2: A corrected text spacy token.
# Output: Boolean; both tokens have a dependant auxiliary verb.
def preceded_by_aux(o_tok, c_tok):
    # If the toks are aux, we need to check if they are the first aux.
    if o_tok[0].dep_.startswith("aux") and c_tok[0].dep_.startswith("aux"):
        # Find the parent verb
        o_head = o_tok[0].head
        c_head = c_tok[0].head
        # Find the children of the parent
        o_children = o_head.children
        c_children = c_head.children
        # Check the orig children.
        for o_child in o_children:
            # Look at the first aux...
            if o_child.dep_.startswith("aux"):
                # Check if the string matches o_tok
                if o_child.text != o_tok[0].text:
                    # If it doesn't, o_tok is not first so check cor
                    for c_child in c_children:
                        # Find the first aux in cor...
                        if c_child.dep_.startswith("aux"):
                            # If that doesn't match either, neither are first aux
                            if c_child.text != c_tok[0].text:
                                return True
                            # Break after the first cor aux
                            break
                # Break after the first orig aux.
                break
    # Otherwise, the toks are main verbs so we need to look for any aux.
    else:
        o_deps = [o_dep.dep_ for o_dep in o_tok[0].children]
        c_deps = [c_dep.dep_ for c_dep in c_tok[0].children]
        if "aux" in o_deps or "auxpass" in o_deps:
            if "aux" in c_deps or "auxpass" in c_deps:
                return True
    return False
