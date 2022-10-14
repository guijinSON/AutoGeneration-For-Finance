import spacy
import re
import country_converter as coco
import torch.nn as nn
from funcs.hallucination.emb import emb_sentence

# df = pd.read_csv('')


def single_epoch_validate(df, model,tokenizer, device):
    output=[]
    output_logits = []
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    NER = spacy.load('en_core_web_sm')
    
    for row in df.iterrows():
        row = row[1]
        src = row['seg_text_origin']
        tgt1 = row['seg_text']
        tgt2 = row['seg_text_clean']

        """
        semantic comparison
        """
        src_emb = emb_sentence(src, model, tokenizer, device)

        if (len(tgt1)//len(tgt2)) > 1.5:
            n = len(tgt1)//len(tgt2)
            seqlen = len(tgt1)//n
            sliced_tgt_emb = [cos(src_emb, emb_sentence(tgt1[i*seqlen:(i+1)*seqlen],model,tokenizer,device)).item() \
                            for i in range(len(tgt1)//seqlen)]
            an = sum(sliced_tgt_emb)/n
        
        else:
            tgt1_emb = emb_sentence(tgt1,model, tokenizer, device)
            an = cos(src_emb,tgt1_emb).item()
        
        tgt2_emb = emb_sentence(tgt2,model, tokenizer, device)
        
        bn = cos(src_emb,tgt2_emb).item()

        """
        number comparison 
        """

        src_num = re.findall("\d+", src)
        tgt1_num = re.findall("\d+",tgt1)
        tgt2_num = re.findall("\d+", tgt2)

        n_an = [n for n in tgt1_num if n not in src_num]
        n_bn = [n for n in tgt2_num if n not in src_num]

        """
        entity comparison #1
        """

        src_entity  = [(_.text,_.label_) for _ in NER(src).ents]
        tgt1_entity = [(_.text,_.label_) for _ in NER(tgt1).ents]
        tgt2_entity = [(_.text,_.label_) for _ in NER(tgt2).ents]
        
        """
        PERSON Entity
        """
        src_entity_p  = [_[0] for _ in src_entity if _[1] == 'PERSON']
        tgt1_entity_p = [_[0] for _ in tgt1_entity if _[1] == 'PERSON']
        tgt2_entity_p = [_[0] for _ in tgt2_entity if _[1] == 'PERSON']

        ep_an = [n for n in tgt1_entity_p if n not in src_entity_p]
        ep_bn = [n for n in tgt2_entity_p if n not in src_entity_p]

        """
        GPE Entity
        """

        src_entity_g  = coco.convert(names=[_[0] for _ in src_entity if _[1] == 'GPE'], to='name_short')
        src_entity_g = [src_entity_g] if isinstance(src_entity_g, str) else src_entity_g

        tgt1_entity_g = coco.convert(names=[_[0] for _ in tgt1_entity if _[1] == 'GPE'], to='name_short')
        tgt1_entity_g = [tgt1_entity_g] if isinstance(tgt1_entity_g, str) else tgt1_entity_g

        tgt2_entity_g = coco.convert(names=[_[0] for _ in tgt2_entity if _[1] == 'GPE'], to='name_short')
        tgt2_entity_g = [tgt2_entity_g] if isinstance(tgt2_entity_g, str) else tgt2_entity_g

        eg_an = [n for n in tgt1_entity_g if n not in src_entity_g]
        eg_bn = [n for n in tgt2_entity_g if n not in src_entity_g]


        e_an = len(n_an) + len(ep_an) + len(eg_an)
        e_bn = len(n_bn) + len(ep_bn) + len(eg_bn)

        if e_an > e_bn:
            output.append(1)
            output_logits.append(bn-an)
        else:
            if an < bn:
                output.append(1)
                output_logits.append(bn-an)

            else:
                output.append(0)
                
    return {
        "Accuracy": sum(output)/len(output),
        "Logits": sum(output_logits)/len(output_logits)
        }   


import torch.nn as nn
import spacy
import re
import country_converter as coco
from funcs.hallucination.emb import emb_sentence

def score_text(src, tgt, model, tokenizer, NER):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    #NER = spacy.load('en_core_web_sm')

    """
    semantic comparison
    """

    src_emb = emb_sentence(src,model,tokenizer,'cpu')

    if len(tgt) > 100:
        n = len(tgt)//100
        sliced_tgt_emb = [cos(src_emb, emb_sentence(tgt[i*100:(i+1)*100],model,tokenizer,'cpu')).item() for i in range(n)]
        src_tgt_cos_score = sum(sliced_tgt_emb)/n
    else:
        tgt_emb = emb_sentence(tgt,model, tokenizer, 'cpu')
        src_tgt_cos_score = cos(src_emb,tgt_emb).item()

    """
    number comparison 
    """

    src_num = re.findall("\d+", src)
    tgt_num = re.findall("\d+",tgt)

    src_tgt_entity_num = [n for n in tgt_num if n not in src_num]

    """
    entity comparison #1
    """

    src_entity  = [(_.text,_.label_) for _ in NER(src).ents]
    tgt_entity  = [(_.text,_.label_) for _ in NER(tgt).ents]

        # """
        # PERSON Entity
        # """
    src_entity_p  = [_[0] for _ in src_entity if _[1] == 'PERSON']
    tgt_entity_p  = [_[0] for _ in tgt_entity if _[1] == 'PERSON']

    src_tgt_entity_person = [n for n in tgt_entity_p if n not in src_entity_p]

        # """
        # GPE Entity
        # """

    src_entity_g  = coco.convert(names=[_[0] for _ in src_entity if _[1] == 'GPE'], to='name_short')
    src_entity_g = [src_entity_g] if isinstance(src_entity_g, str) else src_entity_g

    tgt_entity_g = coco.convert(names=[_[0] for _ in tgt_entity if _[1] == 'GPE'], to='name_short')
    src_tgt_entity_gpe = [tgt_entity_g] if isinstance(tgt_entity_g, str) else tgt_entity_g
    src_tgt_entity_gpe =[_ for _ in src_tgt_entity_gpe if _ != 'not found' or _ not in src_entity_gpe]
    
    return {
        "src_tgt_cos_score":src_tgt_cos_score,
        "src_tgt_entity_num":src_tgt_entity_num,
        "src_tgt_entity_person":src_tgt_entity_person,
        "src_tgt_entity_gpe":src_tgt_entity_gpe
    }
