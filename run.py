from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
import pandas as pd
import re
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import time
import asyncio
from copy import deepcopy
from pathlib import Path
from dotenv import load_dotenv
import os

pd.set_option('display.max_columns', 60)

load_dotenv(override=True)

print(os.getenv('OPENAI_API_KEY'))

exit()

async def run(lcase):
    def extract_last_figure(text):
        # Find all numbers within brackets, optionally preceded by a dollar sign, and possibly containing commas
        numbers = re.findall(r'\[(?:\$)?((?:\d+,)*\d+)\]', text)
        if numbers:
            # If at least one number was found, convert the last one to integer
            # Remove commas before converting
            return int(numbers[-1].replace(',', ''))
        else:
            # If no numbers were found, return None
            return None

    damage_df = pd.read_csv("data/damage_df.csv")

    instructions = open('data/damages_guide.txt', 'r').read()

    base_text_smartgpt_predict = '''
    You are a superforecaster who prides themselves on their ability to produce precice estimates off of messy qualitative information. 
    Below is a description of a case with all dollar amounts redacted:
    {data}






    Some amount of damages were awarded in this case. I want you to guess the value of damages that was ultimately awarded in this case.

    In your answer, please work this out in a step by step way to be sure we have the best estimate. Include your final numeric answer as an integer in brackets. I want the format of your reply to be:
    <step by step reasoning>
    [<numeric final answer>]

    I want to read your final answer directly as an integer, so make sure there is an integer in brackets at the end; if there is not, I will be very upset. 
    Once again, you must put your integer estimate in brackets at the end of your output!

    '''

    prompt_pred_smart = PromptTemplate(
        input_variables=["data"],
        template=base_text_smartgpt_predict,
    )

    chainps = LLMChain(llm=ChatOpenAI(model_name="gpt-4",request_timeout=500), prompt=prompt_pred_smart)







    base_text_guide_predict = '''
    You are a superforecaster who prides themselves on their ability to produce precice estimates off of messy qualitative information. 

    Below is a guide on how to estimate damages from personal injury cases including what factors should matter for how much damages will be awarded.
    {guide}

    Below is a description of a case with all dollar amounts redacted:
    {data}


    Some amount of damages were awarded in this case. I want you to guess the value of damages that was ultimately awarded in this case making use of the guide.

    In your answer, please work this out in a step by step way to be sure we have the best estimate. Include your final numeric answer as an integer in brackets. I want the format of your reply to be:
    <step by step reasoning>
    [<numeric final answer>]

    I want to read your final answer directly as an integer, so make sure there is an integer in brackets at the end; if there is not, I will be very upset. 
    Once again, you must put your integer estimate in brackets at the end of your output!

    '''

    prompt_pred_guide = PromptTemplate(
        input_variables=["guide","data"],
        template=base_text_guide_predict,
    )

    chainpg = LLMChain(llm=ChatOpenAI(model_name="gpt-4",request_timeout=500), prompt=prompt_pred_guide)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    #np.bool = np.bool_
    # Load pre-trained model and tokenizer
    model_name = "intfloat/e5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)

    # Your data

    # Function to convert text to embeddings
    def text_to_embedding(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs[0][:,0,:].cpu().numpy()  # we take the embedding of the [CLS] token
        return embeddings





    from copy import deepcopy

    async def async_guide_predict(chain,udict):
        print('using guide')
        try:
            v = udict['uid']
            pdict=deepcopy(udict)
            del pdict['uid']
            if 'text' in pdict: #just change less code
                pdict['data']=pdict['text']
                del pdict['text']
            res = await chain.arun(pdict)
            rdf = pd.DataFrame({'uid':[v],'prediction':[res]})
        except:
            rdf=pd.DataFrame({})
        return rdf



    async def guide_predict_concurrently(udicts,chain):
        #tasks = [async_generate(chain3,"test") for _ in range(5)]
        tasks = []
        for udict in udicts:
            tasks.append(async_guide_predict(chain,udict))
        
        results = await asyncio.gather(*tasks)
        return results



    async def async_smart_predict(chain,text,v):
        print('smart gpt')
        try:
            res = await chain.arun(text)
            rdf = pd.DataFrame({'uid':[v],'prediction':[res]})
        except:
            rdf=pd.DataFrame({})
        return rdf



    async def gsmart_predict_concurrently(udicts,chain):
        #tasks = [async_generate(chain3,"test") for _ in range(5)]
        tasks = []
        for udict in udicts:
            text = udict['text']
            v= udict['uid']
            tasks.append(async_smart_predict(chain,text,v))
        
        results = await asyncio.gather(*tasks)
        return results


    async def predict_and_extract(uids, udf, max_attempts=10,use_guide=False):
        attempts = 0
        print('attempts')
        print(attempts)
        dataframe = pd.DataFrame({})
        while uids and attempts < max_attempts:
            if use_guide:
                udicts = [{'text': udf['redacted_text'][udf['uid']==uid].min(), 'guide':instructions, 'uid':uid} for uid in uids]
            else:
                udicts = [{'text': udf['redacted_text'][udf['uid']==uid].min(), 'uid':uid} for uid in uids]
            if use_guide:
                pred_df = await guide_predict_concurrently(udicts,chainpg)
            else:
                pred_df = await gsmart_predict_concurrently(udicts,chainps)
            pred_df = pd.concat(pred_df)
            pred_df['numericp'] = pred_df['prediction'].apply(lambda x: extract_last_figure(x))
            apred_df = pred_df[pred_df.numericp.notnull()]
            dataframe = pd.concat([dataframe, apred_df])
            uids = list(pred_df.uid[pred_df.numericp.isnull()])
            print('uid length')
            print(len(uids))
            
            print('dataframe length')
            print(len(dataframe))
            attempts += 1
        return dataframe

    udf = pd.DataFrame({'uid':[0,1,2,3,4],'redacted_text':[lcase,lcase,lcase,lcase,lcase]})

    base_results = await predict_and_extract([0,1,2,3,4],udf)
    guide_results = await predict_and_extract([0,1,2,3,4],udf,True)


    #---------scoring; need to do it 3x each
    nv1 = np.mean(base_results.numericp)


    pdict = {'data':lcase,'guide':instructions}
    ng1  = np.mean(guide_results.numericp)





    #-------similarities



    base_pred_sim = '''
    You are an superforecaster who prides themselves in their ability to produce precise quantiative estimates off of qualitiative data making use of the full scale available. 

    Here is the summary of a legal case in NY (case A):
    {case_a}

    Now here is a second summary of a legal case in NY, (case b):
    {case_b}


    I want you to rate how similar these cases are.

    I want you to rate similarity on a scale from 0 to 100 where 0 is completely different having no similarities whatsoever aside from being a legal case in NY 100 is virtually identical. 
    The more precise your answer, the better; even small differences in values can contain important information (DO NOT ROUND YOUR ANSWER).
    I want you to return a single integer as your answer. I want to read your answer directly into python.

    '''

    prompt_pred_sim = PromptTemplate(
        input_variables=["case_a","case_b"],
        template=base_pred_sim,
    )

    chainsim3 = LLMChain(llm=ChatOpenAI(model_name="gpt-3.5-turbo",request_timeout=200), prompt=prompt_pred_sim)
    chainsim4 = LLMChain(llm=ChatOpenAI(model_name="gpt-4",request_timeout=200), prompt=prompt_pred_sim)


    uids = list(damage_df['uid'])

    text_data = list(damage_df['redacted_text'])



    async def async_sim_predict(chain,udict):
        try:
            udict2 = deepcopy(udict)
            v = udict['uid']
            del udict2['uid']
            res = await chain.arun(udict2)
            rdf = pd.DataFrame({'uid':[v],'prediction':[res]})
        except:
            print('fail')
            rdf=pd.DataFrame({})
        return rdf



    async def gsmart_predict_concurrently(udicts,chain):
        #tasks = [async_generate(chain3,"test") for _ in range(5)]
        tasks = []
        for udict in udicts:
            tasks.append(async_sim_predict(chain,udict))
        
        results = await asyncio.gather(*tasks)
        return results

    fresults = pd.DataFrame({})

    lcount = 20
    sublists = [uids[i:i + lcount] for i in range(0, len(uids), lcount)]

    tcase = 'On October 30, 2016, Michael Davis and his father-in-law R. Foster Hinds wereinstalling a tree stand for deer huntingon property in upstate Canisteo owned by Mr. Hinds and his wife. After the stand was installed, Mr. Davis stepped onto its platform to test it and adefective ratchet strap broke causing him to fall and sustain serious back injuries.In Mr. Davis’s ensuing non-jury lawsuit, the judge in Steuben County determined that Mr. and Mrs. Hinds were negligent and he awarded plaintiffpain and suffering damages in the sum of <redacted>(<redacted> past – five years, <redacted> future -22 years).Defendants appealed but, inDavis v. Hinds(4th Dept. 2023), the judgment was affirmed (except that the claims against Mrs. Hinds were dismissed).Here are the injury details:L-1, L-3 and L-4 fracturesSpinal fusion surgery T10-S1Continuing pain, limitations as to standing and lifting and, unable to resume recreational activitiesInside Information:There wasno expert medical testimony adduced at trial; instead, the parties agreed to use medical records only.In hisclosing argument, plaintiff’s attorney asked the judge to award <redacted> for pain and suffering damages.Plaintiff, 47 years old at the time of his accident,returned to work within three monthsas a high school physical education teacher and did not assert any claims for lost earnings (or medical expenses).'

    for sl in sublists[0:5]:
        print(sl)
        pdicts = []
        for si in sl:
            ucase = damage_df['redacted_text'][damage_df['uid']==si].min()
            pdict = {'case_a':lcase,'case_b':ucase,'uid':si}
            pdicts.append(pdict)
        #res = chainsim.run(pdict)
        results = await gsmart_predict_concurrently(pdicts,chainsim4)
        results = pd.concat(results)
        fresults = pd.concat([fresults,results])
        time.sleep(30)

    fresults['prediction'] = pd.to_numeric(fresults['prediction'], errors='coerce')

    # Drop rows containing NaN
    fresults.dropna(subset=['prediction'], inplace=True)

    # Convert to int
    fresults['prediction'] = fresults['prediction'].astype(int)




    fmerge = pd.merge(fresults,damage_df[['uid','damages','otext']],on='uid')
    fmerge.sort_values('prediction',inplace=True,ascending=False)
    top_20 = fmerge.head(20)

    sim_mean = fmerge.damages.mean()



    log_numeric = np.log(nv1)
    log_numericg = np.log(ng1)
    log_gpt_sim_mean = np.log(sim_mean)


    est = -11.8067+1.0343*log_gpt_sim_mean+0.3946*log_numeric+0.3733*log_numericg
    final_estimate  = np.exp(est)


    top_20['distance']=(top_20['damages']-final_estimate).map(abs)
    top_20.sort_values('distance',inplace=True)

    use_texts = list(top_20['otext'])[0:3]

    damages = list(top_20['damages'])[0:3]




    base_text_find_citation = '''
    Give me an example legal citation for this case; return only the citation and nothing else:
    {data}

    '''

    prompt_pred_cite = PromptTemplate(
        input_variables=["data"],
        template=base_text_find_citation,
    )

    chainp_cite = LLMChain(llm=ChatOpenAI(model_name="gpt-4",request_timeout=500), prompt=prompt_pred_cite)

    citation0 = chainp_cite.run(use_texts[0])
    citation1 = chainp_cite.run(use_texts[1])
    citation2 = chainp_cite.run(use_texts[2])


    cite_string = """
    {0}: (damages {1})\n
    {2}: (damages {3})\n
    {4}: (damages {5})\n
    """.format(citation0,damages[0],citation1,damages[1],citation2,damages[2])



    fstring = """The predicted damages are {0}. Below are similar cases with similar damages: {1}""".format(int(final_estimate),cite_string)
    return fstring

if __name__ == '__main__':
    lcase = open(Path('./txt.txt'), 'r').read()
    asyncio.run(run(lcase))