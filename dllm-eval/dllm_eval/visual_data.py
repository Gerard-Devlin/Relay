"""Pinned visual datasets, question-only model inputs and portable image paths."""
import hashlib,io,json,random,re
from pathlib import Path
VISUAL_VERSIONS={'vision_dominant','vision_intensive','vision_only'}

def official_query(task,doc):
    from .vendor.visual_prompts import mathvista_query,mathverse_query
    if task=='mathvista':
        fields={k:doc[k] for k in ('question','question_type','answer_type','precision','choices')}
        fields.update(unit=doc.get('unit',''),caption='',ocr='')
        return mathvista_query(None,fields,'reason-first',examples=None,shot_num=0,use_caption=False,use_ocr=False)
    if task=='mathverse':
        fields={k:doc[k] for k in ('question','question_type','query_wo')}
        return mathverse_query(None,fields,'format-prompt',None,'query_wo',examples=None,shot_num=0)
    raise ValueError('Unknown visual task')

def version(value):return re.sub(r'[ -]+','_',value.strip().lower())

def download_snapshots(cache):
    """Download only the checkpoints and dataset splits in the fixed profile."""
    from huggingface_hub import snapshot_download
    from relay_cache.guards import ROOT
    config=json.loads((ROOT/'dllm-eval/configs/lladav.json').read_text())
    settings=config['settings']
    for model,revision in ((settings['model'],settings['revision']),(settings['vision_model'],settings['vision_revision'])):
        snapshot_download(model,revision=revision,cache_dir=cache)
    for task,pattern in (('mathvista','**/testmini-*.parquet'),('mathverse','**/testmini.parquet')):
        spec=config['datasets'][task]
        snapshot_download(spec['repo'],revision=spec['revision'],repo_type='dataset',cache_dir=cache,
            allow_patterns=[pattern,'README.md'])

def prepare_dataset(cache,destination):
    import pyarrow.parquet as pq
    from PIL import Image
    cache=Path(cache);dest=Path(destination);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete').exists():return json.loads((dest/'data.json').read_text())
    specs={'mathvista':('datasets--AI4Math--MathVista','2b6ad69445fbb5695c9b165475e8decdbeb97747','**/testmini-*.parquet'),
        'mathverse':('datasets--CaraJ--MathVerse-lmmseval','8479d5e87785a3274da612e169d942c3d972b7fe','**/testmini.parquet')}
    result={};sources={}
    for task,(repo,revision,pattern) in specs.items():
        paths=sorted((cache/repo/'snapshots'/revision).glob(pattern));assert paths,task
        records=[]
        for p in paths:
            sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
            for batch in pq.ParquetFile(p).iter_batches(batch_size=32):
                for doc in batch.to_pylist():
                    if task=='mathverse' and version(doc['problem_version']) not in VISUAL_VERSIONS:continue
                    ident=str(doc['pid'] if task=='mathvista' else doc['sample_index'])
                    data=doc['decoded_image'] if task=='mathvista' else doc['image']
                    if not isinstance(data,dict) or not data.get('bytes'):raise ValueError('Missing visual input: '+ident)
                    image=Image.open(io.BytesIO(data['bytes'])).convert('RGB');image.load()
                    image_path=dest/'images'/task/(ident+'.image');image_path.parent.mkdir(parents=True,exist_ok=True)
                    image_path.write_bytes(data['bytes'])
                    question=official_query(task,doc)
                    # Strong regression: modifying the gold cannot change a model prompt.
                    changed=dict(doc,answer='__not_an_input__',solution='__not_an_input__')
                    assert official_query(task,changed)==question
                    records.append(dict(id=ident,task=task,prompt=question,image=str(image_path.relative_to(dest)).replace("\\","/"),
                        image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),question_type=doc['question_type'],question=doc['question'],evaluation_query=doc.get('query',question),
                        answer=str(doc['answer']),choices=doc.get('choices') or [],precision=doc.get('precision'),
                        answer_type=doc.get('answer_type'),problem_version=doc.get('problem_version'),problem_index=doc.get('problem_index')))
        assert len({x['id'] for x in records})==len(records)
        assert len(records)==(1000 if task=='mathvista' else 2364),(task,len(records))
        random.Random(51713).shuffle(records);result[task]=records
    (dest/'data.json').write_text(json.dumps(result,ensure_ascii=False))
    (dest/'manifest.json').write_text(json.dumps(dict(seed=51713,counts={k:len(v) for k,v in result.items()},sources=sources,
        subsets={'mathvista':'testmini','mathverse':'official mathverse_testmini_vision, three visual versions'},
        zero_shot=True,model_inputs=['official question/choices/answer-type hints','image'],gold_inputs=False),indent=2))
    (dest/'complete').write_text('0\n');return result

def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hf-hub-cache',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--download',action='store_true',help='Download the pinned checkpoints and dataset splits first')
    args=p.parse_args()
    if args.download:download_snapshots(args.hf_hub_cache)
    data=prepare_dataset(args.hf_hub_cache,args.output)
    print(json.dumps({task:len(rows) for task,rows in data.items()}))

if __name__=='__main__':main()
