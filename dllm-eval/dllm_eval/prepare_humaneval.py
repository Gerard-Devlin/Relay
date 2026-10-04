"""Export unchanged HumanEval rows from the existing datasets cache."""
import argparse,json
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    from datasets import load_dataset
    rows=[dict(row) for row in load_dataset('openai/openai_humaneval',split='test')]
    if len(rows)!=164:raise ValueError('Expected 164 HumanEval rows')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
