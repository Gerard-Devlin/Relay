"""Identical output-boundary handling; preserve raw tokens for later regrading."""
def for_scoring(result, tokenizer, stop, length):
    ids = result['token_ids']
    ending = tokenizer.encode(stop, add_special_tokens=False)
    cut, reason = len(ids), 'returned_short_output' if len(ids) < length else 'length_budget'
    if ending:
        for i in range(len(ids) - len(ending) + 1):
            if ids[i:i+len(ending)] == ending:
                cut, reason = i, 'conversation_stop'
                break
    if 126081 in ids and ids.index(126081) < cut:
        cut, reason = ids.index(126081), 'eos'
    return tokenizer.decode(ids[:cut], skip_special_tokens=True), dict(
        reason=reason, budget_exhausted=reason == 'length_budget',
        raw_tokens=len(ids), scored_tokens=cut, stop=stop,
        policy='first explicit conversation-stop token sequence or EOS; raw tokens retained')
