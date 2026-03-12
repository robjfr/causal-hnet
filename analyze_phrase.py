#!/usr/bin/env python3
"""
Analyze a phrase: parse it and show contextual substitution classes for each node.
Usage:
    python analyze_phrase.py "we should go"
    python analyze_phrase.py "that's a beautiful hat"
"""
import sys
import warnings
warnings.filterwarnings('ignore')
# Import ContextPattern for pickle compatibility
import prototype_topdown_units
from prototype_topdown_units import ContextPattern
from bidir_simple import UnitCatalog, IncrementalBidirParser, print_tree, CorpusIndex
from collections import defaultdict
import math

def find_subparse_combinations(left_text: str, right_text: str,
                               catalog: UnitCatalog, corpus_index: CorpusIndex,
                               subparse_cache: dict,
                               max_results: int = 20, max_subs_per_word: int = 15) -> list:
    """
    Find valid multi-word combinations by trying substitutions of subparse elements.
    Uses incremental filtering to avoid combinatorial explosion.

    For example, for "they made" + "the car":
    - Get subparses: ["they", "made"] + ["the", "car"]
    - Try combinations like: "you did a job", "we saw the house", etc.
    - Filter at each step: only keep sequences where adjacent words can follow each other
    - Stop early when we find max_results valid sequences

    Returns: list of (combined_text, score) tuples
    """
    # Get subparses (splits) for each multi-word phrase
    def get_subparse_words(text):
        """Get the words from the best subparse of a multi-word phrase."""
        words = text.split()
        if len(words) == 1:
            return words
        # For multi-word, return the words (we could recursively get subparses, but start simple)
        return words

    left_words = get_subparse_words(left_text)
    right_words = get_subparse_words(right_text)
    all_words = left_words + right_words

    if len(all_words) <= 2:
        # Too short for meaningful subparse combinations
        return []

    # Get substitutes for each word position
    # LAZY APPROACH: Get candidates without scoring (fast!)
    # Only score complete sequences that pass all filters
    position_subs = []
    for i, word in enumerate(all_words):
        word_subs = [word]  # Always include original (no score yet)

        # Get words from contexts (fast - just dictionary lookup, no GPU scoring)
        word_pattern = catalog.get_unit(word)
        if word_pattern:
            # Get words that appear in similar contexts
            # Combine left and right contexts
            context_words = set(word_pattern.left_words.keys()) | set(word_pattern.right_words.keys())

            # Sort by frequency (simple Counter, no GPU)
            context_freq = {}
            for ctx_word in context_words:
                ctx_pattern = catalog.get_unit(ctx_word)
                if ctx_pattern:
                    context_freq[ctx_word] = ctx_pattern.count

            # Take top N by frequency
            sorted_by_freq = sorted(context_freq.items(), key=lambda x: -x[1])
            top_context_words = [w for w, _ in sorted_by_freq[:max_subs_per_word]]

            # Add to word_subs (just words, no scores)
            for sub_word in top_context_words:
                if sub_word not in word_subs:
                    word_subs.append(sub_word)

        position_subs.append(word_subs[:max_subs_per_word])

    # Incremental search with filtering (NO SCORING until complete!)
    valid_sequences = []
    partial_sequences = [[]]  # Just word lists, no scores yet

    for pos in range(len(all_words)):
        new_partials = []

        for partial_words in partial_sequences:
            for sub_word in position_subs[pos]:
                extended_words = partial_words + [sub_word]

                # Filter: check if this extension is valid (adjacency constraint)
                # Use CorpusIndex to avoid min_freq=5 limitation
                if pos > 0:
                    prev_word = partial_words[-1]
                    # Check if the pair exists in corpus (no frequency threshold)
                    pair_text = f"{prev_word} {sub_word}"
                    pair_pattern = get_unit_with_fallback(pair_text, catalog, corpus_index, debug=False)
                    if not pair_pattern:
                        # This pair never appears in corpus → skip
                        continue

                # If complete sequence, check if it exists in corpus
                if len(extended_words) == len(all_words):
                    combined_text = " ".join(extended_words)

                    # Only include if different from original
                    original_text = " ".join(all_words)
                    if combined_text == original_text:
                        continue

                    # Check if this combination exists in catalog/corpus
                    pattern = get_unit_with_fallback(combined_text, catalog, corpus_index, debug=False)
                    if pattern:
                        # NOW score it (only for complete sequences that exist!)
                        # Use corpus frequency as score
                        score = pattern.count
                        valid_sequences.append((combined_text, score))
                        if len(valid_sequences) >= max_results:
                            return valid_sequences
                else:
                    # Continue building this sequence
                    new_partials.append(extended_words)

        partial_sequences = new_partials

        # Early abandonment if no partials remain
        if not partial_sequences:
            break

    return valid_sequences


def get_unit_with_fallback(text: str, catalog: UnitCatalog, corpus_index: CorpusIndex = None, debug: bool = False):
    """
    Get unit from catalog, with fallback to corpus index if not found.
    This allows lookup of longer phrases (5-7 grams) not pre-indexed in catalog.
    """
    # Try catalog first
    pattern = catalog.get_unit(text)
    if pattern is not None:
        return pattern

    # Fallback to corpus index if available
    if corpus_index is not None:
        pattern = corpus_index.get_unit(text)
        if pattern is not None:
            if debug and text.count(' ') >= 3:
                print(f"        [CORPUS] Found in corpus: \"{text}\" (freq={pattern.count})")
            return pattern

    return None


def get_all_substitutes_merged(cache: dict, text: str) -> list:
    """
    Get ALL substitutes for a text from ALL cached parses, merged with source tags.

    Returns:
        List of (sub_text, score, source_type, origins) tuples
        source_type: 'aggregation' or 'expansion'
    """
    parses = [(k, v) for k, v in cache.items() if k[0] == text]
    if not parses:
        return []

    merged_subs = {}  # sub_text -> (best_score, source_type, origins)

    for (_, split), cache_value in parses:
        subs = cache_value[0]
        origins = cache_value[4] if len(cache_value) == 5 else None

        # Determine source type from split
        if isinstance(split, tuple) and len(split) == 2 and split[0] == 'expansion':
            source_type = 'expansion'
            parent_split = split[1]  # The parent context split
        elif isinstance(split, tuple) and len(split) == 2:
            # Regular binary split = aggregation from mutual expansion
            source_type = 'aggregation'
            parent_split = split  # The (left, right) split
        elif split is None:
            # Single word, no split
            source_type = 'other'
            parent_split = None
        else:
            source_type = 'other'
            parent_split = split

        # Merge substitutes, keeping best score for each
        for sub_text, score in subs:
            if sub_text not in merged_subs or score > merged_subs[sub_text][0]:
                sub_origins = origins.get(sub_text) if origins else None
                # For expansion, store parent split info; for aggregation, store the split itself
                if not sub_origins and (source_type == 'expansion' or source_type == 'aggregation'):
                    sub_origins = {'split_type': source_type, 'parent_split': parent_split}
                merged_subs[sub_text] = (score, source_type, sub_origins)

    # Convert to list and sort by score
    result = [(text, score, source, origins)
              for text, (score, source, origins) in merged_subs.items()]
    return sorted(result, key=lambda x: -x[1])


def get_substitutes_for_split(cache: dict, text: str, split) -> list:
    """
    Get substitutes for a specific (text, split) cache entry with metadata.

    Returns:
        List of (sub_text, score, source_type, origins) tuples for THIS specific split only
    """
    cache_key = (text, split)
    if cache_key not in cache:
        return []

    cache_value = cache[cache_key]
    subs = cache_value[0]  # List of (text, score) tuples
    origins_dict = cache_value[4] if len(cache_value) >= 5 else {}

    # Determine source type from split format
    if isinstance(split, tuple) and len(split) == 2 and split[0] == 'expansion':
        source_type = 'expansion'
    elif isinstance(split, tuple):
        source_type = 'aggregation'
    else:
        source_type = 'other'

    # Convert to list with metadata
    result = []
    for sub_text, score in subs:
        sub_origins = origins_dict.get(sub_text) if origins_dict else None
        result.append((sub_text, score, source_type, sub_origins))

    return result


def get_best_parse(cache: dict, text: str) -> tuple:
    """
    Get the best parse for a text from cache (lowest energy).

    Returns:
        (substitutes, left_contexts, right_contexts, energy, split, origins) or ([], [], [], inf, None, None) if not found
    """
    # Find all parses for this text
    parses = [(k, v) for k, v in cache.items() if k[0] == text]
    if not parses:
        return ([], [], [], float('inf'), None, None)

    # Return parse with best (lowest) energy
    best_key, best_value = min(parses, key=lambda x: x[1][3])  # x[1][3] is energy

    # Handle variable-length cache entries (some have origins, some don't)
    if len(best_value) == 5:
        # Has origins: (subs, eff_left, eff_right, energy, origins)
        return best_value[:4] + (best_key[1], best_value[4])
    else:
        # No origins: (subs, eff_left, eff_right, energy)
        return best_value + (best_key[1], None)


def count_bridging_paths(left_sub: str, right_sub: str,
                         left_text: str, right_text: str,
                         catalog, corpus_index=None) -> tuple:
    """
    Count context→context→context paths between substitutes.

    Returns:
        (direct_bridges, left_validated, right_validated)
        - direct_bridges: words W where left_sub → W → right_sub
        - left_validated: words W where left_sub → W and W → right_text
        - right_validated: words W where left_text → W and W → right_sub
    """
    # Use corpus_index fallback to handle rare words/phrases
    left_sub_pattern = get_unit_with_fallback(left_sub, catalog, corpus_index)
    right_sub_pattern = get_unit_with_fallback(right_sub, catalog, corpus_index)
    left_text_pattern = get_unit_with_fallback(left_text, catalog, corpus_index)
    right_text_pattern = get_unit_with_fallback(right_text, catalog, corpus_index)

    direct_bridges = 0
    left_validated = 0
    right_validated = 0

    if left_sub_pattern and right_sub_pattern:
        # Direct bridging: words that can follow left_sub AND precede right_sub
        left_sub_right = set(left_sub_pattern.right_words.keys())
        right_sub_left = set(right_sub_pattern.left_words.keys())
        direct_bridges = len(left_sub_right & right_sub_left)

    if left_sub_pattern and right_text_pattern:
        # Left validation: words that can follow left_sub AND precede right_text
        left_sub_right = set(left_sub_pattern.right_words.keys())
        right_text_left = set(right_text_pattern.left_words.keys())
        left_validated = len(left_sub_right & right_text_left)

    if left_text_pattern and right_sub_pattern:
        # Right validation: words that can follow left_text AND precede right_sub
        left_text_right = set(left_text_pattern.right_words.keys())
        right_sub_left = set(right_sub_pattern.left_words.keys())
        right_validated = len(left_text_right & right_sub_left)

    return (direct_bridges, left_validated, right_validated)


def compute_outer_context_score(combined_text: str,
                                  outer_left_context: list,
                                  outer_right_context: list,
                                  catalog, corpus_index=None) -> float:
    """
    Score how well a combined substitute matches the outer context of the head phrase.

    This is the consensus score against external context that was previously
    only used for expansion substitutes.

    Returns:
        Score between 0 and 1 indicating context match quality
    """
    # Use corpus_index fallback to handle rare phrases
    combined_pattern = get_unit_with_fallback(combined_text, catalog, corpus_index)
    if not combined_pattern:
        return 0.0

    # Check overlap with outer left context (words that can precede this phrase)
    left_score = 0.0
    if outer_left_context:
        combined_left_ctx = set(combined_pattern.left_words.keys())
        outer_left_set = set(outer_left_context)
        if combined_left_ctx and outer_left_set:
            intersection = len(combined_left_ctx & outer_left_set)
            union = len(combined_left_ctx | outer_left_set)
            left_score = intersection / union if union > 0 else 0.0

    # Check overlap with outer right context (words that can follow this phrase)
    right_score = 0.0
    if outer_right_context:
        combined_right_ctx = set(combined_pattern.right_words.keys())
        outer_right_set = set(outer_right_context)
        if combined_right_ctx and outer_right_set:
            intersection = len(combined_right_ctx & outer_right_set)
            union = len(combined_right_ctx | outer_right_set)
            right_score = intersection / union if union > 0 else 0.0

    # Return average of left and right scores
    if left_score > 0 or right_score > 0:
        return (left_score + right_score) / 2.0
    return 0.0


def run_mutual_expansion(left_text: str, right_text: str,
                         left_eff_left: list, left_eff_right: list,
                         right_eff_left: list, right_eff_right: list,
                         catalog: UnitCatalog,
                         max_class_members: int = 10,
                         scoring: str = 'cosine',
                         verbose: bool = False,
                         corpus_index: 'CorpusIndex' = None) -> tuple:
    """
    Run mutual expansion between left and right elements.

    Returns:
        (left_subs, right_subs, left_candidates, right_candidates)
        - left_subs: Initial substitutes for left (from step 1)
        - right_subs: Initial substitutes for right (from step 2)
        - left_candidates: Final LEFT candidates (from step 4)
        - right_candidates: Final RIGHT candidates (from step 3)
    """
    # Skip if either side has no context fillers
    if not left_eff_right and not right_eff_left:
        return ([], [], [], [])

    # Step 1: Get substitutes for LEFT
    if verbose:
        print(f"      [STEP 1] Get substitutes for LEFT: \"{left_text}\"")
        print(f"        Candidates: {len(right_eff_left)} words from RIGHT's left_words")

    left_scoring = 'containment'
    right_scoring = 'containment'

    all_left_subs = catalog.gpu_contextual_candidates(
        target=left_text,
        candidates=right_eff_left,
        max_results=max_class_members * 3,
        scoring=left_scoring,
        trace=False
    )
    all_left_subs = [(t, s) for t, s in all_left_subs if t != left_text]
    left_subs = all_left_subs[:max_class_members]

    # Collect right_words from left substitutes → candidates for RIGHT
    context_word_counts = {}
    n_subs_with_pattern = 0
    for sub_text, _ in left_subs:
        sub_pattern = get_unit_with_fallback(sub_text, catalog, corpus_index)
        if sub_pattern:
            n_subs_with_pattern += 1
            for word in sub_pattern.right_words.keys():
                context_word_counts[word] = context_word_counts.get(word, 0) + 1

    min_share = max(2, int(n_subs_with_pattern * 0.3)) if n_subs_with_pattern > 0 else 2
    right_contexts_of_left_element = {w for w, c in context_word_counts.items() if c >= min_share}

    # Step 2: Get substitutes for RIGHT
    if verbose:
        print(f"      [STEP 2] Get substitutes for RIGHT: \"{right_text}\"")
        print(f"        Candidates: {len(left_eff_right)} words from LEFT's right_words")

    all_right_subs = catalog.gpu_contextual_candidates(
        target=right_text,
        candidates=left_eff_right,
        max_results=max_class_members * 3,
        scoring=right_scoring,
        trace=False
    )
    all_right_subs = [(t, s) for t, s in all_right_subs if t != right_text]
    right_subs = all_right_subs[:max_class_members]

    # Collect left_words from right substitutes → candidates for LEFT
    context_word_counts_r = {}
    n_rsubs_with_pattern = 0
    for sub_text, _ in right_subs:
        sub_pattern = get_unit_with_fallback(sub_text, catalog, corpus_index)
        if sub_pattern:
            n_rsubs_with_pattern += 1
            for word in sub_pattern.left_words.keys():
                context_word_counts_r[word] = context_word_counts_r.get(word, 0) + 1

    min_share_r = max(2, int(n_rsubs_with_pattern * 0.3)) if n_rsubs_with_pattern > 0 else 2
    left_contexts_of_right_element = {w for w, c in context_word_counts_r.items() if c >= min_share_r}

    # Step 3: Score RIGHT candidates
    right_candidates = catalog.gpu_contextual_candidates(
        target=right_text,
        candidates=list(right_contexts_of_left_element),
        max_results=max_class_members + 1,
        scoring=right_scoring,
        trace=False
    )
    right_candidates = [(t, s) for t, s in right_candidates if t != right_text][:max_class_members]

    # Step 4: Score LEFT candidates
    left_candidates = catalog.gpu_contextual_candidates(
        target=left_text,
        candidates=list(left_contexts_of_right_element),
        max_results=max_class_members + 1,
        scoring=left_scoring,
        trace=False
    )
    left_candidates = [(t, s) for t, s in left_candidates if t != left_text][:max_class_members]

    return (left_subs, right_subs, left_candidates, right_candidates)


def aggregate_contexts_from_constituents(text: str, subparse_cache: dict, catalog: UnitCatalog, verbose: bool = False, corpus_index: 'CorpusIndex' = None) -> tuple:
    """
    For multi-word units, aggregate contexts from COMBINATIONS of substitutes across binary splits.

    For "i want to", we test all binary parses:
    - ("i" "want to"): aggregate from combinations of i-subs × want-to-subs
    - ("i want" "to"): aggregate from combinations of i-want-subs × to-subs

    For each (left_sub, right_sub) pair:
    - Collect right_words from left_sub (what can follow left_sub)
    - Collect left_words from right_sub (what can precede right_sub)

    This captures actual alternative phrasings, not just constituent contexts.
    """
    tokens = text.split()
    if len(tokens) == 1:
        # Single word - use its own pattern
        pattern = get_unit_with_fallback(text, catalog, corpus_index)
        eff_left = list(pattern.left_words.keys()) if pattern else []
        eff_right = list(pattern.right_words.keys()) if pattern else []
        return (eff_left, eff_right)

    # Multi-word: aggregate from combinations across all binary parses
    all_left = set()
    all_right = set()

    # Include the unit's own context if it exists
    pattern = get_unit_with_fallback(text, catalog, corpus_index)
    own_left_count = 0
    own_right_count = 0
    if pattern:
        own_left = list(pattern.left_words.keys())
        own_right = list(pattern.right_words.keys())
        all_left.update(own_left)
        all_right.update(own_right)
        own_left_count = len(own_left)
        own_right_count = len(own_right)

    # Test all binary splits of this multi-word unit
    combination_count = 0
    for split in range(1, len(tokens)):
        left_text = " ".join(tokens[:split])
        right_text = " ".join(tokens[split:])

        # Get cached substitution classes for both sides (best parse)
        left_subs, left_eff_left, left_eff_right, _, _, _ = get_best_parse(subparse_cache, left_text)
        right_subs, right_eff_left, right_eff_right, _, _, _ = get_best_parse(subparse_cache, right_text)

        # Bootstrap constituents if not cached
        if not left_subs:
            if len(left_text.split()) == 1:
                # Single word: use identity substitute
                left_pattern = get_unit_with_fallback(left_text, catalog, corpus_index)
                if left_pattern:
                    left_subs = [(left_text, 1.0)]
                    left_eff_left = list(left_pattern.left_words.keys())
                    left_eff_right = list(left_pattern.right_words.keys())
            else:
                # Multi-word: recursively aggregate contexts and run mutual expansion
                left_eff_left, left_eff_right = aggregate_contexts_from_constituents(
                    left_text, subparse_cache, catalog, verbose=False, corpus_index=corpus_index
                )
                # Generate substitutes through mutual expansion of its constituents
                # For now, use identity as placeholder - will be generated in next iteration
                left_subs = [(left_text, 1.0)]

        if not right_subs:
            if len(right_text.split()) == 1:
                # Single word: use identity substitute
                right_pattern = get_unit_with_fallback(right_text, catalog, corpus_index)
                if right_pattern:
                    right_subs = [(right_text, 1.0)]
                    right_eff_left = list(right_pattern.left_words.keys())
                    right_eff_right = list(right_pattern.right_words.keys())
            else:
                # Multi-word: recursively aggregate contexts and run mutual expansion
                right_eff_left, right_eff_right = aggregate_contexts_from_constituents(
                    right_text, subparse_cache, catalog, verbose=False, corpus_index=corpus_index
                )
                # Generate substitutes through mutual expansion of its constituents
                # For now, use identity as placeholder - will be generated in next iteration
                right_subs = [(right_text, 1.0)]

        # Now run mutual expansion between left and right to generate actual substitutes
        if left_eff_left or left_eff_right or right_eff_left or right_eff_right:
            if verbose:
                print(f"      [AGGREGATION] Running mutual expansion for \"{text}\" split: (\"{left_text}\", \"{right_text}\")")
            left_subs_exp, right_subs_exp, _, _ = run_mutual_expansion(
                left_text, right_text,
                left_eff_left, left_eff_right,
                right_eff_left, right_eff_right,
                catalog,
                max_class_members=10,
                scoring='cosine',
                verbose=verbose,
                corpus_index=corpus_index
            )
            # Use generated substitutes if we got any, otherwise keep identity
            if left_subs_exp:
                left_subs = left_subs_exp
            if right_subs_exp:
                right_subs = right_subs_exp

        # Skip if either side still has no substitutes
        if not left_subs or not right_subs:
            continue

        # Aggregate contexts from all combinations of left_subs × right_subs
        # Also collect these combinations as substitutes to cache
        all_combinations = []
        combination_origins = {}  # Track where each combination came from

        for left_sub_text, left_score in left_subs:
            for right_sub_text, right_score in right_subs:
                # Form the joined phrase from this combination
                combined_text = f"{left_sub_text} {right_sub_text}"

                # Look up this complete alternative phrasing in catalog/corpus
                combined_pattern = get_unit_with_fallback(combined_text, catalog, corpus_index)
                if not combined_pattern:
                    continue

                # Collect contexts from this alternative complete phrasing
                # left_words: what can precede this entire phrase
                for word in combined_pattern.left_words.keys():
                    all_left.add(word)

                # right_words: what can follow this entire phrase
                for word in combined_pattern.right_words.keys():
                    all_right.add(word)

                combination_count += 1

                # NEW: Also save this combination as a potential substitute
                combo_score = left_score * right_score
                all_combinations.append((combined_text, combo_score))

                # NEW: Track origin for debugging/tracing
                # Show full substitution path: what → substitute
                combination_origins[combined_text] = {
                    'left_source': left_text,      # e.g., "hat"
                    'left_sub': left_sub_text,     # e.g., "together"
                    'left_score': left_score,
                    'right_source': right_text,    # e.g., "again"
                    'right_sub': right_sub_text,   # e.g., "and"
                    'right_score': right_score
                }

        # NEW: Cache the combinations as substitutes for this text
        # This makes them available when this span is used in larger contexts
        if all_combinations and len(tokens) > 1:
            # Sort by score and cache
            all_combinations = sorted(all_combinations, key=lambda x: -x[1])[:30]
            # Use a special marker to indicate these came from aggregation
            agg_energy = -len(all_combinations) * 0.1  # Simple energy estimate

            # Store with origin information for tracing
            subparse_cache[(text, ('aggregation', tuple(tokens)))] = (
                all_combinations, list(all_left), list(all_right), agg_energy, combination_origins
            )
            if verbose:
                print(f"      Cached {len(all_combinations)} combinations as substitutes for \"{text}\"")

    if verbose and len(tokens) > 1:
        print(f"      Context aggregation for \"{text}\": "
              f"own=({own_left_count}L, {own_right_count}R), "
              f"{combination_count} combinations → "
              f"enriched=({len(all_left)}L, {len(all_right)}R)")

    return (list(all_left), list(all_right))

def compute_energy(num_substitutes: int, consensus_score: float, full_length_count: int = None) -> float:
    """
    Compute energy from substitution class properties.

    Lower energy = better defined meaning:
    - More substitutes → lower energy (well-supported pattern)
    - Higher consensus → lower energy (converged meaning)

    Args:
        num_substitutes: Number of substitutes found
        consensus_score: Average context overlap consensus (0-1)
        full_length_count: Number of substitutes matching the span's word count (0 if not applicable)

    Returns:
        Energy value (lower is better)
    """
    # Avoid log(0)
    num_subs = max(1, num_substitutes)
    consensus = max(0.01, consensus_score)
    full_length = max(1, full_length_count) if full_length_count is not None else 1

    # Energy: lower is better
    # - More total subs = lower energy (good)
    # - High consensus = higher energy (bad) — diverse contexts = good compositional unit
    # - More full-length subs = lower energy (good) — productive parse structure
    energy = -math.log(num_subs) + math.log(consensus) - math.log(full_length)

    return energy

def compute_consensus_score(substitutes: list, catalog: UnitCatalog, target_text: str, corpus_index: 'CorpusIndex' = None) -> float:
    """
    Compute consensus score: how much substitutes agree on their contexts.

    Returns:
        Consensus score 0-1 (1 = perfect agreement)
    """
    if not substitutes:
        return 0.01

    target_pattern = get_unit_with_fallback(target_text, catalog, corpus_index)
    if not target_pattern:
        return 0.01

    # Get target's context words
    target_left = set(target_pattern.left_words.keys())
    target_right = set(target_pattern.right_words.keys())
    all_target_context = target_left | target_right

    if not all_target_context:
        return 0.01

    # For each context word, count how many substitutes share it
    word_counts = {}
    for sub_text, _ in substitutes:
        sub_pattern = get_unit_with_fallback(sub_text, catalog, corpus_index)
        if sub_pattern:
            sub_contexts = set(sub_pattern.left_words.keys()) | set(sub_pattern.right_words.keys())
            for word in sub_contexts & all_target_context:
                word_counts[word] = word_counts.get(word, 0) + 1

    if not word_counts:
        return 0.01

    # Average frequency (normalized by num substitutes)
    avg_frequency = sum(word_counts.values()) / (len(word_counts) * len(substitutes))

    return min(1.0, avg_frequency)

def analyze(phrase: str, catalog: UnitCatalog, parser: IncrementalBidirParser,
            corpus_index: CorpusIndex = None,
            max_class_members: int = 500, scoring: str = 'cosine', external_left: list = [], external_right: list = []):
    """Analyze a phrase and display results."""
    tokens = phrase.lower().split()
    print("=" * 70)
    print(f'INPUT: "{phrase}"')
    print(f'SCORING: {scoring}')
    print("=" * 70)

    # Incremental processing
    print("\n[1] INCREMENTAL BOTTOM-UP ANALYSIS (SEQUENTIAL PRESENTATION)")
    print("-" * 40)
    parser.reset()  # Clear for new sequence
    subparse_cache = {}  # prefix_text -> (subs, eff_left, eff_right)
    html_buffer = []  # NEW: Buffer HTML during parsing for efficiency
    for k, word in enumerate(tokens, 1):
        parser.add_word(word)  # Add word sequentially
        print(f"\nAdding word '{word}' at position {k}")

        # Parse ALL subspans ending at position k (from shortest to longest)
        # CKY order: process shorter spans first so constituents are available for longer spans
        for start_pos in range(k - 1, -1, -1):  # k-1 down to 0 (shortest spans first)
            span_tokens = tokens[start_pos:k]
            span_text = " ".join(span_tokens)
            span_length = len(span_tokens)

            print(f"  Parsing span [{start_pos}:{k}]: \"{span_text}\"")

            # Bootstrap single-word spans
            if span_length == 1:
                # For single word, no candidates available yet - just cache the unit itself
                subs = []
                # Use aggregation (handles both single and multi-word gracefully)
                eff_left, eff_right = aggregate_contexts_from_constituents(
                    span_text, subparse_cache, catalog
                )
                # No split for single word, energy is high (no substitutes)
                energy = float('inf')
                subparse_cache[(span_text, None)] = (subs, eff_left, eff_right, energy)

                # Add to GPU index for dynamic scoring (bootstrap: uniform weights)
                from collections import Counter
                catalog.add_unit_to_gpu_index(span_text, Counter({w: 1 for w in eff_left}), Counter({w: 1 for w in eff_right}))

                print(f"    Bootstrap subs for \"{span_text}\": {subs}")
                continue

            # Test all pairwise splits (binary mappings), building right-to-left
            # Process in REVERSE order so constituents are cached before they're needed
            for split_pos in range(start_pos + span_length - 1, start_pos, -1):  # right-to-left within span
                left_text = " ".join(tokens[start_pos:split_pos])
                right_text = " ".join(tokens[split_pos:k])
                print(f"    Testing split: (\"{left_text}\" \"{right_text}\")")

                # Get ALL substitutes for left and right (including aggregation and expansion)
                # This ensures aggregation substitutes are available as seeds for expansion
                left_subs_all = get_all_substitutes_merged(subparse_cache, left_text)
                right_subs_all = get_all_substitutes_merged(subparse_cache, right_text)

                # Also get best parse for energy and basic info
                _, left_eff_left_base, left_eff_right_base, left_energy, left_split, _ = get_best_parse(subparse_cache, left_text)
                _, right_eff_left_base, right_eff_right_base, right_energy, right_split, _ = get_best_parse(subparse_cache, right_text)

                # Aggregate contexts from ALL substitutes (not just best parse)
                # This includes aggregation substitutes as seeds for expansion
                left_subs = [(text, score) for text, score, _, _ in left_subs_all]
                right_subs = [(text, score) for text, score, _, _ in right_subs_all]

                # Build context sets from ALL substitutes
                # Use Counters to track how many substitutes contribute each context word
                from collections import Counter
                left_eff_left_counts = Counter(left_eff_left_base) if left_eff_left_base else Counter()
                left_eff_right_counts = Counter(left_eff_right_base) if left_eff_right_base else Counter()
                right_eff_left_counts = Counter(right_eff_left_base) if right_eff_left_base else Counter()
                right_eff_right_counts = Counter(right_eff_right_base) if right_eff_right_base else Counter()

                # Add contexts from all substitutes (including aggregation)
                for sub_text, _, _, _ in left_subs_all:
                    sub_pattern = get_unit_with_fallback(sub_text, catalog, corpus_index)
                    if sub_pattern:
                        for w in sub_pattern.left_words.keys():
                            left_eff_left_counts[w] += 1
                        for w in sub_pattern.right_words.keys():
                            left_eff_right_counts[w] += 1

                for sub_text, _, _, _ in right_subs_all:
                    sub_pattern = get_unit_with_fallback(sub_text, catalog, corpus_index)
                    if sub_pattern:
                        for w in sub_pattern.left_words.keys():
                            right_eff_left_counts[w] += 1
                        for w in sub_pattern.right_words.keys():
                            right_eff_right_counts[w] += 1

                # Convert to lists for candidate generation (keys only)
                left_eff_left = list(left_eff_left_counts.keys())
                left_eff_right = list(left_eff_right_counts.keys())
                right_eff_left = list(right_eff_left_counts.keys())
                right_eff_right = list(right_eff_right_counts.keys())

                # Bootstrap right if not cached yet
                if not right_subs:
                    # Use constituent aggregation for multi-word units
                    right_eff_left, right_eff_right = aggregate_contexts_from_constituents(
                        right_text, subparse_cache, catalog, verbose=True
                    )
                    # For initial bootstrap, right_subs is just the unit itself
                    right_subs = [(right_text, 1.0)]
                    # Bootstrap has no split (identity), energy is high
                    right_energy = float('inf')
                    subparse_cache[(right_text, None)] = (right_subs, right_eff_left, right_eff_right, right_energy)

                    # Add to GPU index for dynamic scoring (bootstrap: uniform weights)
                    from collections import Counter
                    catalog.add_unit_to_gpu_index(right_text, Counter({w: 1 for w in right_eff_left}), Counter({w: 1 for w in right_eff_right}))

                # MUTUAL EXPANSION: Only proceed if we have context from BOTH sides
                # This ensures we have actual fillers to seed the expansion

                print(f"      Left context available: {len(left_eff_left)} left, {len(left_eff_right)} right")
                print(f"      Right context available: {len(right_eff_left)} left, {len(right_eff_right)} right")

                # Skip mutual expansion if either side has no context fillers
                if not left_eff_right and not right_eff_left:
                    print(f"      Skipping mutual expansion: no context fillers from either side yet")
                    # Cache the span with just the units themselves
                    combined_subs = [(left_text, 1.0), (right_text, 1.0)]
                    combined_energy = float('inf')  # No real substitutes
                    subparse_cache[(span_text, (left_text, right_text))] = (combined_subs, left_eff_left, left_eff_right, combined_energy)
                    html_buffer.append((span_text, (left_text, right_text), combined_energy, span_length))

                    # Add to GPU index for dynamic scoring (bootstrap: uniform weights)
                    from collections import Counter
                    catalog.add_unit_to_gpu_index(span_text, Counter({w: 1 for w in left_eff_left}), Counter({w: 1 for w in left_eff_right}))

                    continue

                # Step 1: Get substitutes for LEFT
                # Candidates ARE right_eff_left (left_words of RIGHT = words in LEFT's position)
                # Score them by similarity to LEFT's context patterns
                print(f"      [STEP 1] Get substitutes for LEFT: \"{left_text}\"")
                print(f"        Candidates: {len(right_eff_left)} words from RIGHT's left_words")
                left_scoring = 'containment'
                all_left_subs = catalog.gpu_contextual_candidates(
                    target=left_text,
                    candidates=right_eff_left,
                    max_results=max_class_members * 3,
                    scoring=left_scoring,
                    trace=True  # Debug scoring
                )
                all_left_subs = [(t, s) for t, s in all_left_subs if t != left_text]
                left_subs_for_expansion = all_left_subs[:max_class_members]
                print(f"        Found {len(all_left_subs)} scored, keeping top {len(left_subs_for_expansion)}")
                # TRACE: Check specific verbs
                for tv in ["leave", "left", "borrow", "borrowed", "wrote", "met", "called", "saw"]:
                    found_all = [(t, s) for t, s in all_left_subs if t == tv]
                    found_top = [(t, s) for t, s in left_subs_for_expansion if t == tv]
                    if found_all:
                        rank = next(i for i, (t, _) in enumerate(all_left_subs) if t == tv)
                        print(f"        TRACE LEFT: '{tv}' score={found_all[0][1]:.4f}, rank={rank}, in top 500={bool(found_top)}")

                # Collect right_words from left substitutes → candidates for RIGHT
                # Context consensus: only keep words shared by multiple substitutes
                print(f"      Left substitutes: {[t for t, _ in left_subs_for_expansion]}")
                context_word_counts = {}  # word → how many substitutes have it as right_word
                n_subs_with_pattern = 0
                for sub_text, _ in left_subs_for_expansion:
                    sub_pattern = get_unit_with_fallback(sub_text, catalog, corpus_index)
                    if sub_pattern:
                        n_subs_with_pattern += 1
                        for word in sub_pattern.right_words.keys():
                            context_word_counts[word] = context_word_counts.get(word, 0) + 1

                # Keep words shared by ≥ 30% of substitutes (minimum 2)
                # CONSENSUS_THRESHOLD = 0.3  # Original: filters to only common words
                CONSENSUS_THRESHOLD = 0.0  # Disabled: allow all context words through
                min_share = max(2, int(n_subs_with_pattern * CONSENSUS_THRESHOLD))
                right_contexts_of_left_element = {w for w, c in context_word_counts.items() if c >= min_share}
                print(f"      Context consensus: {len(context_word_counts)} raw → "
                      f"{len(right_contexts_of_left_element)} shared (≥{min_share}/{n_subs_with_pattern} subs)")

                # Step 2: Get substitutes for RIGHT
                # Candidates ARE left_eff_right (right_words of LEFT = words in RIGHT's position)
                # Score them by similarity to RIGHT's context patterns
                print(f"    [STEP 2] Get substitutes for RIGHT: \"{right_text}\"")
                print(f"      Candidates: {len(left_eff_right)} words from LEFT's right_words")
                right_scoring = 'containment'
                all_right_subs = catalog.gpu_contextual_candidates(
                    target=right_text,
                    candidates=left_eff_right,
                    max_results=max_class_members * 3,
                    scoring=right_scoring
                )
                all_right_subs = [(t, s) for t, s in all_right_subs if t != right_text]
                right_subs_for_expansion = all_right_subs[:max_class_members]
                print(f"      Found {len(all_right_subs)} scored, keeping top {len(right_subs_for_expansion)}")

                # Collect left_words from right substitutes → candidates for LEFT
                # Context consensus: only keep words shared by multiple substitutes
                print(f"      Right substitutes: {[t for t, _ in right_subs_for_expansion]}")
                context_word_counts_r = {}  # word → how many substitutes have it as left_word
                n_rsubs_with_pattern = 0
                for sub_text, _ in right_subs_for_expansion:
                    sub_pattern = get_unit_with_fallback(sub_text, catalog, corpus_index)
                    if sub_pattern:
                        n_rsubs_with_pattern += 1
                        for word in sub_pattern.left_words.keys():
                            context_word_counts_r[word] = context_word_counts_r.get(word, 0) + 1

                # Keep words shared by ≥ 30% of substitutes (minimum 2)
                # CONSENSUS_THRESHOLD = 0.3  # Original: filters to only common words
                CONSENSUS_THRESHOLD = 0.0  # Disabled: allow all context words through
                min_share_r = max(2, int(n_rsubs_with_pattern * CONSENSUS_THRESHOLD))
                left_contexts_of_right_element = {w for w, c in context_word_counts_r.items() if c >= min_share_r}
                print(f"      Context consensus: {len(context_word_counts_r)} raw → "
                      f"{len(left_contexts_of_right_element)} shared (≥{min_share_r}/{n_rsubs_with_pattern} subs)")

                # LEFT candidates = Step 1 subs (from *R, scored against *LEFT*)
                # RIGHT candidates = Step 2 subs (from L*, scored against *RIGHT*)
                # Steps 3-4 removed: they were a round-trip through single-word
                # intermediaries that lost the specificity of the *R connection.
                left_candidates = list(left_subs_for_expansion)
                right_candidates = list(right_subs_for_expansion)

                print(f"    Left candidates: {left_candidates[:5]}")
                print(f"    Right candidates: {right_candidates[:5]}")

                # CACHED SUBSTITUTES: Use lower-level cached substitutes as primary candidates
                # These are the multi-word substitutes already computed for each element
                # (e.g., "his cellar", "your towel" for "my hat") — they are fundamental
                # because their contexts record real corpus co-occurrences.
                # Single-word candidates from Steps 2-4 supplement these.

                if right_subs and len(right_subs) > 1:
                    # right_subs are cached lower-level substitutes for right_text
                    # Filter: keep those whose LEFT contexts include words distributionally
                    # similar to left_text. E.g., "his cellar" has left context "find" —
                    # score "find" against "found"'s contexts to check compatibility.
                    cached_right_texts = [t for t, s in right_subs if t != right_text]
                    if cached_right_texts:
                        print(f"    [CACHED SUBS] Filtering {len(cached_right_texts)} cached subs for RIGHT \"{right_text}\" by left-context compatibility with \"{left_text}\"")

                        # For each cached sub, collect its left contexts and score them
                        # as potential substitutes for left_text using containment
                        scored_cached_right = []
                        for sub_text, sub_score in right_subs:
                            if sub_text == right_text:
                                continue
                            sub_pattern = get_unit_with_fallback(sub_text, catalog, corpus_index)
                            if not sub_pattern:
                                continue
                            sub_left_ctx = list(sub_pattern.left_words.keys())
                            if not sub_left_ctx:
                                continue
                            # Score the sub's left contexts as candidates for left_text
                            # This asks: are the words preceding this sub (e.g. "find", "borrow")
                            # distributionally similar to left_text (e.g. "found")?
                            ctx_scores = catalog.gpu_contextual_candidates(
                                target=left_text,
                                candidates=sub_left_ctx,
                                max_results=5,
                                scoring=left_scoring
                            )
                            if ctx_scores:
                                # Use best matching left context as the score
                                best_ctx_score = ctx_scores[0][1]
                                if best_ctx_score > 0:
                                    scored_cached_right.append((sub_text, best_ctx_score))

                        scored_cached_right = sorted(scored_cached_right, key=lambda x: -x[1])
                        # TRACE: Check specific subs
                        trace_subs = ["your gloves", "his wife", "your towel"]
                        for ts in trace_subs:
                            found_in_cached = [(t, s) for t, s in scored_cached_right if t == ts]
                            found_in_right_subs = [(t, s) for t, s in right_subs if t == ts]
                            print(f"      TRACE cached RIGHT: '{ts}' in right_subs={bool(found_in_right_subs)} (score={found_in_right_subs}), in scored_cached_right={bool(found_in_cached)} (score={found_in_cached})")
                        print(f"      Found {len(scored_cached_right)} cached RIGHT subs with compatible left contexts")
                        if scored_cached_right:
                            print(f"      Sample cached RIGHT: {scored_cached_right[:5]}")
                            # Merge: cached subs are primary, single-word subs supplement
                            existing_texts = {t for t, _ in scored_cached_right}
                            for t, s in right_candidates:
                                if t not in existing_texts:
                                    scored_cached_right.append((t, s))
                                    existing_texts.add(t)
                            right_candidates = sorted(scored_cached_right, key=lambda x: -x[1])[:max_class_members * 2]
                            print(f"      Merged RIGHT candidates: {len(right_candidates)} total ({sum(1 for t, _ in right_candidates if ' ' in t)} multi-word)")

                if left_subs and len(left_subs) > 1:
                    # left_subs are cached lower-level substitutes for left_text
                    # Filter: keep those whose RIGHT contexts include words distributionally
                    # similar to right_text.
                    cached_left_texts = [t for t, s in left_subs if t != left_text]
                    if cached_left_texts:
                        print(f"    [CACHED SUBS] Filtering {len(cached_left_texts)} cached subs for LEFT \"{left_text}\" by right-context compatibility with \"{right_text}\"")

                        scored_cached_left = []
                        for sub_text, sub_score in left_subs:
                            if sub_text == left_text:
                                continue
                            sub_pattern = get_unit_with_fallback(sub_text, catalog, corpus_index)
                            if not sub_pattern:
                                continue
                            sub_right_ctx = list(sub_pattern.right_words.keys())
                            if not sub_right_ctx:
                                continue
                            # Score the sub's right contexts as candidates for right_text
                            # This asks: are the words following this sub distributionally
                            # similar to right_text?
                            ctx_scores = catalog.gpu_contextual_candidates(
                                target=right_text,
                                candidates=sub_right_ctx,
                                max_results=5,
                                scoring=right_scoring
                            )
                            if ctx_scores:
                                best_ctx_score = ctx_scores[0][1]
                                if best_ctx_score > 0:
                                    scored_cached_left.append((sub_text, best_ctx_score))

                        scored_cached_left = sorted(scored_cached_left, key=lambda x: -x[1])
                        print(f"      Found {len(scored_cached_left)} cached LEFT subs with compatible right contexts")
                        if scored_cached_left:
                            print(f"      Sample cached LEFT: {scored_cached_left[:5]}")
                            # Merge: cached subs are primary, single-word subs supplement
                            existing_texts = {t for t, _ in scored_cached_left}
                            for t, s in left_candidates:
                                if t not in existing_texts:
                                    scored_cached_left.append((t, s))
                                    existing_texts.add(t)
                            left_candidates = sorted(scored_cached_left, key=lambda x: -x[1])[:max_class_members * 2]
                            print(f"      Merged LEFT candidates: {len(left_candidates)} total ({sum(1 for t, _ in left_candidates if ' ' in t)} multi-word)")

                # Compute energies for left and right based on their substitution classes
                left_consensus = compute_consensus_score(left_subs_for_expansion, catalog, left_text, corpus_index=corpus_index)
                left_energy = compute_energy(len(left_subs_for_expansion), left_consensus)

                right_consensus = compute_consensus_score(right_subs_for_expansion, catalog, right_text, corpus_index=corpus_index)
                right_energy = compute_energy(len(right_subs_for_expansion), right_consensus)

                print(f"    LEFT energy: {left_energy:.2f} (n={len(left_subs_for_expansion)}, consensus={left_consensus:.3f})")
                print(f"    RIGHT energy: {right_energy:.2f} (n={len(right_subs_for_expansion)}, consensus={right_consensus:.3f})")

                # Update cache for whole prefix
                # Combine left and right candidates as substitutes for the full span
                # Form cartesian product: each left substitute paired with each right substitute
                combined_candidates = []
                external_filtered_candidates = []
                origins_dict = {}  # Track origin of each substitute

                # DEBUG counters
                total_combinations_tried = 0
                combinations_in_catalog = 0
                combinations_with_external_context = 0

                # Compute outer context (words outside the current span in the full sentence)
                outer_left_context = list(tokens[0:start_pos]) if start_pos > 0 else []
                outer_right_context = list(tokens[k:]) if k < len(tokens) else []

                # DEBUG: Check candidate types
                left_multiword = [t for t, _ in left_candidates if ' ' in t]
                right_multiword = [t for t, _ in right_candidates if ' ' in t]
                print(f"    Candidate types: LEFT has {len(left_multiword)} multi-word (of {len(left_candidates)}), RIGHT has {len(right_multiword)} multi-word (of {len(right_candidates)})")

                # DEBUG: Trace specific examples
                left_candidate_dict = {t: s for t, s in left_candidates}
                right_candidate_dict = {t: s for t, s in right_candidates}
                trace_pairs = [("leave", "your gloves"), ("left", "his wife"), ("borrow", "your towel"), ("borrowed", "your towel")]
                for tl, tr in trace_pairs:
                    l_in = tl in left_candidate_dict
                    r_in = tr in right_candidate_dict
                    combo = f"{tl} {tr}"
                    combo_exists = get_unit_with_fallback(combo, catalog, corpus_index) is not None if l_in and r_in else None
                    print(f"    TRACE: '{tl}' in LEFT={l_in} (score={left_candidate_dict.get(tl, 'N/A')}), '{tr}' in RIGHT={r_in} (score={right_candidate_dict.get(tr, 'N/A')}), '{combo}' in corpus={combo_exists}")

                for left_sub, left_score in left_candidates:
                    for right_sub, right_score in right_candidates:
                        # Form the combined phrase
                        combined_text = f"{left_sub} {right_sub}"
                        total_combinations_tried += 1

                        # Only keep combinations that exist in catalog or corpus
                        # This avoids synthetic phrases with no observed contexts
                        combined_pattern = get_unit_with_fallback(combined_text, catalog, corpus_index)
                        if not combined_pattern:
                            continue

                        combinations_in_catalog += 1

                        # UNIFIED SCORING: Two components
                        # 1. Context path counting (NEW)
                        direct_bridges, left_validated, right_validated = count_bridging_paths(
                            left_sub, right_sub, left_text, right_text, catalog, corpus_index
                        )
                        # Normalize path counts (log scale to prevent explosion)
                        import math
                        path_score = math.log(1 + direct_bridges + left_validated + right_validated)

                        # 2. Inner boundary diversity
                        # Measure diversity among substitutes' contexts at the boundary
                        # HIGH diversity = open contexts = good compositional unit

                        # LEFT element: measure RIGHT-facing diversity (contexts at boundary)
                        # Use unique/total ratio for more breadth
                        left_right_diversity = 0.0
                        if len(left_candidates) >= 3:
                            all_right_contexts = []
                            for lsub, _ in left_candidates[:100]:
                                lsub_pattern = get_unit_with_fallback(lsub, catalog, corpus_index)
                                if lsub_pattern:
                                    all_right_contexts.extend(lsub_pattern.right_words.keys())
                            if len(all_right_contexts) > 0:
                                unique = len(set(all_right_contexts))
                                total = len(all_right_contexts)
                                left_right_diversity = unique / total  # High unique/total = high diversity

                        # RIGHT element: measure LEFT-facing diversity (contexts at boundary)
                        right_left_diversity = 0.0
                        if len(right_candidates) >= 3:
                            all_left_contexts = []
                            for rsub, _ in right_candidates[:100]:
                                rsub_pattern = get_unit_with_fallback(rsub, catalog, corpus_index)
                                if rsub_pattern:
                                    all_left_contexts.extend(rsub_pattern.left_words.keys())
                            if len(all_left_contexts) > 0:
                                unique = len(set(all_left_contexts))
                                total = len(all_left_contexts)
                                right_left_diversity = unique / total  # High unique/total = high diversity

                        # Boundary diversity bonus: both sides should have diverse contexts
                        boundary_diversity = 0.0
                        if left_right_diversity > 0 and right_left_diversity > 0:
                            boundary_diversity = math.sqrt(left_right_diversity * right_left_diversity)

                        # 3. Outer context diversity (breaks symmetry)
                        # Measure diversity among SUBSTITUTES' outer contexts
                        # HIGH diversity = open, productive contexts = good unit

                        # LEFT element: measure diversity in LEFT contexts (outer)
                        left_outer_diversity = 0.0
                        if len(left_candidates) >= 3:
                            all_left_outer_contexts = []
                            for lsub, _ in left_candidates[:100]:
                                lsub_pattern = get_unit_with_fallback(lsub, catalog, corpus_index)
                                if lsub_pattern:
                                    all_left_outer_contexts.extend(lsub_pattern.left_words.keys())
                            if len(all_left_outer_contexts) > 0:
                                unique = len(set(all_left_outer_contexts))
                                total = len(all_left_outer_contexts)
                                left_outer_diversity = unique / total  # High unique/total = high diversity

                        # RIGHT element: measure diversity in RIGHT contexts (outer)
                        right_outer_diversity = 0.0
                        if len(right_candidates) >= 3:
                            all_right_outer_contexts = []
                            for rsub, _ in right_candidates[:100]:
                                rsub_pattern = get_unit_with_fallback(rsub, catalog, corpus_index)
                                if rsub_pattern:
                                    all_right_outer_contexts.extend(rsub_pattern.right_words.keys())
                            if len(all_right_outer_contexts) > 0:
                                unique = len(set(all_right_outer_contexts))
                                total = len(all_right_outer_contexts)
                                right_outer_diversity = unique / total  # High unique/total = high diversity

                        # Combined: geometric mean of diversity values
                        diversity_bonus = 0.0
                        if left_outer_diversity > 0 and right_outer_diversity > 0:
                            diversity_bonus = math.sqrt(left_outer_diversity * right_outer_diversity)

                        # 4. Outer context consensus (existing - checks sentence context)
                        outer_context_score = compute_outer_context_score(
                            combined_text, outer_left_context, outer_right_context, catalog, corpus_index
                        )

                        # Combined score: base * boundary_diversity * paths * outer_diversity * outer_context
                        # Weight both diversities heavily to reward compositional units
                        combined_score = left_score * right_score * (1.0 + boundary_diversity * 3.0) * (1.0 + path_score * 0.1) * (1.0 + diversity_bonus * 5.0) * (1.0 + outer_context_score)

                        # DEBUG: Track multi-word scoring
                        if ' ' in combined_text and combinations_in_catalog <= 5:
                            print(f"      MULTIWORD: '{combined_text}' = {left_sub}+{right_sub}, score={combined_score:.4f} (base={left_score*right_score:.4f}, boundary_div={boundary_diversity:.4f} [L→R={left_right_diversity:.3f},R→L={right_left_diversity:.3f}], paths={path_score:.4f}, outer_div={diversity_bonus:.4f} [L={left_outer_diversity:.3f},R={right_outer_diversity:.3f}], outer_ctx={outer_context_score:.4f})")

                        combined_candidates.append((combined_text, combined_score))

                        # Track origin: which left and right subs were combined
                        origins_dict[combined_text] = {
                            'left_sub': left_sub,
                            'left_score': left_score,
                            'left_source': right_text,  # Left subs come from left context of right element
                            'right_sub': right_sub,
                            'right_score': right_score,
                            'right_source': left_text,  # Right subs come from right context of left element
                            'parent_split': (left_text, right_text)  # The split that created this substitute
                        }

                        # NEW: Also filter multi-word combinations on external context
                        # Check if any substitute of the left element can precede this combination
                        # This tests compatibility with the substitution class, not just the head phrase
                        compatible_left_subs = [lsub for lsub, _ in left_candidates
                                               if lsub in combined_pattern.left_words]
                        if compatible_left_subs:
                            # Track which external context this came from
                            combinations_with_external_context += 1
                            context_info = f"filtered_on_L_subs:{','.join(compatible_left_subs[:3])}"
                            external_filtered_candidates.append((combined_text, combined_score, context_info))

                print(f"    Combination statistics: {total_combinations_tried} tried, {combinations_in_catalog} in catalog, {combinations_with_external_context} with external context")
                if combinations_in_catalog > 0 and combinations_in_catalog <= 10:
                    print(f"    Combinations found in catalog: {[t for t, s in combined_candidates[:10]]}")

                # Sort by score and keep top per word-length
                # Separate lists prevent shorter subs from crowding out full-length ones
                from collections import defaultdict
                by_length = defaultdict(list)
                for t, s in combined_candidates:
                    by_length[len(t.split())].append((t, s))

                combined_candidates = []
                for wlen in sorted(by_length.keys()):
                    bucket = sorted(by_length[wlen], key=lambda x: -x[1])[:max_class_members * 3]
                    combined_candidates.extend(bucket)

                external_filtered_candidates = sorted(external_filtered_candidates, key=lambda x: -x[1])[:max_class_members * 2]

                print(f"    Found {len(combined_candidates)} combined candidates, {len(external_filtered_candidates)} externally filtered")

                # DEBUG: Show what types of candidates we have by word-length
                for wlen in sorted(by_length.keys()):
                    bucket = by_length[wlen]
                    kept = [t for t, s in combined_candidates if len(t.split()) == wlen]
                    top3 = sorted(bucket, key=lambda x: -x[1])[:3]
                    print(f"    {wlen}-word subs: {len(bucket)} found, {len(kept)} kept, top: {[(t, f'{s:.3f}') for t, s in top3]}")
                multi_word_combos = [t for t, s in combined_candidates if ' ' in t]
                if multi_word_combos:
                    print(f"    Sample multi-word combinations: {multi_word_combos[:5]}")
                else:
                    print(f"    WARNING: No multi-word combinations despite {combinations_in_catalog} combinations in catalog")

                # NEW: Multi-level aggregation - combine existing aggregation substitutes
                print(f"    [MULTI-LEVEL AGGREGATION] Attempting aggregations of aggregations...")
                multilevel_candidates = []
                multilevel_stats = {'tried': 0, 'in_catalog': 0, 'scored': 0}
                multilevel_samples_not_found = []  # Track samples that weren't in catalog

                # Direction 1: Get aggregation substitutes for LEFT, combine with ALL substitutes for RIGHT
                left_agg_subs = get_all_substitutes_merged(subparse_cache, left_text)
                left_agg_subs = [(t, s, src, orig) for t, s, src, orig in left_agg_subs if src == 'aggregation']
                print(f"      Found {len(left_agg_subs)} aggregation substitutes for left element \"{left_text}\"")

                # Get ALL substitutes for right (both aggregation and expansion)
                right_all_subs = get_all_substitutes_merged(subparse_cache, right_text)
                print(f"      Found {len(right_all_subs)} total substitutes for right element \"{right_text}\"")

                if left_agg_subs and right_all_subs:
                    for left_agg_text, left_agg_score, _, left_agg_orig in left_agg_subs[:100]:
                        for right_sub_text, right_sub_score, _, _ in right_all_subs[:100]:
                            multilevel_text = f"{left_agg_text} {right_sub_text}"
                            multilevel_stats['tried'] += 1

                            # Check if direct phrase-level combination exists in catalog or corpus
                            multilevel_pattern = get_unit_with_fallback(multilevel_text, catalog, corpus_index, debug=True)
                            if not multilevel_pattern:
                                # Track a few samples of combinations not found
                                if len(multilevel_samples_not_found) < 5:
                                    multilevel_samples_not_found.append(f"{left_agg_text} + {right_sub_text} = {multilevel_text}")
                                continue

                            multilevel_stats['in_catalog'] += 1

                            # UNIFIED SCORING (same as regular aggregation)
                            # 1. Context path counting
                            direct_bridges, left_validated, right_validated = count_bridging_paths(
                                left_agg_text, right_sub_text, left_text, right_text, catalog, corpus_index
                            )
                            import math
                            path_score = math.log(1 + direct_bridges + left_validated + right_validated)

                            # 2. Inner boundary diversity (use unique/total ratio)
                            # LEFT element: measure RIGHT-facing diversity (contexts at boundary)
                            left_right_diversity = 0.0
                            if len(left_agg_subs) >= 3:
                                all_right_contexts = []
                                for lsub_text, _, _, _ in left_agg_subs[:100]:
                                    lsub_pattern = get_unit_with_fallback(lsub_text, catalog, corpus_index)
                                    if lsub_pattern:
                                        all_right_contexts.extend(lsub_pattern.right_words.keys())
                                if len(all_right_contexts) > 0:
                                    unique = len(set(all_right_contexts))
                                    total = len(all_right_contexts)
                                    left_right_diversity = unique / total

                            # RIGHT element: measure LEFT-facing diversity (contexts at boundary)
                            right_left_diversity = 0.0
                            if len(right_all_subs) >= 3:
                                all_left_contexts = []
                                for rsub_text, _, _, _ in right_all_subs[:100]:
                                    rsub_pattern = get_unit_with_fallback(rsub_text, catalog, corpus_index)
                                    if rsub_pattern:
                                        all_left_contexts.extend(rsub_pattern.left_words.keys())
                                if len(all_left_contexts) > 0:
                                    unique = len(set(all_left_contexts))
                                    total = len(all_left_contexts)
                                    right_left_diversity = unique / total

                            boundary_diversity = 0.0
                            if left_right_diversity > 0 and right_left_diversity > 0:
                                boundary_diversity = math.sqrt(left_right_diversity * right_left_diversity)

                            # 3. Outer context diversity (use unique/total ratio)
                            left_outer_diversity = 0.0
                            if len(left_agg_subs) >= 3:
                                all_left_outer_contexts = []
                                for lsub_text, _, _, _ in left_agg_subs[:100]:
                                    lsub_pattern = get_unit_with_fallback(lsub_text, catalog, corpus_index)
                                    if lsub_pattern:
                                        all_left_outer_contexts.extend(lsub_pattern.left_words.keys())
                                if len(all_left_outer_contexts) > 0:
                                    unique = len(set(all_left_outer_contexts))
                                    total = len(all_left_outer_contexts)
                                    left_outer_diversity = unique / total

                            right_outer_diversity = 0.0
                            if len(right_all_subs) >= 3:
                                all_right_outer_contexts = []
                                for rsub_text, _, _, _ in right_all_subs[:100]:
                                    rsub_pattern = get_unit_with_fallback(rsub_text, catalog, corpus_index)
                                    if rsub_pattern:
                                        all_right_outer_contexts.extend(rsub_pattern.right_words.keys())
                                if len(all_right_outer_contexts) > 0:
                                    unique = len(set(all_right_outer_contexts))
                                    total = len(all_right_outer_contexts)
                                    right_outer_diversity = unique / total

                            diversity_bonus = 0.0
                            if left_outer_diversity > 0 and right_outer_diversity > 0:
                                diversity_bonus = math.sqrt(left_outer_diversity * right_outer_diversity)

                            # 4. Outer context score
                            outer_context_score = compute_outer_context_score(
                                multilevel_text, outer_left_context, outer_right_context, catalog, corpus_index
                            )

                            # Combined score
                            multilevel_score = left_agg_score * right_sub_score * (1.0 + boundary_diversity * 3.0) * (1.0 + path_score * 0.1) * (1.0 + diversity_bonus * 5.0) * (1.0 + outer_context_score)

                            if multilevel_score > 0:
                                multilevel_stats['scored'] += 1
                                multilevel_candidates.append((multilevel_text, multilevel_score))

                                # Track origin for this multi-level aggregation
                                origins_dict[multilevel_text] = {
                                    'left_sub': left_agg_text,
                                    'left_score': left_agg_score,
                                    'left_source': f"aggregation_of_{left_text}",
                                    'right_sub': right_sub_text,
                                    'right_score': right_sub_score,
                                    'right_source': right_text,
                                    'parent_split': (left_text, right_text),
                                    'multilevel': True  # Mark as multi-level aggregation
                                }

                # Direction 2: Get aggregation substitutes for RIGHT, combine with ALL substitutes for LEFT
                right_agg_subs = get_all_substitutes_merged(subparse_cache, right_text)
                right_agg_subs = [(t, s, src, orig) for t, s, src, orig in right_agg_subs if src == 'aggregation']
                print(f"      Found {len(right_agg_subs)} aggregation substitutes for right element \"{right_text}\"")

                # Get ALL substitutes for left
                left_all_subs = get_all_substitutes_merged(subparse_cache, left_text)
                print(f"      Found {len(left_all_subs)} total substitutes for left element \"{left_text}\"")

                if right_agg_subs and left_all_subs:
                    for left_sub_text, left_sub_score, _, _ in left_all_subs[:100]:
                        for right_agg_text, right_agg_score, _, right_agg_orig in right_agg_subs[:100]:
                            multilevel_text = f"{left_sub_text} {right_agg_text}"
                            multilevel_stats['tried'] += 1

                            # Check if direct phrase-level combination exists in catalog or corpus
                            multilevel_pattern = get_unit_with_fallback(multilevel_text, catalog, corpus_index, debug=True)
                            if not multilevel_pattern:
                                # Track a few samples of combinations not found
                                if len(multilevel_samples_not_found) < 5:
                                    multilevel_samples_not_found.append(f"{left_sub_text} + {right_agg_text} = {multilevel_text}")
                                continue

                            multilevel_stats['in_catalog'] += 1

                            # UNIFIED SCORING (same as regular aggregation)
                            # 1. Context path counting
                            direct_bridges, left_validated, right_validated = count_bridging_paths(
                                left_sub_text, right_agg_text, left_text, right_text, catalog, corpus_index
                            )
                            import math
                            path_score = math.log(1 + direct_bridges + left_validated + right_validated)

                            # 2. Inner boundary diversity (use unique/total ratio)
                            left_right_diversity = 0.0
                            if len(left_all_subs) >= 3:
                                all_right_contexts = []
                                for lsub_text, _, _, _ in left_all_subs[:100]:
                                    lsub_pattern = get_unit_with_fallback(lsub_text, catalog, corpus_index)
                                    if lsub_pattern:
                                        all_right_contexts.extend(lsub_pattern.right_words.keys())
                                if len(all_right_contexts) > 0:
                                    unique = len(set(all_right_contexts))
                                    total = len(all_right_contexts)
                                    left_right_diversity = unique / total

                            right_left_diversity = 0.0
                            if len(right_agg_subs) >= 3:
                                all_left_contexts = []
                                for rsub_text, _, _, _ in right_agg_subs[:100]:
                                    rsub_pattern = get_unit_with_fallback(rsub_text, catalog, corpus_index)
                                    if rsub_pattern:
                                        all_left_contexts.extend(rsub_pattern.left_words.keys())
                                if len(all_left_contexts) > 0:
                                    unique = len(set(all_left_contexts))
                                    total = len(all_left_contexts)
                                    right_left_diversity = unique / total

                            boundary_diversity = 0.0
                            if left_right_diversity > 0 and right_left_diversity > 0:
                                boundary_diversity = math.sqrt(left_right_diversity * right_left_diversity)

                            # 3. Outer context diversity (use unique/total ratio)
                            left_outer_diversity = 0.0
                            if len(left_all_subs) >= 3:
                                all_left_outer_contexts = []
                                for lsub_text, _, _, _ in left_all_subs[:100]:
                                    lsub_pattern = get_unit_with_fallback(lsub_text, catalog, corpus_index)
                                    if lsub_pattern:
                                        all_left_outer_contexts.extend(lsub_pattern.left_words.keys())
                                if len(all_left_outer_contexts) > 0:
                                    unique = len(set(all_left_outer_contexts))
                                    total = len(all_left_outer_contexts)
                                    left_outer_diversity = unique / total

                            right_outer_diversity = 0.0
                            if len(right_agg_subs) >= 3:
                                all_right_outer_contexts = []
                                for rsub_text, _, _, _ in right_agg_subs[:100]:
                                    rsub_pattern = get_unit_with_fallback(rsub_text, catalog, corpus_index)
                                    if rsub_pattern:
                                        all_right_outer_contexts.extend(rsub_pattern.right_words.keys())
                                if len(all_right_outer_contexts) > 0:
                                    unique = len(set(all_right_outer_contexts))
                                    total = len(all_right_outer_contexts)
                                    right_outer_diversity = unique / total

                            diversity_bonus = 0.0
                            if left_outer_diversity > 0 and right_outer_diversity > 0:
                                diversity_bonus = math.sqrt(left_outer_diversity * right_outer_diversity)

                            # 4. Outer context score
                            outer_context_score = compute_outer_context_score(
                                multilevel_text, outer_left_context, outer_right_context, catalog, corpus_index
                            )

                            # Combined score
                            multilevel_score = left_sub_score * right_agg_score * (1.0 + boundary_diversity * 3.0) * (1.0 + path_score * 0.1) * (1.0 + diversity_bonus * 5.0) * (1.0 + outer_context_score)

                            if multilevel_score > 0:
                                multilevel_stats['scored'] += 1
                                multilevel_candidates.append((multilevel_text, multilevel_score))

                                # Track origin
                                origins_dict[multilevel_text] = {
                                    'left_sub': left_sub_text,
                                    'left_score': left_sub_score,
                                    'left_source': left_text,
                                    'right_sub': right_agg_text,
                                    'right_score': right_agg_score,
                                    'right_source': f"aggregation_of_{right_text}",
                                    'parent_split': (left_text, right_text),
                                    'multilevel': True
                                }

                # Add multi-level aggregations to combined_candidates
                # Always print stats, even if no candidates found
                print(f"      Multi-level stats: {multilevel_stats['tried']} tried, {multilevel_stats['in_catalog']} in catalog, {multilevel_stats['scored']} scored")

                if multilevel_samples_not_found:
                    print(f"      Sample combinations NOT in corpus:")
                    for sample in multilevel_samples_not_found[:5]:
                        print(f"        ✗ {sample}")

                if multilevel_candidates:
                    # Remove duplicates and sort by score
                    multilevel_unique = {}
                    for text, score in multilevel_candidates:
                        if text not in multilevel_unique or score > multilevel_unique[text]:
                            multilevel_unique[text] = score

                    multilevel_candidates = [(text, score) for text, score in multilevel_unique.items()]
                    multilevel_candidates = sorted(multilevel_candidates, key=lambda x: -x[1])  # No arbitrary limit - let them compete

                    print(f"      Found {len(multilevel_candidates)} unique multi-level aggregations")
                    if multilevel_candidates:
                        print(f"      Sample multi-level aggregations:")
                        for i, (text, score) in enumerate(multilevel_candidates[:5], 1):
                            print(f"        {i}. \"{text}\" (score={score:.3f})")

                    # Merge into combined_candidates (per word-length to avoid crowding)
                    combined_candidates.extend(multilevel_candidates)
                    by_length_ml = defaultdict(list)
                    for t, s in combined_candidates:
                        by_length_ml[len(t.split())].append((t, s))
                    combined_candidates = []
                    for wlen in sorted(by_length_ml.keys()):
                        bucket = sorted(by_length_ml[wlen], key=lambda x: -x[1])[:max_class_members * 4]
                        combined_candidates.extend(bucket)
                else:
                    print(f"      No combinations passed catalog/corpus check (none of the {multilevel_stats['tried']} combinations exist in corpus)")

                # Merge current substitutes with externally filtered multi-word combinations
                # Convert to uniform format first: (text, score, context_info)
                all_substitutes_with_context = []

                # Add combinations from mutual expansion
                for text, score in combined_candidates:
                    all_substitutes_with_context.append((text, score, f"from_mutual_expansion"))

                # Add externally filtered multi-word units (may overlap with above)
                for text, score, context_info in external_filtered_candidates:
                    # Check if not already added
                    if not any(t == text for t, _, _ in all_substitutes_with_context):
                        all_substitutes_with_context.append((text, score, context_info))

                # Sort by score and keep top per word-length
                by_length_final = defaultdict(list)
                for t, s, c in all_substitutes_with_context:
                    by_length_final[len(t.split())].append((t, s, c))
                all_substitutes_with_context = []
                for wlen in sorted(by_length_final.keys()):
                    bucket = sorted(by_length_final[wlen], key=lambda x: -x[1])[:max_class_members * 3]
                    all_substitutes_with_context.extend(bucket)

                # Convert back to (text, score) format for energy computation
                combined_candidates = [(text, score) for text, score, _ in all_substitutes_with_context]

                # Compute energy using Hamiltonian-aligned consensus (among substitutes' contexts)
                # Measure consensus among combined_candidates' contexts (not with target)
                # Use simple unique/total ratio for global diversity measure
                combined_consensus = 0.01  # Default
                if len(combined_candidates) >= 3:
                    # Collect all contexts from top substitutes (use more for breadth)
                    all_left_contexts = []
                    all_right_contexts = []
                    for sub_text, _ in combined_candidates[:100]:
                        sub_pattern = get_unit_with_fallback(sub_text, catalog, corpus_index)
                        if sub_pattern:
                            all_left_contexts.extend(sub_pattern.left_words.keys())
                            all_right_contexts.extend(sub_pattern.right_words.keys())

                    # Compute consensus as 1 - (unique/total)
                    # High unique/total = high diversity = low consensus
                    left_consensus = 0.0
                    if len(all_left_contexts) > 0:
                        unique_left = len(set(all_left_contexts))
                        total_left = len(all_left_contexts)
                        left_consensus = 1.0 - (unique_left / total_left)

                    right_consensus = 0.0
                    if len(all_right_contexts) > 0:
                        unique_right = len(set(all_right_contexts))
                        total_right = len(all_right_contexts)
                        right_consensus = 1.0 - (unique_right / total_right)

                    # Average of left and right consensus
                    if left_consensus > 0 and right_consensus > 0:
                        combined_consensus = (left_consensus + right_consensus) / 2.0
                    elif left_consensus > 0:
                        combined_consensus = left_consensus
                    elif right_consensus > 0:
                        combined_consensus = right_consensus

                # Count full-length substitutes (matching span word count)
                full_length_subs_list = [(t, s) for t, s in combined_candidates if len(t.split()) == span_length]
                full_length_subs = len(full_length_subs_list)
                if span_length >= 3:
                    print(f"      TRACE full_length: {full_length_subs} subs with {span_length} words")
                    for t, s in full_length_subs_list[:10]:
                        print(f"        '{t}' (score={s:.4f})")
                    # Also check: is "borrow your towel" in combined_candidates at all?
                    byt = [(t, s) for t, s in combined_candidates if 'borrow your' in t]
                    if byt:
                        print(f"      TRACE 'borrow your*' in combined: {byt[:5]}")
                combined_energy = compute_energy(len(combined_candidates), combined_consensus, full_length_subs)

                # Update context words from both sides
                eff_left = list(set(left_eff_left) | set(right_eff_left))
                eff_right = list(set(left_eff_right) | set(right_eff_right))
                # Merge consensus counts from both sides for GPU index weighting
                eff_left_counts = left_eff_left_counts + right_eff_left_counts
                eff_right_counts = left_eff_right_counts + right_eff_right_counts

                # Cache with split structure, energy, and origins
                subparse_cache[(span_text, (left_text, right_text))] = (combined_candidates, eff_left, eff_right, combined_energy, origins_dict)

                # NEW: Collect parse info for HTML generation
                html_buffer.append((span_text, (left_text, right_text), combined_energy, span_length))

                # Also cache the constituent substitutes from this split
                # Use a marker ('from_expansion', parent_split) to indicate these came from mutual expansion
                # This allows reuse when processing longer phrases
                if left_subs_for_expansion:
                    # Cache substitutes generated for left constituent during this expansion
                    subparse_cache[(left_text, ('expansion', (left_text, right_text)))] = (
                        left_subs_for_expansion, left_eff_left, left_eff_right, left_energy
                    )

                if right_subs_for_expansion:
                    # Cache substitutes generated for right constituent during this expansion
                    subparse_cache[(right_text, ('expansion', (left_text, right_text)))] = (
                        right_subs_for_expansion, right_eff_left, right_eff_right, right_energy
                    )

                print(f"      COMBINED energy: {combined_energy:.2f} (n={len(combined_candidates)}, consensus={combined_consensus:.3f}, full_length={full_length_subs})")

                # Show top substitutes with context information
                print(f"      Top substitutes with context info:")
                for i, (text, score, context_info) in enumerate(all_substitutes_with_context[:5], 1):
                    print(f"        {i}. \"{text}\" (score={score:.3f}) [{context_info}]")

                # Add to GPU index for dynamic scoring with consensus-weighted contexts
                catalog.add_unit_to_gpu_index(span_text, eff_left_counts, eff_right_counts)

                print(f"      Cache updated for \"{span_text}\" split (\"{left_text}\", \"{right_text}\"): {len(combined_candidates)} combined candidates")

                # Store energies for this split in the parser
                # The parser will use these to compute span energies
                parser.store_split_energy(left_text, right_text, left_energy + right_energy)

    tree = parser.get_current_parse()

    # Final sections (original)
    print("\n[2] PARSE TREE")
    print("-" * 40)

    # Display the actual parse tree structure with energies
    if tree:
        def print_parse_tree(node, prefix="", is_left=None):
            """Recursively print parse tree with energies and frequencies."""
            # Determine connector based on position
            if is_left is None:
                connector = "└─ "
                extension = "   "
            elif is_left:
                connector = "├─ "
                extension = "│  "
            else:
                connector = "└─ "
                extension = "   "

            # Check if we should use the stored split instead of the unit
            # This shows the internal structure even when a unit was chosen
            use_split = (node.left is None and node.right is None and
                        node.best_split_left is not None and node.best_split_right is not None)

            if use_split:
                # Show as split using the alternative split structure
                print(f"{prefix}{connector}[SPLIT] E={node.best_split_energy:.2f} \"{node.span.text}\"")
                print_parse_tree(node.best_split_left, prefix + extension, is_left=True)
                print_parse_tree(node.best_split_right, prefix + extension, is_left=False)
            elif node.left is None and node.right is None:
                # True leaf node - this is a unit with no possible splits
                pattern = catalog.get_unit(node.span.text)
                freq = pattern.count if pattern else 0
                print(f"{prefix}{connector}[UNIT] E={node.energy:.2f} freq={freq} \"{node.span.text}\"")
            else:
                # Internal node - this is a split that was actually chosen
                print(f"{prefix}{connector}[SPLIT] E={node.energy:.2f} \"{node.span.text}\"")
                if node.left and node.right:
                    # Both children exist
                    print_parse_tree(node.left, prefix + extension, is_left=True)
                    print_parse_tree(node.right, prefix + extension, is_left=False)
                elif node.left:
                    # Only left child
                    print_parse_tree(node.left, prefix + extension, is_left=False)
                elif node.right:
                    # Only right child
                    print_parse_tree(node.right, prefix + extension, is_left=False)

        print_parse_tree(tree)
    else:
        print("No parse tree generated")

    print("\n[2b] ALL PARSE CANDIDATES EVALUATED (from mutual expansion)")
    print("-" * 40)

    # Show all spans that were parsed, with their candidates and frequencies
    print("CANDIDATE SUBSTITUTES BY SPAN:")
    # Group by text and show best parse for each
    texts_seen = set()
    for (span_text, split), cache_value in sorted(subparse_cache.items(), key=lambda x: len(x[0][0].split())):
        if span_text in texts_seen:
            continue
        texts_seen.add(span_text)

        # Handle variable-length cache entries (some have origins, some don't)
        if len(cache_value) == 5:
            subs, eff_left, eff_right, energy, origins = cache_value
        else:
            subs, eff_left, eff_right, energy = cache_value

        # Get best parse for this text
        best_subs, _, _, best_energy, best_split, _ = get_best_parse(subparse_cache, span_text)

        pattern = catalog.get_unit(span_text)
        freq = pattern.count if pattern else 0

        span_tokens = span_text.split()
        indent = "  " * (len(span_tokens) - 1)

        print(f"{indent}[{len(span_tokens)}-GRAM] \"{span_text}\" freq={freq} energy={best_energy:.2f} split={best_split}")

        # Show top 3 candidates for this span
        if best_subs:
            for sub_text, score in best_subs[:3]:
                sub_freq = catalog.get_unit(sub_text)
                sub_freq = sub_freq.count if sub_freq else 0
                print(f"{indent}  → \"{sub_text}\" (score={score:.6f}, freq={sub_freq})")
        else:
            print(f"{indent}  (Substitutes dependent on external context, so not cached)")

    print("\n[3] CONTEXTUAL SUBSTITUTION CLASSES")
    print("-" * 40)

    # Display substitution classes for nodes in the final parse tree
    # Use cached candidates from mutual expansion during incremental analysis
    if tree:
        def show_node_subs(node, depth=0):
            """Recursively show substitution classes for each node in parse tree."""
            indent = "  " * depth
            span_text = node.span.text

            # Look up this span in the cache - should have candidates from mutual expansion
            subs, eff_left, eff_right, energy, split, _ = get_best_parse(subparse_cache, span_text)

            print(f"{indent}Span: \"{span_text}\"")
            if subs:
                print(f"{indent}  Substitution class (from mutual expansion):")
                for sub_text, score in subs[:5]:  # Show top 5
                    print(f"{indent}    - {str(sub_text):<25s} (score: {score:.6f})")
            else:
                print(f"{indent}  (No substitutes found in cache)")

            # Recursively show left and right children
            if node.left:
                show_node_subs(node.left, depth + 1)
            if node.right:
                show_node_subs(node.right, depth + 1)

        show_node_subs(tree)

        # Also show all competing split candidates that were evaluated
        print(f"\n  All mutual expansion candidates evaluated:")
        texts_seen = set()
        for (span_text, split) in sorted(subparse_cache.keys(), key=lambda x: x[0]):
            if span_text in texts_seen:
                continue
            texts_seen.add(span_text)
            subs, _, _, energy, _, _ = get_best_parse(subparse_cache, span_text)
            if subs:
                print(f"    {span_text:<25s}: {len(subs)} candidates, top score: {subs[0][1]:.6f}, energy: {energy:.2f}")
    else:
        print("No parse tree to display")

    # Return tree, cache, and parse info buffer for HTML export
    print(f"\nCollected {len(html_buffer)} parses for HTML generation")
    return tree, subparse_cache, html_buffer

def export_html_tree(phrase: str, tree, subparse_cache: dict, catalog: UnitCatalog, parse_buffer: list, corpus_index: CorpusIndex = None, output_file: str = "parse_tree.html"):
    """
    Export parse tree as interactive HTML file.

    Args:
        phrase: Input phrase
        tree: Parse tree
        subparse_cache: Cache with all parses
        catalog: Unit catalog
        parse_buffer: Parse info buffer
        corpus_index: Optional corpus index for looking up phrases not in catalog
        output_file: Output HTML filename
    """

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Parse Tree: {phrase}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        h1 {{ color: #333; }}
        .tree {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .node {{ margin: 10px 0; padding: 10px; border-left: 3px solid #4CAF50; background: #fafafa; }}
        .node:hover {{ background: #f0f0f0; cursor: pointer; }}
        .span-text {{ font-weight: bold; color: #2196F3; cursor: pointer; }}
        .span-text:hover {{ text-decoration: underline; background: #e3f2fd; }}
        .energy {{ color: #FF5722; font-size: 0.9em; }}
        .split-info {{ color: #666; font-size: 0.85em; font-style: italic; }}
        .substitutes {{ display: none; margin-top: 10px; padding: 10px; background: #e8f5e9; border-radius: 4px; border: 2px solid #4CAF50; }}
        .substitutes.visible {{ display: block; }}
        .sub-item {{ margin: 5px 0; padding: 5px; background: white; border-radius: 3px; }}
        .score {{ color: #666; font-size: 0.9em; }}
        .substitute-text {{ font-weight: bold; position: relative; cursor: help; }}
        .substitute-text:hover {{ background: #ffeb3b; }}
        .sub-aggregation {{ color: #000; }}
        .sub-expansion {{ color: #f57c00; }}
        .sub-other {{ color: #666; }}
        .left-context {{ color: #d32f2f; font-size: 0.85em; }}
        .right-context {{ color: #388e3c; font-size: 0.85em; }}
        .context-label {{ font-weight: bold; font-size: 0.8em; color: #666; }}
        .origin-tooltip {{ display: none; position: absolute; bottom: 100%; left: 0; background: #333; color: white; padding: 8px; border-radius: 4px; white-space: nowrap; z-index: 1000; font-size: 0.9em; }}
        .substitute-text:hover .origin-tooltip {{ display: block; }}
        .length-breakdown {{ font-size: 1.1em; color: #1976D2; padding: 5px 0; }}
        .length-group {{ margin: 10px 0; }}
        .substitutes-by-length {{ margin-top: 5px; }}
        .substitutes-by-length.visible {{ display: block; }}
        .indent-1 {{ margin-left: 20px; }}
        .indent-2 {{ margin-left: 40px; }}
        .indent-3 {{ margin-left: 60px; }}
        .indent-4 {{ margin-left: 80px; }}
        .indent-5 {{ margin-left: 100px; }}
        /* Tab styles */
        .tab-bar {{ display: flex; gap: 0; margin-bottom: 0; }}
        .tab-btn {{ padding: 12px 24px; border: 2px solid #ccc; border-bottom: none; background: #e0e0e0;
                    cursor: pointer; font-size: 1em; font-weight: bold; border-radius: 8px 8px 0 0; color: #666; }}
        .tab-btn.active {{ background: white; color: #2196F3; border-color: #2196F3; }}
        .tab-panel {{ display: none; }}
        .tab-panel.active {{ display: block; }}
        #graph-view {{ background: white; border: 2px solid #2196F3; border-top: none; border-radius: 0 0 8px 8px;
                       padding: 10px; min-height: 600px; }}
        #graph-svg {{ width: 100%; height: 700px; }}
        .graph-node-label {{ font-size: 12px; font-family: Arial, sans-serif; pointer-events: none; }}
        .graph-tooltip {{ position: absolute; background: #333; color: white; padding: 8px 12px;
                          border-radius: 4px; font-size: 12px; pointer-events: none; z-index: 1000;
                          max-width: 350px; line-height: 1.4; }}
        #corpus-view {{ background: white; border: 2px solid #2196F3; border-top: none;
                        border-radius: 0 0 8px 8px; padding: 20px; min-height: 400px; }}
        #corpus-svg {{ width: 100%; overflow: visible; }}
        #containment-view {{ background: white; border: 2px solid #2196F3; border-top: none;
                             border-radius: 0 0 8px 8px; padding: 10px; min-height: 600px; }}
        #containment-svg {{ width: 100%; height: 800px; }}
    </style>
    <script>
        function toggleNode(id) {{
            const el = document.getElementById(id);
            if (!el) {{
                console.error('toggleNode: Element not found with id:', id);
                alert('Element not found: ' + id);
                return;
            }}
            const currentDisplay = window.getComputedStyle(el).display;
            console.log('toggleNode:', id, 'current display:', currentDisplay);
            if (currentDisplay === 'none') {{
                el.style.display = 'block';
                el.style.backgroundColor = '#fff9c4';
            }} else {{
                el.style.display = 'none';
                el.style.backgroundColor = '';
            }}
        }}
        function switchTab(tabName) {{
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
            document.getElementById(tabName).classList.add('active');
            document.querySelector('[data-tab="' + tabName + '"]').classList.add('active');
            if (tabName === 'graph-view' && typeof initGraph === 'function' && !window._graphInitialized) {{
                window._graphInitialized = true;
                initGraph();
            }}
            if (tabName === 'corpus-view' && typeof initCorpusView === 'function' && !window._corpusInitialized) {{
                window._corpusInitialized = true;
                initCorpusView();
            }}
            if (tabName === 'containment-view' && typeof initContainmentView === 'function' && !window._containmentInitialized) {{
                window._containmentInitialized = true;
                initContainmentView();
            }}
        }}
    </script>
</head>
<body>
    <h1>Parse Tree: "{phrase}"</h1>
    <div class="tab-bar">
        <button class="tab-btn active" data-tab="tree-view" onclick="switchTab('tree-view')">Tree View</button>
        <button class="tab-btn" data-tab="graph-view" onclick="switchTab('graph-view')">Graph View</button>
        <button class="tab-btn" data-tab="corpus-view" onclick="switchTab('corpus-view')">Corpus View</button>
        <button class="tab-btn" data-tab="containment-view" onclick="switchTab('containment-view')">Containment View</button>
    </div>
    <div id="tree-view" class="tab-panel active">
    <div class="tree">
"""

    # NEW: Render parse tree recursively from cache (SIMPLIFIED - no expensive lookups)
    def render_parse_from_cache_simple(span_text, split, cache, depth=0):
        indent_class = f"indent-{min(depth, 5)}"

        # Get this parse from cache
        cache_key = (span_text, split)
        if cache_key not in cache:
            # Just show as leaf
            return f'<div class="{indent_class}"><span class="span-text">{span_text}</span></div>'

        cache_value = cache[cache_key]
        energy = cache_value[3]
        num_subs = len(cache_value[0])

        # Simple display - just text, energy, and substitute count
        html_parts = []
        html_parts.append(f'<div class="{indent_class}">')
        html_parts.append(f'<span class="span-text">{span_text}</span> ')
        html_parts.append(f'<span class="energy">E={energy:.2f}</span> ')
        html_parts.append(f'<span class="score">({num_subs} subs)</span> ')

        # Substitutes section - merged from all parses
        html_parts.append(f'<div class="substitutes" id="{node_id}">')
        merged_subs = get_all_substitutes_merged(cache, span_text)

        if merged_subs:
            agg_count = sum(1 for _, _, src, _ in merged_subs if src == 'aggregation')
            exp_count = sum(1 for _, _, src, _ in merged_subs if src == 'expansion')
            html_parts.append(f'<strong>Substitutes ({len(merged_subs)}): ')
            html_parts.append(f'<span class="sub-aggregation">{agg_count} aggregation</span>, ')
            html_parts.append(f'<span class="sub-expansion">{exp_count} expansion</span></strong>')

            for sub_text, score, source_type, sub_origins in merged_subs[:50]:  # Limit to 50 for display
                sub_pattern = get_unit_with_fallback(sub_text, catalog, corpus_index)
                if sub_pattern:
                    left_ctx = [w for w, c in sub_pattern.left_words.most_common(5)]
                    right_ctx = [w for w, c in sub_pattern.right_words.most_common(5)]
                    left_ctx_str = ', '.join(left_ctx) if left_ctx else '(none)'
                    right_ctx_str = ', '.join(right_ctx) if right_ctx else '(none)'

                    origin_html = ''
                    if sub_origins:
                        left_path = f'"{sub_origins["left_source"]}" → "{sub_origins["left_sub"]}" ({sub_origins["left_score"]:.2f})'
                        right_path = f'"{sub_origins["right_source"]}" → "{sub_origins["right_sub"]}" ({sub_origins["right_score"]:.2f})'
                        # Check if this used word-level substitutions
                        if sub_origins.get('subparse_combo'):
                            origin_html = f'<span class="origin-tooltip">Word-level combination from {left_path} + {right_path} → "{sub_text}"</span>'
                        else:
                            origin_html = f'<span class="origin-tooltip">Aggregation: {left_path} + {right_path}</span>'
                    elif source_type == 'expansion':
                        origin_html = f'<span class="origin-tooltip">Expansion: from parent context</span>'

                    source_class = f'sub-{source_type}'
                    html_parts.append(f'<div class="sub-item">')
                    html_parts.append(f'<span class="context-label">L:</span> <span class="left-context">{left_ctx_str}</span> | ')
                    html_parts.append(f'<span class="substitute-text {source_class}">{sub_text}{origin_html}</span> ')
                    html_parts.append(f'<span class="score">(score: {score:.4f})</span> | ')
                    html_parts.append(f'<span class="context-label">R:</span> <span class="right-context">{right_ctx_str}</span>')
                    html_parts.append(f'</div>')
        html_parts.append('</div>')

        # Recursively render children if this is a binary split
        # Add depth limit to prevent infinite recursion
        if split and isinstance(split, tuple) and len(split) == 2 and depth < 20:
            # Handle special split types (aggregation, expansion)
            if split[0] in ['aggregation', 'expansion']:
                # For these, split[1] is a tuple of all tokens, not a binary split
                # Try to find a binary split by checking the cache for actual splits
                tokens_tuple = split[1]
                if len(tokens_tuple) == 2:
                    # Binary case - can render directly
                    left_text, right_text = tokens_tuple
                else:
                    # Multi-word case - find best binary split by checking cache
                    # Look for a split of this span in the cache
                    best_binary_split = None
                    for (cached_text, cached_split), _ in cache.items():
                        if cached_text == span_text and cached_split != split:
                            if isinstance(cached_split, tuple) and len(cached_split) == 2:
                                if cached_split[0] not in ['aggregation', 'expansion']:
                                    best_binary_split = cached_split
                                    break

                    if best_binary_split:
                        left_text, right_text = best_binary_split
                    else:
                        # No binary split found, just show tokens as leaves
                        for token in tokens_tuple:
                            html_parts.append(f'<div class="indent-{min(depth+1, 5)}"><span class="span-text">{token}</span></div>')
                        html_parts.append('</div>')
                        return ''.join(html_parts)
            else:
                # Regular binary split
                left_text, right_text = split

            # Only recurse if children are multi-word (not leaves)
            if ' ' in left_text:
                # Find best split for left child
                _, _, _, _, left_split, _ = get_best_parse(cache, left_text)
                html_parts.append(render_parse_from_cache(left_text, left_split, cache, catalog, depth + 1))
            else:
                # Leaf node - just show the word
                html_parts.append(f'<div class="indent-{min(depth+1, 5)}"><span class="span-text">{left_text}</span></div>')

            if ' ' in right_text:
                # Find best split for right child
                _, _, _, _, right_split, _ = get_best_parse(cache, right_text)
                html_parts.append(render_parse_from_cache(right_text, right_split, cache, catalog, depth + 1))
            else:
                # Leaf node - just show the word
                html_parts.append(f'<div class="indent-{min(depth+1, 5)}"><span class="span-text">{right_text}</span></div>')

        html_parts.append('</div>')
        return ''.join(html_parts)

    # Helper function to create consistent IDs from span text
    _node_counter = [0]  # Mutable container for closure
    def make_span_id(span_text):
        return f"subs-{span_text.replace(' ', '-')}"

    def next_node_id():
        _node_counter[0] += 1
        return _node_counter[0]

    # Generate tree HTML recursively
    def render_node(node, depth=0):
        indent_class = f"indent-{min(depth, 5)}"
        span_text = node.span.text

        # Get best parse for this span
        subs, _, _, energy, split, origins = get_best_parse(subparse_cache, span_text)

        node_id = make_span_id(span_text)

        html_parts = []
        html_parts.append(f'<div class="{indent_class}">')
        html_parts.append(f'<div class="node" onclick="toggleNode(\'{node_id}\')">')
        html_parts.append(f'<span class="span-text">{span_text}</span> ')
        html_parts.append(f'<span class="energy">E={energy:.2f}</span> ')
        if split:
            split_str = str(split) if isinstance(split, tuple) and split[0] != 'expansion' else 'expansion'
            html_parts.append(f'<span class="split-info">split: {split_str}</span>')
        html_parts.append('</div>')

        # Substitutes section (hidden by default) - MERGED from all parses
        html_parts.append(f'<div class="substitutes" id="{node_id}">')

        # Get ALL substitutes from all parses, merged
        merged_subs = get_all_substitutes_merged(subparse_cache, span_text)

        if merged_subs:
            # Count by source type
            agg_count = sum(1 for _, _, src, _ in merged_subs if src == 'aggregation')
            exp_count = sum(1 for _, _, src, _ in merged_subs if src == 'expansion')
            html_parts.append(f'<strong>Substitutes ({len(merged_subs)}): ')
            html_parts.append(f'<span class="sub-aggregation">{agg_count} aggregation</span>, ')
            html_parts.append(f'<span class="sub-expansion">{exp_count} expansion</span></strong>')

            # Show ALL merged substitutes (no limit) since they're hidden until clicked
            for sub_text, score, source_type, sub_origins in merged_subs:
                # Get contexts for this substitute from catalog or corpus index
                sub_pattern = get_unit_with_fallback(sub_text, catalog, corpus_index)
                if sub_pattern:
                    # Get top contexts (most frequent)
                    left_ctx = [w for w, c in sub_pattern.left_words.most_common(10)]
                    right_ctx = [w for w, c in sub_pattern.right_words.most_common(10)]
                    left_ctx_str = ', '.join(left_ctx) if left_ctx else '(none)'
                    right_ctx_str = ', '.join(right_ctx) if right_ctx else '(none)'

                    # Build origin tooltip based on source type
                    origin_html = ''
                    if sub_origins:
                        # Aggregation: show combination origin
                        left_path = f'"{sub_origins["left_source"]}" → "{sub_origins["left_sub"]}" ({sub_origins["left_score"]:.2f})'
                        right_path = f'"{sub_origins["right_source"]}" → "{sub_origins["right_sub"]}" ({sub_origins["right_score"]:.2f})'
                        # Check if this used word-level substitutions
                        if sub_origins.get('subparse_combo'):
                            origin_html = f'<span class="origin-tooltip">Word-level combination from {left_path} + {right_path} → "{sub_text}"</span>'
                        else:
                            origin_html = f'<span class="origin-tooltip">Aggregation: {left_path} + {right_path}</span>'
                    elif source_type == 'expansion':
                        # Expansion: show it came from parent context
                        origin_html = f'<span class="origin-tooltip">Expansion: from parent context mutual expansion</span>'

                    # Color code by source type
                    source_class = f'sub-{source_type}'

                    html_parts.append(f'<div class="sub-item">')
                    html_parts.append(f'<span class="context-label">L:</span> <span class="left-context">{left_ctx_str}</span> | ')
                    html_parts.append(f'<span class="substitute-text {source_class}">{sub_text}{origin_html}</span> ')
                    html_parts.append(f'<span class="score">(score: {score:.4f})</span> | ')
                    html_parts.append(f'<span class="context-label">R:</span> <span class="right-context">{right_ctx_str}</span>')
                    html_parts.append(f'</div>')
                else:
                    # No pattern in catalog, just show substitute
                    source_class = f'sub-{source_type}'
                    html_parts.append(f'<div class="sub-item"><span class="{source_class}">{sub_text}</span> <span class="score">(score: {score:.4f})</span></div>')
        else:
            html_parts.append('<em>No substitutes</em>')
        html_parts.append('</div>')

        # NEW: Show all alternative parses for this span
        all_parses = [(k, v) for k, v in subparse_cache.items() if k[0] == span_text]
        if len(all_parses) > 1:
            # Sort by energy (best first)
            all_parses_sorted = sorted(all_parses, key=lambda x: x[1][3])

            html_parts.append(f'<div class="alternatives" style="margin-top: 10px; padding: 10px; background: #fff3e0; border-radius: 4px;">')
            html_parts.append(f'<strong>Alternative parses ({len(all_parses)}):</strong>')

            for parse_key, parse_value in all_parses_sorted:
                _, parse_split = parse_key
                parse_subs, _, _, parse_energy = parse_value[:4]
                parse_origins = parse_value[4] if len(parse_value) == 5 else None

                is_current = (parse_split == split)
                style = 'font-weight: bold; color: #2196F3;' if is_current else ''

                split_str = str(parse_split) if isinstance(parse_split, tuple) else 'None'
                num_multiword = len([s for s, _ in parse_subs if ' ' in s])

                html_parts.append(f'<div style="margin: 5px 0; padding: 5px; background: white; border-radius: 3px; {style}">')
                html_parts.append(f'{"→ " if is_current else "  "}Split: {split_str} | ')
                html_parts.append(f'Energy: {parse_energy:.2f} | ')
                html_parts.append(f'Subs: {len(parse_subs)} ({num_multiword} multi-word)')
                if is_current:
                    html_parts.append(f' <em>(current)</em>')
                html_parts.append(f'</div>')

            html_parts.append('</div>')

        # Render children
        if node.left:
            html_parts.append(render_node(node.left, depth + 1))
        if node.right:
            html_parts.append(render_node(node.right, depth + 1))

        html_parts.append('</div>')
        return ''.join(html_parts)

    # Render alternative trees (simple structure, no expensive lookups)
    print(f"Adding {len(parse_buffer)} alternative parse trees...")

    # Find the BEST split for a span (lowest energy = winning parse)
    def find_best_split(span_text):
        best_split = None
        best_energy = float('inf')
        for text, split, energy, _ in parse_buffer:
            if text == span_text and energy < best_energy:
                best_split = split
                best_energy = energy
        return best_split

    # Render tree in bracket notation: (my (hat again))
    def render_as_brackets(span_text, split):
        if not split:
            return span_text

        if isinstance(split, tuple) and len(split) == 2:
            if split[0] in ['aggregation', 'expansion']:
                tokens = split[1]
                if len(tokens) == 2:
                    left_text, right_text = tokens
                else:
                    return f"({' '.join(tokens)})"
            else:
                left_text, right_text = split

            # Recursively render children with WINNING parse structure
            if ' ' in left_text:
                left_split = find_best_split(left_text)
                left_repr = render_as_brackets(left_text, left_split)
            else:
                left_repr = left_text

            if ' ' in right_text:
                right_split = find_best_split(right_text)
                right_repr = render_as_brackets(right_text, right_split)
            else:
                right_repr = right_text

            return f"({left_repr} {right_repr})"

        return span_text

    # Render a leaf node (single word) with substitutes
    def render_leaf(word_text, depth, context_side=None, context_word=None):
        """
        Render a single-word leaf.

        Args:
            word_text: The word to render
            depth: Indentation depth
            context_side: 'left' or 'right' indicating which side of parent split
            context_word: The sibling word that provides context
        """
        indent_class = f"indent-{min(depth, 5)}"
        # Make ID unique with global counter to avoid duplicate IDs across different parses
        span_id = make_span_id(word_text) + f'-leaf-{depth}-{next_node_id()}'

        result = f'<div class="{indent_class}">'
        result += f'<span class="span-text" onclick="toggleNode(\'{span_id}\')" style="cursor: pointer;">{word_text}</span>'
        # DEBUG: Show ID being used
        result += f' <span style="color: #999; font-size: 0.7em;">[id: {span_id}]</span>'

        # Add substitutes section for single word
        result += f'<div class="substitutes" id="{span_id}">'
        merged_subs = get_all_substitutes_merged(subparse_cache, word_text)
        if merged_subs:
            agg_count = sum(1 for _, _, src, _ in merged_subs if src == 'aggregation')
            exp_count = sum(1 for _, _, src, _ in merged_subs if src == 'expansion')
            result += f'<strong>Substitutes ({len(merged_subs)}): '
            result += f'<span class="sub-aggregation">{agg_count} aggregation</span>, '
            result += f'<span class="sub-expansion">{exp_count} expansion</span></strong>'

            for sub_text, score, source_type, sub_origins in merged_subs:  # Show all
                sub_pattern = get_unit_with_fallback(sub_text, catalog, corpus_index)
                if sub_pattern:
                    left_ctx = [w for w, c in sub_pattern.left_words.most_common(5)]
                    right_ctx = [w for w, c in sub_pattern.right_words.most_common(5)]
                    left_ctx_str = ', '.join(left_ctx) if left_ctx else '(none)'
                    right_ctx_str = ', '.join(right_ctx) if right_ctx else '(none)'

                    # Build origin tooltip (special handling for leaves with context info)
                    origin_html = ''
                    if context_word and source_type == 'expansion':
                        # For leaf nodes, show which word provided the context
                        side_label = 'Right' if context_side == 'right' else 'Left'
                        origin_html = f'<span class="origin-tooltip">{side_label} context of ({context_word})</span>'
                    elif sub_origins:
                        if 'left_sub' in sub_origins and 'right_sub' in sub_origins:
                            # Detailed aggregation: show which substitutes were combined and their context sources
                            left_sub = sub_origins['left_sub']
                            right_sub = sub_origins['right_sub']
                            left_source = sub_origins['left_source']
                            right_source = sub_origins['right_source']
                            origin_html = f'<span class="origin-tooltip">Aggregate from "{left_sub}" left context of ({left_source}) and "{right_sub}" right context of ({right_source})</span>'
                        elif sub_origins.get('split_type') == 'expansion':
                            # Expansion: show parent split
                            parent = sub_origins.get('parent_split', 'unknown')
                            origin_html = f'<span class="origin-tooltip">Expansion: from parent context {parent}</span>'
                        elif sub_origins.get('split_type') == 'aggregation':
                            # Simple aggregation: show the split
                            parent = sub_origins.get('parent_split', 'unknown')
                            origin_html = f'<span class="origin-tooltip">Aggregation: from constituent combination {parent}</span>'
                    elif source_type == 'expansion':
                        # Fallback expansion tooltip
                        origin_html = f'<span class="origin-tooltip">Expansion: from parent context mutual expansion</span>'

                    source_class = f'sub-{source_type}'
                    result += f'<div class="sub-item">'
                    result += f'<span class="context-label">L:</span> <span class="left-context">{left_ctx_str}</span> | '
                    result += f'<span class="substitute-text {source_class}">{sub_text}{origin_html}</span> '
                    result += f'<span class="score">(score: {score:.4f})</span> | '
                    result += f'<span class="context-label">R:</span> <span class="right-context">{right_ctx_str}</span>'
                    result += f'</div>'
        else:
            result += '<em>No substitutes</em>'
        result += '</div></div>'
        return result

    # Simple tree renderer - no cache scans, just uses the split info!
    def render_simple_tree(span_text, split, depth=0):
        indent_class = f"indent-{min(depth, 5)}"

        # Get substitutes and energy for THIS specific split
        cache_key = (span_text, split)
        if cache_key in subparse_cache:
            energy = subparse_cache[cache_key][3]
            # Get substitutes directly from this cache entry (no merging, no filtering needed)
            merged_subs = get_substitutes_for_split(subparse_cache, span_text, split)
            info = f'E={energy:.2f}, {len(merged_subs)} subs'
        else:
            energy = None
            merged_subs = []
            info = ''

        # Create unique ID for this node's substitutes
        span_id = make_span_id(span_text) + f'-alt-{depth}-{next_node_id()}'

        result = f'<div class="{indent_class}">'
        result += f'<span class="span-text" onclick="toggleNode(\'{span_id}\')" style="cursor: pointer;">{span_text}</span>'
        if info:
            result += f' <span class="energy">{info}</span>'
        # DEBUG: Show ID being used
        result += f' <span style="color: #999; font-size: 0.7em;">[id: {span_id}]</span>'

        # Add substitutes section (hidden by default)
        result += f'<div class="substitutes" id="{span_id}">'

        if merged_subs:
            # Group substitutes by word count
            by_length = {}
            for sub_text, score, source_type, sub_origins in merged_subs:
                word_count = sub_text.count(' ') + 1
                if word_count not in by_length:
                    by_length[word_count] = []
                by_length[word_count].append((sub_text, score, source_type, sub_origins))

            # Sort each group by score
            for length in by_length:
                by_length[length].sort(key=lambda x: -x[1])

            # Build header with length breakdown
            result += f'<strong>Substitutes ({len(merged_subs)} total): '
            result += f'<span class="sub-aggregation">aggregation from this split</span></strong><br>'
            result += '<span class="length-breakdown">'
            for length in sorted(by_length.keys()):
                count = len(by_length[length])
                result += f'{length}-word: <strong>{count}</strong> &nbsp; '
            result += '</span><br><br>'

            # Display substitutes grouped by length
            for length in sorted(by_length.keys(), reverse=True):  # Show longest first
                subs_at_length = by_length[length]
                length_id = f'{span_id}-len{length}'

                # Collapsible section for this length
                result += f'<div class="length-group">'
                result += f'<strong onclick="toggleNode(\'{length_id}\')" style="cursor: pointer; color: #2196F3;">'
                result += f'▸ {length}-word substitutes ({len(subs_at_length)})</strong>'
                result += f'<div class="substitutes-by-length" id="{length_id}" style="display: none; margin-left: 20px;">'

                for sub_text, score, source_type, sub_origins in subs_at_length:
                    sub_pattern = get_unit_with_fallback(sub_text, catalog, corpus_index)
                    if sub_pattern:
                        left_ctx = [w for w, c in sub_pattern.left_words.most_common(5)]
                        right_ctx = [w for w, c in sub_pattern.right_words.most_common(5)]
                        left_ctx_str = ', '.join(left_ctx) if left_ctx else '(none)'
                        right_ctx_str = ', '.join(right_ctx) if right_ctx else '(none)'

                        # Build origin tooltip
                        origin_html = ''
                        if sub_origins:
                            if 'left_source' in sub_origins:
                                # Detailed aggregation: show combination origin
                                left_path = f'"{sub_origins["left_source"]}" → "{sub_origins["left_sub"]}" ({sub_origins["left_score"]:.2f})'
                                right_path = f'"{sub_origins["right_source"]}" → "{sub_origins["right_sub"]}" ({sub_origins["right_score"]:.2f})'
                                # Check if this used word-level substitutions
                                if sub_origins.get('subparse_combo'):
                                    origin_html = f'<span class="origin-tooltip">Word-level combination from {left_path} + {right_path} → "{sub_text}"</span>'
                                else:
                                    origin_html = f'<span class="origin-tooltip">Aggregation: {left_path} + {right_path}</span>'
                            elif sub_origins.get('split_type') == 'expansion':
                                # Expansion: show parent split
                                parent = sub_origins.get('parent_split', 'unknown')
                                origin_html = f'<span class="origin-tooltip">Expansion: from parent context {parent}</span>'
                            elif sub_origins.get('split_type') == 'aggregation':
                                # Simple aggregation: show the split
                                parent = sub_origins.get('parent_split', 'unknown')
                                origin_html = f'<span class="origin-tooltip">Aggregation: from constituent combination {parent}</span>'
                        elif source_type == 'expansion':
                            # Fallback expansion tooltip
                            origin_html = f'<span class="origin-tooltip">Expansion: from parent context mutual expansion</span>'

                        source_class = f'sub-{source_type}'
                        result += f'<div class="sub-item">'
                        result += f'<span class="context-label">L:</span> <span class="left-context">{left_ctx_str}</span> | '
                        result += f'<span class="substitute-text {source_class}">{sub_text}{origin_html}</span> '
                        result += f'<span class="score">(score: {score:.4f})</span> | '
                        result += f'<span class="context-label">R:</span> <span class="right-context">{right_ctx_str}</span>'
                        result += f'</div>'

                # Close this length group
                result += f'</div></div>'  # Close substitutes-by-length and length-group

        else:
            result += '<em>No substitutes</em>'
        result += '</div>'

        # Recurse on children (split tells us what they are!)
        if split and isinstance(split, tuple) and len(split) == 2 and depth < 10:
            if split[0] in ['aggregation', 'expansion']:
                tokens = split[1]
                if len(tokens) == 2:
                    left_text, right_text = tokens
                else:
                    result += f' <em>[{len(tokens)} tokens]</em></div>'
                    return result
            else:
                left_text, right_text = split

            # Recurse - for multi-word children, use their WINNING split
            if ' ' in left_text:
                left_split = find_best_split(left_text)
                result += render_simple_tree(left_text, left_split, depth + 1)
            else:
                # Left child: substitutes come from left_words of right sibling
                result += render_leaf(left_text, depth + 1, context_side='left', context_word=right_text)

            if ' ' in right_text:
                right_split = find_best_split(right_text)
                result += render_simple_tree(right_text, right_split, depth + 1)
            else:
                # Right child: substitutes come from right_words of left sibling
                result += render_leaf(right_text, depth + 1, context_side='right', context_word=left_text)

        result += '</div>'
        return result

    # Render alternative trees in processing order
    html += '<h2 style="margin-top: 40px;">Alternative Parse Trees (incremental processing order)</h2>'

    all_parses = [(span_length, span_text, split, energy)
                  for span_text, split, energy, span_length in parse_buffer]
    all_parses.sort(key=lambda x: (x[0], x[3]))  # Sort by length, then energy

    current_length = 0
    for span_length, span_text, split, energy in all_parses:
        if span_length < 2:
            continue

        if span_length != current_length:
            if current_length > 0:
                html += '</div>'
            html += f'<h3 style="margin-top: 20px; color: #666;">{span_length}-word spans</h3>'
            html += '<div style="margin-left: 20px;">'
            current_length = span_length

        # Show in bracket notation: (my (hat again))
        bracket_repr = render_as_brackets(span_text, split)
        html += f'<div class="tree" style="margin: 10px 0; padding: 10px; background: #fafafa; border-left: 3px solid #2196F3;">'
        html += f'<strong style="font-family: monospace; font-size: 1.1em;">{bracket_repr}</strong>'

        # Get energy info
        cache_key = (span_text, split)
        if cache_key in subparse_cache:
            energy = subparse_cache[cache_key][3]
            num_subs = len(subparse_cache[cache_key][0])
            html += f' <span class="energy">E={energy:.2f}, {num_subs} subs</span>'
        html += '<br>'

        html += render_simple_tree(span_text, split, depth=0)
        html += '</div>'

    if current_length > 0:
        html += '</div>'

    html += """
    </div>
    </div><!-- end tree-view tab -->
"""

    # Build graph data from the winning parse tree
    import json as _json

    graph_nodes = []
    graph_edges = []
    node_set = set()

    def add_graph_node(span_text, depth, node_type='span'):
        if span_text in node_set:
            return
        node_set.add(span_text)

        subs, left_ctx, right_ctx, energy, split, origins = get_best_parse(subparse_cache, span_text)
        num_subs = len(subs) if subs else 0
        word_count = len(span_text.split())

        # Get top contexts
        top_left = left_ctx[:5] if left_ctx else []
        top_right = right_ctx[:5] if right_ctx else []

        graph_nodes.append({
            'id': span_text,
            'label': span_text,
            'type': node_type,
            'energy': round(energy, 2) if energy != float('inf') else None,
            'num_subs': num_subs,
            'depth': depth,
            'word_count': word_count,
            'left_ctx': top_left,
            'right_ctx': top_right,
        })

        # Add top substitutes grouped by word length
        if subs:
            from collections import defaultdict
            by_len = defaultdict(list)
            for s_text, s_score in subs:
                by_len[len(s_text.split())].append((s_text, s_score))
            for wlen in sorted(by_len.keys()):
                top_for_len = sorted(by_len[wlen], key=lambda x: -x[1])[:5]
                for s_text, s_score in top_for_len:
                    sub_id = f"sub:{span_text}:{s_text}"
                    if sub_id not in node_set:
                        node_set.add(sub_id)
                        graph_nodes.append({
                            'id': sub_id,
                            'label': s_text,
                            'type': 'substitute',
                            'score': round(s_score, 3),
                            'parent': span_text,
                            'depth': depth,
                            'word_count': len(s_text.split()),
                        })
                        graph_edges.append({
                            'source': span_text,
                            'target': sub_id,
                            'type': 'substitution',
                        })

        # Recurse into children
        if split and isinstance(split, tuple) and len(split) == 2:
            if split[0] not in ['aggregation', 'expansion']:
                left_text, right_text = split
                add_graph_node(left_text, depth + 1, 'span' if ' ' in left_text else 'leaf')
                add_graph_node(right_text, depth + 1, 'span' if ' ' in right_text else 'leaf')
                graph_edges.append({'source': span_text, 'target': left_text, 'type': 'split'})
                graph_edges.append({'source': span_text, 'target': right_text, 'type': 'split'})

    # Walk from root
    root_text = phrase.lower()
    add_graph_node(root_text, 0)

    graph_data = _json.dumps({'nodes': graph_nodes, 'edges': graph_edges})


    html += f"""
	<div id="graph-view" class="tab-panel">
		<div style="margin-bottom:10px;padding:5px;background:#fafafa;border-radius:6px;font-size:13px;line-height:1.2;">
			<strong style="color:#333;">Legend: </strong>
			<span style="margin-right:8px;"><svg width="12" height="12" style="vertical-align:middle;"><circle cx="6" cy="6" r="5" fill="#4CAF50" stroke="#333" stroke-width="1.5"/></svg> Low energy (good)</span>
			<span style="margin-right:8px;"><svg width="12" height="12" style="vertical-align:middle;"><circle cx="6" cy="6" r="5" fill="#FF5722" stroke="#333" stroke-width="1.5"/></svg> High energy (poor)</span>
			<span style="margin-right:8px;"><svg width="10" height="10" style="vertical-align:middle;"><circle cx="5" cy="5" r="4" fill="#E8F5E9" stroke="#388E3C" stroke-width="1"/></svg> Leaf</span>
			<span style="margin-right:8px;"><svg width="10" height="10" style="vertical-align:middle;"><circle cx="5" cy="5" r="4" fill="#B3E5FC" stroke="#0288D1" stroke-width="1"/></svg> Substitute (click span to show)</span>
			<span style="margin-right:8px;"><svg width="20" height="10" style="vertical-align:middle;"><line x1="0" y1="5" x2="20" y2="5" stroke="#333" stroke-width="2.5"/></svg> Split</span>
			<span style="margin-right:8px;"><svg width="20" height="10" style="vertical-align:middle;"><line x1="0" y1="5" x2="20" y2="5" stroke="#bbb" stroke-width="1" stroke-dasharray="4,3"/></svg> Substitution</span>
			Node size = substitute count
		</div>
		<svg id="graph-svg"></svg>
		<div id="graph-tooltip" class="graph-tooltip" style="display:none;"></div>
	</div>



    <script>
    const graphData = {graph_data};

    function initGraph() {{
        const svg = d3.select('#graph-svg');
        const width = svg.node().getBoundingClientRect().width || 1200;
        const height = 700;
        svg.attr('viewBox', [0, 0, width, height]);

        const tooltip = d3.select('#graph-tooltip');

        // Separate span/leaf nodes from substitute nodes
        const spanNodes = graphData.nodes.filter(n => n.type !== 'substitute');
        const subNodes = graphData.nodes.filter(n => n.type === 'substitute');

        // Initially hide substitutes
        subNodes.forEach(n => n.hidden = true);

        // Color scale for energy (green=low/good, red=high/bad)
        const energies = spanNodes.map(n => n.energy).filter(e => e !== null);
        const eMin = d3.min(energies) || -15;
        const eMax = d3.max(energies) || 0;
        const colorScale = d3.scaleLinear()
            .domain([eMin, (eMin + eMax) / 2, eMax])
            .range(['#4CAF50', '#FFC107', '#FF5722']);

        // Size scale for nodes
        const sizeScale = d3.scaleSqrt()
            .domain([0, d3.max(spanNodes, n => n.num_subs) || 100])
            .range([15, 45]);

        // Build links with references
        const allLinks = graphData.edges.map(e => ({{...e}}));

        // Force simulation
        const simulation = d3.forceSimulation(graphData.nodes)
            .force('link', d3.forceLink(allLinks).id(d => d.id)
                .distance(d => d.type === 'split' ? 120 : 60)
                .strength(d => d.type === 'split' ? 1.0 : 0.3))
            .force('charge', d3.forceManyBody().strength(d => d.type === 'substitute' ? -30 : -200))
            .force('y', d3.forceY(d => {{
                if (d.type === 'substitute') return height * 0.3 + (d.depth || 0) * 100;
                return 80 + (d.depth || 0) * 130;
            }}).strength(0.3))
            .force('x', d3.forceX(width / 2).strength(0.05))
            .force('collide', d3.forceCollide(d => d.type === 'substitute' ? 8 : sizeScale(d.num_subs || 0) + 5));

        // Arrow markers
        svg.append('defs').selectAll('marker')
            .data(['split', 'substitution'])
            .join('marker')
            .attr('id', d => 'arrow-' + d)
            .attr('viewBox', '0 -5 10 10')
            .attr('refX', 15).attr('refY', 0)
            .attr('markerWidth', 6).attr('markerHeight', 6)
            .attr('orient', 'auto')
            .append('path')
            .attr('d', 'M0,-5L10,0L0,5')
            .attr('fill', d => d === 'split' ? '#333' : '#aaa');

        // Container for zoom
        const g = svg.append('g');

        svg.call(d3.zoom()
            .scaleExtent([0.3, 4])
            .on('zoom', e => g.attr('transform', e.transform)));

        // Links
        const link = g.append('g').selectAll('line')
            .data(allLinks)
            .join('line')
            .attr('stroke', d => d.type === 'split' ? '#333' : '#bbb')
            .attr('stroke-width', d => d.type === 'split' ? 2.5 : 1)
            .attr('stroke-dasharray', d => d.type === 'substitution' ? '4,3' : null)
            .attr('marker-end', d => 'url(#arrow-' + d.type + ')');

        // Nodes
        const node = g.append('g').selectAll('g')
            .data(graphData.nodes)
            .join('g')
            .attr('class', 'graph-node')
            .style('display', d => d.hidden ? 'none' : null)
            .call(d3.drag()
                .on('start', (e, d) => {{ if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }})
                .on('drag', (e, d) => {{ d.fx = e.x; d.fy = e.y; }})
                .on('end', (e, d) => {{ if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }}));

        // Circles
        node.append('circle')
            .attr('r', d => {{
                if (d.type === 'substitute') return 6;
                return sizeScale(d.num_subs || 0);
            }})
            .attr('fill', d => {{
                if (d.type === 'substitute') return '#B3E5FC';
                if (d.type === 'leaf') return '#E8F5E9';
                return d.energy !== null ? colorScale(d.energy) : '#ccc';
            }})
            .attr('stroke', d => {{
                if (d.type === 'substitute') return '#0288D1';
                if (d.type === 'leaf') return '#388E3C';
                return '#333';
            }})
            .attr('stroke-width', d => d.type === 'substitute' ? 1 : 2);

        // Labels
        node.append('text')
            .attr('class', 'graph-node-label')
            .attr('dy', d => d.type === 'substitute' ? -10 : -(sizeScale(d.num_subs || 0) + 5))
            .attr('text-anchor', 'middle')
            .text(d => d.label)
            .style('font-size', d => d.type === 'substitute' ? '10px' : '13px')
            .style('font-weight', d => d.type === 'substitute' ? 'normal' : 'bold');

        // Energy labels inside span nodes
        node.filter(d => d.type === 'span' && d.energy !== null)
            .append('text')
            .attr('text-anchor', 'middle')
            .attr('dy', 4)
            .style('font-size', '10px')
            .style('fill', 'white')
            .style('font-weight', 'bold')
            .style('pointer-events', 'none')
            .text(d => 'E=' + d.energy);

        // Hover tooltips
        node.on('mouseover', (e, d) => {{
            let html = '<strong>' + d.label + '</strong><br>';
            if (d.type === 'substitute') {{
                html += 'Score: ' + (d.score || 'N/A') + '<br>';
                html += 'Words: ' + d.word_count;
            }} else {{
                if (d.energy !== null) html += 'Energy: ' + d.energy + '<br>';
                html += 'Substitutes: ' + (d.num_subs || 0) + '<br>';
                html += 'Words: ' + d.word_count;
                if (d.left_ctx && d.left_ctx.length) html += '<br>L: ' + d.left_ctx.join(', ');
                if (d.right_ctx && d.right_ctx.length) html += '<br>R: ' + d.right_ctx.join(', ');
            }}
            tooltip.html(html)
                .style('display', 'block')
                .style('left', (e.pageX + 15) + 'px')
                .style('top', (e.pageY - 10) + 'px');
        }})
        .on('mouseout', () => tooltip.style('display', 'none'));

        // Click span nodes to toggle substitute visibility
        node.filter(d => d.type !== 'substitute').on('click', (e, d) => {{
            const subs = graphData.nodes.filter(n => n.type === 'substitute' && n.parent === d.id);
            const subIds = new Set(subs.map(n => n.id));
            const showing = !subs[0]?.hidden;
            subs.forEach(n => n.hidden = showing);

            node.filter(nd => subIds.has(nd.id))
                .style('display', nd => nd.hidden ? 'none' : null);
            link.style('display', l => {{
                const sid = typeof l.source === 'object' ? l.source.id : l.source;
                const tid = typeof l.target === 'object' ? l.target.id : l.target;
                const sn = graphData.nodes.find(n => n.id === sid);
                const tn = graphData.nodes.find(n => n.id === tid);
                if ((sn && sn.hidden) || (tn && tn.hidden)) return 'none';
                return null;
            }});

            simulation.alpha(0.3).restart();
        }});

        // Tick
        simulation.on('tick', () => {{
            link
                .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
            node.attr('transform', d => 'translate(' + d.x + ',' + d.y + ')');
        }});
    }}
    </script>
    </div>
"""

    # Build corpus view data
    corpus_words = phrase.lower().split()
    corpus_arcs = []
    corpus_subs = []

    # Assign colors to parse groups
    arc_colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336', '#00BCD4', '#795548', '#607D8B']

    def collect_arcs(span_text, depth, color_idx):
        """Recursively collect arcs from winning parse tree."""
        words_in_span = span_text.split()
        if len(words_in_span) < 2:
            return color_idx

        # Find start position in the input phrase
        span_start = None
        for i in range(len(corpus_words) - len(words_in_span) + 1):
            if corpus_words[i:i+len(words_in_span)] == words_in_span:
                span_start = i
                break
        if span_start is None:
            return color_idx

        subs, left_ctx, right_ctx, energy, split, origins = get_best_parse(subparse_cache, span_text)
        color = arc_colors[color_idx % len(arc_colors)]

        corpus_arcs.append({
            'start': span_start,
            'end': span_start + len(words_in_span) - 1,
            'energy': round(energy, 2) if energy != float('inf') else None,
            'label': span_text,
            'depth': depth,
            'color': color,
        })

        # Get top substitutes — prefer full-length, then sample other lengths
        if subs:
            from collections import defaultdict as _defaultdict
            by_len = _defaultdict(list)
            for s_text, s_score in subs:
                by_len[len(s_text.split())].append((s_text, s_score))

            picked = []
            # Full-length first
            full_len = len(words_in_span)
            if full_len in by_len:
                picked.extend(sorted(by_len[full_len], key=lambda x: -x[1])[:3])
            # Then other lengths
            for wlen in sorted(by_len.keys()):
                if wlen != full_len and len(picked) < 5:
                    picked.extend(sorted(by_len[wlen], key=lambda x: -x[1])[:2])

            for s_text, s_score in picked[:5]:
                # Try to find corpus context for this substitute
                left_context_words = []
                right_context_words = []
                if corpus_index and s_text in corpus_index.ngram_index:
                    positions = corpus_index.ngram_index[s_text]
                    if positions:
                        pos = positions[0]  # Take first occurrence
                        s_len = len(s_text.split())
                        # Extract context window (3 words each side)
                        lstart = max(0, pos - 3)
                        left_context_words = [t for t in corpus_index.tokens[lstart:pos]
                                              if not t.startswith('__')]
                        rend = min(len(corpus_index.tokens), pos + s_len + 3)
                        right_context_words = [t for t in corpus_index.tokens[pos + s_len:rend]
                                               if not t.startswith('__')]
                elif s_text in (catalog.units if hasattr(catalog, 'units') else {}):
                    pattern = catalog.get_unit(s_text)
                    if pattern:
                        left_context_words = [w for w, c in pattern.left_words.most_common(3)]
                        right_context_words = [w for w, c in pattern.right_words.most_common(3)]

                corpus_subs.append({
                    'span': span_text,
                    'span_start': span_start,
                    'span_end': span_start + len(words_in_span) - 1,
                    'text': s_text,
                    'words': s_text.split(),
                    'score': round(s_score, 3),
                    'left_ctx': left_context_words[:3],
                    'right_ctx': right_context_words[:3],
                    'color': color,
                })

        next_color = color_idx + 1

        # Recurse into children
        if split and isinstance(split, tuple) and len(split) == 2:
            if split[0] not in ['aggregation', 'expansion']:
                left_text, right_text = split
                if ' ' in left_text:
                    next_color = collect_arcs(left_text, depth + 1, next_color)
                if ' ' in right_text:
                    next_color = collect_arcs(right_text, depth + 1, next_color)

        return next_color

    collect_arcs(phrase.lower(), 0, 0)

    corpus_view_data = _json.dumps({
        'words': corpus_words,
        'arcs': corpus_arcs,
        'substitutes': corpus_subs,
    })

    html += f"""
    <div id="corpus-view" class="tab-panel">
        <svg id="corpus-svg"></svg>
    </div>
    <script>
    const corpusData = {corpus_view_data};

    function initCorpusView() {{
        const svg = d3.select('#corpus-svg');
        const container = svg.node().parentNode;
        const totalWidth = container.getBoundingClientRect().width - 40 || 1000;

        const wordBoxW = 90;
        const wordBoxH = 32;
        const wordGap = 6;
        const phraseWidth = corpusData.words.length * (wordBoxW + wordGap) - wordGap;
        const phraseX = (totalWidth - phraseWidth) / 2;

        // Calculate heights
        const maxArcDepth = corpusData.arcs.length > 0 ? d3.max(corpusData.arcs, d => d.depth) : 0;
        const arcAreaH = (maxArcDepth + 1) * 40 + 30;
        const phraseY = arcAreaH + 10;
        const subStartY = phraseY + wordBoxH + 30;
        const subRowH = 36;
        const totalSubs = corpusData.substitutes.length;
        const totalH = subStartY + totalSubs * subRowH + 40;

        svg.attr('viewBox', '0 0 ' + totalWidth + ' ' + totalH)
           .attr('height', totalH);

        // Draw arcs above the phrase
        const arcGroup = svg.append('g');
        corpusData.arcs.forEach((arc, i) => {{
            const x1 = phraseX + arc.start * (wordBoxW + wordGap) + wordBoxW / 2;
            const x2 = phraseX + arc.end * (wordBoxW + wordGap) + wordBoxW / 2;
            const arcH = arcAreaH - arc.depth * 40 - 10;
            const midX = (x1 + x2) / 2;

            // Draw arc path
            arcGroup.append('path')
                .attr('d', 'M ' + x1 + ' ' + phraseY + ' L ' + x1 + ' ' + arcH +
                           ' L ' + x2 + ' ' + arcH + ' L ' + x2 + ' ' + phraseY)
                .attr('fill', 'none')
                .attr('stroke', arc.color)
                .attr('stroke-width', 2)
                .attr('opacity', 0.8);

            // Energy label on the arc
            if (arc.energy !== null) {{
                arcGroup.append('text')
                    .attr('x', midX)
                    .attr('y', arcH - 4)
                    .attr('text-anchor', 'middle')
                    .attr('font-size', '11px')
                    .attr('fill', arc.color)
                    .attr('font-weight', 'bold')
                    .text('E=' + arc.energy);
            }}
        }});

        // Draw input phrase word boxes
        const phraseGroup = svg.append('g');
        corpusData.words.forEach((word, i) => {{
            const x = phraseX + i * (wordBoxW + wordGap);

            phraseGroup.append('rect')
                .attr('x', x).attr('y', phraseY)
                .attr('width', wordBoxW).attr('height', wordBoxH)
                .attr('rx', 4)
                .attr('fill', '#E3F2FD')
                .attr('stroke', '#1565C0')
                .attr('stroke-width', 1.5);

            phraseGroup.append('text')
                .attr('x', x + wordBoxW / 2).attr('y', phraseY + wordBoxH / 2 + 5)
                .attr('text-anchor', 'middle')
                .attr('font-size', '14px')
                .attr('font-weight', 'bold')
                .attr('fill', '#1565C0')
                .text(word);
        }});

        // Draw substitute rows below
        const subGroup = svg.append('g');
        const subWordW = 70;
        const subWordH = 24;
        const subWordGap = 3;
        const ctxWordW = 55;

        corpusData.substitutes.forEach((sub, si) => {{
            const rowY = subStartY + si * subRowH;
            const subWords = sub.words;
            const leftCtx = sub.left_ctx || [];
            const rightCtx = sub.right_ctx || [];

            // Total width of this row
            const ctxLeftW = leftCtx.length * (ctxWordW + subWordGap);
            const subPhraseW = subWords.length * (subWordW + subWordGap) - subWordGap;
            const ctxRightW = rightCtx.length * (ctxWordW + subWordGap);
            const rowW = ctxLeftW + subPhraseW + ctxRightW + 20;
            const rowX = (totalWidth - rowW) / 2;

            let curX = rowX;

            // Left context (grey)
            leftCtx.forEach(w => {{
                subGroup.append('text')
                    .attr('x', curX + ctxWordW / 2).attr('y', rowY + subWordH / 2 + 4)
                    .attr('text-anchor', 'middle')
                    .attr('font-size', '11px')
                    .attr('fill', '#999')
                    .text(w);
                curX += ctxWordW + subWordGap;
            }});

            // Separator
            if (leftCtx.length > 0) {{
                subGroup.append('text')
                    .attr('x', curX).attr('y', rowY + subWordH / 2 + 4)
                    .attr('font-size', '11px').attr('fill', '#ccc')
                    .text('|');
                curX += 10;
            }}

            // Substitute word boxes
            const subStartX = curX;
            subWords.forEach((w, wi) => {{
                subGroup.append('rect')
                    .attr('x', curX).attr('y', rowY)
                    .attr('width', subWordW).attr('height', subWordH)
                    .attr('rx', 3)
                    .attr('fill', sub.color + '22')
                    .attr('stroke', sub.color)
                    .attr('stroke-width', 1);

                subGroup.append('text')
                    .attr('x', curX + subWordW / 2).attr('y', rowY + subWordH / 2 + 4)
                    .attr('text-anchor', 'middle')
                    .attr('font-size', '11px')
                    .attr('font-weight', 'bold')
                    .attr('fill', sub.color)
                    .text(w);
                curX += subWordW + subWordGap;
            }});
            const subEndX = curX - subWordGap;

            // Separator
            if (rightCtx.length > 0) {{
                curX += 4;
                subGroup.append('text')
                    .attr('x', curX).attr('y', rowY + subWordH / 2 + 4)
                    .attr('font-size', '11px').attr('fill', '#ccc')
                    .text('|');
                curX += 10;
            }}

            // Right context (grey)
            rightCtx.forEach(w => {{
                subGroup.append('text')
                    .attr('x', curX + ctxWordW / 2).attr('y', rowY + subWordH / 2 + 4)
                    .attr('text-anchor', 'middle')
                    .attr('font-size', '11px')
                    .attr('fill', '#999')
                    .text(w);
                curX += ctxWordW + subWordGap;
            }});

            // Score label
            subGroup.append('text')
                .attr('x', curX + 10).attr('y', rowY + subWordH / 2 + 4)
                .attr('font-size', '10px')
                .attr('fill', '#999')
                .text('(' + sub.score + ')');

            // Draw connecting line from span arc down to substitute row
            const spanMidX = phraseX + ((sub.span_start + sub.span_end) / 2) * (wordBoxW + wordGap) + wordBoxW / 2;
            const subMidX = (subStartX + subEndX) / 2;

            subGroup.append('path')
                .attr('d', 'M ' + spanMidX + ' ' + (phraseY + wordBoxH) +
                           ' C ' + spanMidX + ' ' + (rowY - 5) +
                           ' ' + subMidX + ' ' + (phraseY + wordBoxH + 15) +
                           ' ' + subMidX + ' ' + rowY)
                .attr('fill', 'none')
                .attr('stroke', sub.color)
                .attr('stroke-width', 1)
                .attr('stroke-dasharray', '4,3')
                .attr('opacity', 0.5);
        }});
    }}
    </script>
"""

    # Build containment view data
    _color_idx = [0]

    def build_containment_hierarchy(span_text, depth=0):
        """Build a hierarchy dict for D3 circle packing."""
        words = span_text.split()
        color = arc_colors[_color_idx[0] % len(arc_colors)]

        subs, left_ctx, right_ctx, energy, split, origins = get_best_parse(subparse_cache, span_text)
        num_subs = len(subs) if subs else 0

        node = {
            'id': span_text,
            'label': span_text,
            'energy': round(energy, 2) if energy != float('inf') else None,
            'num_subs': num_subs,
            'color': color,
            'type': 'span',
            'children': [],
        }

        if len(words) < 2 or not split:
            # Leaf word — no children, just a value for packing
            node['type'] = 'word'
            node['value'] = 10
            return node

        _color_idx[0] += 1

        # Add child spans / leaf words
        if split and isinstance(split, tuple) and len(split) == 2 and split[0] not in ['aggregation', 'expansion']:
            left_text, right_text = split
            if ' ' in left_text:
                node['children'].append(build_containment_hierarchy(left_text, depth + 1))
            else:
                node['children'].append({
                    'id': f'word:{left_text}:{span_text}',
                    'label': left_text,
                    'type': 'word',
                    'color': '#1565C0',
                    'value': 10,
                })
            if ' ' in right_text:
                node['children'].append(build_containment_hierarchy(right_text, depth + 1))
            else:
                node['children'].append({
                    'id': f'word:{right_text}:{span_text}',
                    'label': right_text,
                    'type': 'word',
                    'color': '#1565C0',
                    'value': 10,
                })

        # Add top substitutes
        if subs:
            from collections import defaultdict as _dl
            by_len = _dl(list)
            for s_text, s_score in subs:
                by_len[len(s_text.split())].append((s_text, s_score))
            full_len = len(words)
            picked = []
            if full_len in by_len:
                picked.extend(sorted(by_len[full_len], key=lambda x: -x[1])[:3])
            for wlen in sorted(by_len.keys()):
                if wlen != full_len and len(picked) < 5:
                    picked.extend(sorted(by_len[wlen], key=lambda x: -x[1])[:1])

            for s_text, s_score in picked[:5]:
                node['children'].append({
                    'id': f'sub:{span_text}:{s_text}',
                    'label': s_text,
                    'type': 'substitute',
                    'score': round(s_score, 3),
                    'color': color,
                    'value': 6,
                })

        return node

    containment_hierarchy = build_containment_hierarchy(phrase.lower())

    # Build sequences
    containment_sequences = []

    # Input phrase sequence — map words to their node IDs in the hierarchy
    def find_word_ids(node, word_list, idx=0):
        """Find node IDs for sequential input words in the hierarchy."""
        ids = {}
        def walk(n):
            if n.get('type') == 'word' and n['label'] in word_list:
                # Match to earliest unmatched position
                for i, w in enumerate(word_list):
                    if w == n['label'] and i not in ids:
                        ids[i] = n['id']
                        break
            for c in n.get('children', []):
                walk(c)
        walk(node)
        return [ids.get(i, f'word:{w}') for i, w in enumerate(word_list)]

    input_word_ids = find_word_ids(containment_hierarchy, corpus_words)
    containment_sequences.append({
        'words': corpus_words,
        'node_ids': input_word_ids,
        'type': 'input',
        'color': '#1565C0',
    })

    # Substitute sequences with corpus context
    def collect_sub_sequences(node):
        if node.get('type') != 'span':
            return
        color = node.get('color', '#666')
        for child in node.get('children', []):
            if child.get('type') == 'substitute':
                s_text = child['label']
                left_ctx_words = []
                right_ctx_words = []
                if corpus_index and s_text in corpus_index.ngram_index:
                    positions = corpus_index.ngram_index[s_text]
                    if positions:
                        pos = positions[0]
                        s_len = len(s_text.split())
                        lstart = max(0, pos - 3)
                        left_ctx_words = [t for t in corpus_index.tokens[lstart:pos]
                                          if not t.startswith('__')]
                        rend = min(len(corpus_index.tokens), pos + s_len + 3)
                        right_ctx_words = [t for t in corpus_index.tokens[pos + s_len:rend]
                                           if not t.startswith('__')]
                elif hasattr(catalog, 'units') and s_text in catalog.units:
                    pattern = catalog.get_unit(s_text)
                    if pattern:
                        left_ctx_words = [w for w, c in pattern.left_words.most_common(2)]
                        right_ctx_words = [w for w, c in pattern.right_words.most_common(2)]

                containment_sequences.append({
                    'words': s_text.split(),
                    'node_id': child['id'],
                    'parent_span': node['id'],
                    'type': 'substitute',
                    'color': color,
                    'left_ctx': left_ctx_words[:2],
                    'right_ctx': right_ctx_words[:2],
                })
            elif child.get('type') == 'span':
                collect_sub_sequences(child)

    collect_sub_sequences(containment_hierarchy)

    containment_data = _json.dumps({
        'hierarchy': containment_hierarchy,
        'sequences': containment_sequences,
    })

    html += f"""
    <div id="containment-view" class="tab-panel">
        <svg id="containment-svg"></svg>
        <div id="containment-tooltip" class="graph-tooltip" style="display:none;"></div>
    </div>
    <script>
    const containmentData = {containment_data};

    function initContainmentView() {{
        const svg = d3.select('#containment-svg');
        const width = svg.node().parentNode.getBoundingClientRect().width - 20 || 1200;
        const height = 800;
        svg.attr('viewBox', [0, 0, width, height]);

        const tooltip = d3.select('#containment-tooltip');

        // Build D3 hierarchy and pack layout
        const root = d3.hierarchy(containmentData.hierarchy)
            .sum(d => d.value || 0)
            .sort((a, b) => (b.value || 0) - (a.value || 0));

        const pack = d3.pack()
            .size([width - 40, height - 120])
            .padding(d => d.depth === 0 ? 20 : 12);

        pack(root);

        // Offset to center
        const ox = 20;
        const oy = 60;

        const g = svg.append('g');

        // Zoom
        svg.call(d3.zoom()
            .scaleExtent([0.3, 4])
            .on('zoom', e => g.attr('transform', e.transform)));

        // Draw containment circles (spans only — those with children)
        const spanNodes = root.descendants().filter(d => d.data.type === 'span' && d.children);
        g.selectAll('.containment-circle')
            .data(spanNodes)
            .join('circle')
            .attr('class', 'containment-circle')
            .attr('cx', d => d.x + ox)
            .attr('cy', d => d.y + oy)
            .attr('r', d => d.r)
            .attr('fill', d => d.data.color + '08')
            .attr('stroke', d => d.data.color)
            .attr('stroke-width', 2)
            .attr('stroke-dasharray', d => d.depth === 0 ? null : '6,3');

        // Energy labels on containment circles
        g.selectAll('.energy-label')
            .data(spanNodes)
            .join('text')
            .attr('x', d => d.x + ox)
            .attr('y', d => d.y + oy - d.r + 14)
            .attr('text-anchor', 'middle')
            .attr('font-size', '11px')
            .attr('font-weight', 'bold')
            .attr('fill', d => d.data.color)
            .text(d => {{
                let t = d.data.label;
                if (t.length > 25) t = t.substring(0, 22) + '...';
                return t + (d.data.energy !== null ? ' (E=' + d.data.energy + ')' : '');
            }});

        // Draw leaf nodes (words and substitutes)
        const leafNodes = root.leaves();

        // Build ID→position map for sequence drawing
        const nodePositions = {{}};
        leafNodes.forEach(d => {{
            nodePositions[d.data.id] = {{ x: d.x + ox, y: d.y + oy }};
        }});

        // Word nodes
        const wordLeaves = leafNodes.filter(d => d.data.type === 'word');
        g.selectAll('.word-node')
            .data(wordLeaves)
            .join('circle')
            .attr('cx', d => d.x + ox)
            .attr('cy', d => d.y + oy)
            .attr('r', 14)
            .attr('fill', '#E3F2FD')
            .attr('stroke', '#1565C0')
            .attr('stroke-width', 2);

        g.selectAll('.word-label')
            .data(wordLeaves)
            .join('text')
            .attr('x', d => d.x + ox)
            .attr('y', d => d.y + oy + 4)
            .attr('text-anchor', 'middle')
            .attr('font-size', '12px')
            .attr('font-weight', 'bold')
            .attr('fill', '#1565C0')
            .text(d => d.data.label);

        // Substitute nodes
        const subLeaves = leafNodes.filter(d => d.data.type === 'substitute');
        g.selectAll('.sub-node')
            .data(subLeaves)
            .join('circle')
            .attr('cx', d => d.x + ox)
            .attr('cy', d => d.y + oy)
            .attr('r', 10)
            .attr('fill', d => d.data.color + '33')
            .attr('stroke', d => d.data.color)
            .attr('stroke-width', 1.5);

        g.selectAll('.sub-label')
            .data(subLeaves)
            .join('text')
            .attr('x', d => d.x + ox)
            .attr('y', d => d.y + oy + 18)
            .attr('text-anchor', 'middle')
            .attr('font-size', '9px')
            .attr('fill', d => d.data.color)
            .text(d => {{
                let t = d.data.label;
                if (t.length > 20) t = t.substring(0, 18) + '...';
                return t;
            }});

        // Draw sequence links
        const linkGen = d3.line().curve(d3.curveBasis);

        // Input phrase sequence
        const inputSeq = containmentData.sequences.find(s => s.type === 'input');
        if (inputSeq) {{
            const points = inputSeq.node_ids
                .map(id => nodePositions[id])
                .filter(p => p);
            if (points.length > 1) {{
                g.append('path')
                    .attr('d', linkGen(points.map(p => [p.x, p.y])))
                    .attr('fill', 'none')
                    .attr('stroke', '#1565C0')
                    .attr('stroke-width', 2.5)
                    .attr('opacity', 0.6);

                // Arrow indicators between consecutive words
                for (let i = 0; i < points.length - 1; i++) {{
                    const mx = (points[i].x + points[i+1].x) / 2;
                    const my = (points[i].y + points[i+1].y) / 2;
                    const angle = Math.atan2(points[i+1].y - points[i].y, points[i+1].x - points[i].x);
                    g.append('polygon')
                        .attr('points', '-4,-3 4,0 -4,3')
                        .attr('fill', '#1565C0')
                        .attr('opacity', 0.5)
                        .attr('transform', 'translate(' + mx + ',' + my + ') rotate(' + (angle * 180 / Math.PI) + ')');
                }}
            }}
        }}

        // Substitute sequences — draw from context through substitute node
        containmentData.sequences.filter(s => s.type === 'substitute').forEach(seq => {{
            const subPos = nodePositions[seq.node_id];
            if (!subPos) return;

            const allWords = (seq.left_ctx || []).concat(seq.words).concat(seq.right_ctx || []);
            const ctxLen = (seq.left_ctx || []).length;
            const subLen = seq.words.length;

            // Lay out sequence words horizontally centered on the substitute node
            const wordSpacing = 28;
            const totalW = (allWords.length - 1) * wordSpacing;
            const startX = subPos.x - totalW / 2;
            const seqY = subPos.y + 30;

            const seqPoints = allWords.map((w, i) => ({{
                x: startX + i * wordSpacing,
                y: seqY,
                word: w,
                isCtx: i < ctxLen || i >= ctxLen + subLen,
            }}));

            // Draw sequence line
            if (seqPoints.length > 1) {{
                g.append('path')
                    .attr('d', linkGen(seqPoints.map(p => [p.x, p.y])))
                    .attr('fill', 'none')
                    .attr('stroke', seq.color)
                    .attr('stroke-width', 1)
                    .attr('opacity', 0.4)
                    .attr('stroke-dasharray', '3,2');
            }}

            // Draw word labels along sequence
            seqPoints.forEach(p => {{
                g.append('text')
                    .attr('x', p.x).attr('y', p.y + 4)
                    .attr('text-anchor', 'middle')
                    .attr('font-size', '8px')
                    .attr('fill', p.isCtx ? '#aaa' : seq.color)
                    .attr('font-weight', p.isCtx ? 'normal' : 'bold')
                    .text(p.word);
            }});
        }});

        // Tooltips
        g.selectAll('circle').on('mouseover', function(e) {{
            const d = d3.select(this).datum();
            if (!d || !d.data) return;
            let html = '<strong>' + d.data.label + '</strong><br>';
            if (d.data.type === 'substitute') {{
                html += 'Score: ' + (d.data.score || '') + '<br>';
                html += 'Type: substitute';
            }} else if (d.data.type === 'span') {{
                if (d.data.energy !== null) html += 'Energy: ' + d.data.energy + '<br>';
                html += 'Substitutes: ' + (d.data.num_subs || 0);
            }} else {{
                html += 'Input word';
            }}
            tooltip.html(html)
                .style('display', 'block')
                .style('left', (e.pageX + 12) + 'px')
                .style('top', (e.pageY - 10) + 'px');
        }})
        .on('mouseout', () => tooltip.style('display', 'none'));
    }}
    </script>
</body>
</html>
"""

    with open(output_file, 'w') as f:
        f.write(html)

    print(f"\nInteractive parse tree saved to: {output_file}")
    print(f"Open it in your browser to explore the tree structure!")


def main():
    import argparse
    parser_args = argparse.ArgumentParser(
        description='Analyze a phrase: parse it and show contextual substitution classes.'
    )
    parser_args.add_argument('phrase', help='Phrase to analyze')
    parser_args.add_argument('--scoring', '-s',
                            choices=['cosine', 'ic_cosine', 'pmi'],
                            default='cosine',
                            help='Scoring method: cosine (default), ic_cosine, or pmi')
    parser_args.add_argument('--compare', '-c', action='store_true',
                            help='Compare all three scoring methods')
    args = parser_args.parse_args()

    # Load models
    print("Loading models...")
    catalog = UnitCatalog()
    catalog.load('unit_catalog.pkl')
    catalog.build_gpu_index(min_freq=1)  # Index all units, maximize information

    # Initialize corpus index for runtime lookup of longer phrases
    print("\nInitializing corpus index...")
    corpus_index = CorpusIndex('dialog_corpus.txt', max_ngram=7, context_window=1, cache_size=10000)

    parser = IncrementalBidirParser(catalog, debug=False)

    # Analyze
    if args.compare:
        for scoring in ['cosine', 'ic_cosine', 'pmi']:
            tree, cache, parse_buffer = analyze(args.phrase, catalog, parser, corpus_index, scoring=scoring)
            print("\n" + "=" * 70 + "\n")
        # Export HTML for last scoring method
        export_html_tree(args.phrase, tree, cache, catalog, parse_buffer, corpus_index, f"parse_tree_{args.phrase.replace(' ', '_')}.html")
    else:
        tree, cache, parse_buffer = analyze(args.phrase, catalog, parser, corpus_index, scoring=args.scoring)
        # Export HTML
        export_html_tree(args.phrase, tree, cache, catalog, parse_buffer, corpus_index, f"parse_tree_{args.phrase.replace(' ', '_')}.html")

    # Show corpus index usage statistics
    if corpus_index:
        corpus_index.get_usage_stats(top_n=20)

if __name__ == "__main__":
    main()
