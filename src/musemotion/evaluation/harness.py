"""The evaluation run: generate, score, compare, write one metrics file.

Every system in the comparison table is measured through the same code path, which is what
makes the columns readable straight down. The rows are:

* **real** - held-out EMOPIA test clips. The ceiling for every accuracy column, and the
  reference frame for novelty, diversity, and distributional fidelity.
* **guidance=x** - the generator conditioned on each quadrant in turn, swept over guidance
  scales. ``guidance=1.0`` is classifier-free guidance switched off.
* **unconditional** - sampled from the null emotion embedding. A floor: whatever the probe
  reports when nothing conditioned the music.
* **random_piano** - the deterministic baseline generator. A second, independent floor made
  of music no model produced.
* **end_to_end** - text through the classifier and then the generator, which is the system
  as it actually ships.

Nothing here retrains or reconfigures the pipeline. The generator and classifier checkpoints
are loaded and used exactly as committed.
"""
from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from musemotion.config import resolve_path
from musemotion.emotions import EMOPIA_QUADRANTS, quadrant_id, quadrant_name
from musemotion.evaluation import distributions as dist
from musemotion.evaluation.features import feature_matrix, symbolic_features
from musemotion.evaluation.novelty import (
    DEFAULT_DIVERSITY_SIZES,
    DEFAULT_NGRAM_SIZES,
    SYMBOL_VIEWS,
    CorpusIndex,
    build_corpus_index,
    set_diversity,
)
from musemotion.evaluation.probe import (
    DEFAULT_NOTE_BUDGET,
    crop_notes,
    load_clip_splits,
    load_probe_metadata,
    load_probes,
    probe_agreement,
)
from musemotion.evaluation.prompts import EVALUATION_PROMPTS, limited_prompts
from musemotion.evaluation.report import (
    comparison_row,
    comparison_table_csv,
    comparison_table_markdown,
    round_trip_result,
    stage_attribution,
    well_formedness,
)
from musemotion.music.tokenizer import MidiNote, MusicTokenizer


DEFAULT_GUIDANCE_SCALES: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0)


@dataclass
class GeneratedClip:
    """One generated clip plus the bookkeeping the metrics need."""

    emotion_id: int
    notes: list[MidiNote]
    token_ids: list[int]
    source_text: str | None = None
    classified_emotion_id: int | None = None
    # Carried per clip rather than assumed, so the EOS check follows the tokenizer that
    # produced the clip instead of relying on the special tokens keeping their current order.
    eos_token_id: int = 2

    @property
    def emitted_eos(self) -> bool:
        return bool(self.token_ids) and self.token_ids[-1] == self.eos_token_id

    def structure_record(self) -> dict[str, Any]:
        features = symbolic_features(self.notes)
        return {
            "emitted_eos": self.emitted_eos,
            "note_count": len(self.notes),
            "token_count": len(self.token_ids),
            "repeated_pitch_fraction": features["repeated_pitch_fraction"],
        }


def run_evaluation(config: dict[str, Any]) -> dict[str, Any]:
    """Execute a full evaluation run and return the metrics payload."""
    import torch

    from musemotion.inference.pipeline import MusicGeneratorComponent

    generation_config = config.get("generation", {})
    data_config = config.get("data", {})
    output_config = config.get("output", {})

    note_budget = int(data_config.get("note_budget", DEFAULT_NOTE_BUDGET))
    max_tokens = int(generation_config.get("max_tokens", note_budget * 4))
    clips_per_quadrant = int(generation_config.get("clips_per_quadrant", 50))
    batch_size = int(generation_config.get("batch_size", 25))
    temperature = float(generation_config.get("temperature", 1.0))
    top_k = generation_config.get("top_k", 32)
    seed = int(generation_config.get("seed", 1508))
    guidance_scales = [float(value) for value in generation_config.get("guidance_scales", DEFAULT_GUIDANCE_SCALES)]
    default_guidance = float(generation_config.get("guidance_scale", guidance_scales[-1] if guidance_scales else 3.0))

    tokenizer = MusicTokenizer.load(resolve_path(config["generator"]["tokenizer"]))
    reference = load_clip_splits(
        resolve_path(data_config.get("tokenized_dir", "artifacts/music/tokenized")),
        tokenizer,
        note_budget,
    )
    if not reference.get("train"):
        raise FileNotFoundError(
            "No tokenized EMOPIA training clips found. Novelty and fidelity metrics need the "
            "training corpus as their reference frame; run prepare_emopia first."
        )

    # Probes are mandatory, unlike the classifier, but a config that omits them should say so
    # rather than die with a bare KeyError two lines above the message written to explain
    # exactly this situation.
    probes_config = config.get("probes") or {}
    if not probes_config.get("directory"):
        raise FileNotFoundError(
            "No probes.directory configured. The evaluation needs the MIDI-to-quadrant probes; "
            "point it at models/real_training/music_probe or run musemotion.cli.train_probe."
        )
    probe_dir = resolve_path(probes_config["directory"])
    probes = load_probes(probe_dir, tokenizer_path=resolve_path(config["generator"]["tokenizer"]))
    if not probes:
        raise FileNotFoundError(
            f"No probe artifacts found under {probe_dir}. Run musemotion.cli.train_probe first."
        )
    probe_metadata = load_probe_metadata(probe_dir)
    ceilings = _probe_ceilings(probe_metadata)

    # A probe asked about a different clip length than it was trained on still answers, just
    # worse - measured, the ceiling falls from 0.657 to 0.574 when a 128-note probe is given
    # 32-note clips. Every accuracy in the run is divided by that ceiling, so a quiet mismatch
    # would move every number without any of them looking wrong.
    trained_budget = probe_metadata.get("note_budget")
    if trained_budget is not None and int(trained_budget) != note_budget:
        print(
            f"warning: the probes were trained on {trained_budget}-note clips but this run scores "
            f"{note_budget}-note clips. Accuracies and the ceiling will both be depressed by the "
            "length mismatch rather than by anything about the generator."
        )

    generator = MusicGeneratorComponent.from_checkpoint(
        resolve_path(config["generator"]["checkpoint"]),
        tokenizer_path=resolve_path(config["generator"]["tokenizer"]),
    )

    corpus = {
        view: build_corpus_index((row["notes"] for row in reference["train"]), view)
        for view in SYMBOL_VIEWS
    }
    reference_features = feature_matrix(
        (row["notes"] for row in reference["train"]), dist.comparable_feature_names()
    )

    systems: dict[str, dict[str, Any]] = {}
    # Shared across every row: standardisation uses the reference statistics alone, so the
    # reference set's own spread is the same number for all of them.
    fidelity_cache: dict[str, Any] = {}

    # The reference row: real held-out clips, scored exactly like generated ones.
    real_clips = [
        GeneratedClip(
            emotion_id=int(row["emotion_id"]),
            notes=row["notes"],
            token_ids=row["token_ids"],
            eos_token_id=tokenizer.eos_token_id,
        )
        for row in reference.get("test", [])
    ]
    if real_clips:
        # eos_is_learned=False: these clips were decoded and re-encoded, and the encoder appends
        # EOS unconditionally, so an EOS rate here would describe the tokenizer, not the music.
        systems["real"] = _score_system(
            real_clips, probes, ceilings, corpus, reference_features, max_tokens, config,
            eos_is_learned=False, fidelity_cache=fidelity_cache,
        )

    # The guidance sweep.
    for guidance in guidance_scales:
        label = f"guidance={guidance:g}"
        clips = _generate_conditioned(
            generator,
            tokenizer,
            clips_per_quadrant=clips_per_quadrant,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            guidance_scale=guidance,
            batch_size=batch_size,
            seed=seed,
            note_budget=note_budget,
            torch_module=torch,
        )
        systems[label] = _score_system(
            clips, probes, ceilings, corpus, reference_features, max_tokens, config,
            fidelity_cache=fidelity_cache,
        )

    # Floor 1: no conditioning at all, sampled from the null emotion embedding.
    unconditional = _generate_unconditional(
        generator,
        tokenizer,
        total_clips=clips_per_quadrant * len(EMOPIA_QUADRANTS),
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        batch_size=batch_size,
        seed=seed,
        note_budget=note_budget,
        torch_module=torch,
    )
    if unconditional:
        systems["unconditional"] = _score_system(
            unconditional, probes, ceilings, corpus, reference_features, max_tokens, config,
            score_round_trip=False, fidelity_cache=fidelity_cache,
        )

    # Floor 2: the deterministic random-piano baseline.
    random_clips = _generate_random_piano(
        count=int(generation_config.get("random_baseline_clips", 40)),
        note_budget=note_budget,
        tokenizer=tokenizer,
    )
    if random_clips:
        # Also re-encoded from MIDI rather than sampled, so its EOS rate is not a model property.
        systems["random_piano"] = _score_system(
            random_clips, probes, ceilings, corpus, reference_features, max_tokens, config,
            score_round_trip=False, eos_is_learned=False, fidelity_cache=fidelity_cache,
        )

    # The system as shipped: text through the classifier, then generation.
    end_to_end_payload: dict[str, Any] = {}
    if config.get("end_to_end", {}).get("enabled", True):
        end_to_end_payload = _run_end_to_end(
            config,
            generator,
            tokenizer,
            probes,
            ceilings,
            corpus,
            reference_features,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            guidance_scale=default_guidance,
            batch_size=batch_size,
            seed=seed,
            note_budget=note_budget,
            torch_module=torch,
            fidelity_cache=fidelity_cache,
            balanced_accuracies=_balanced_accuracies(systems, default_guidance),
        )
        if end_to_end_payload.get("system"):
            systems["end_to_end"] = end_to_end_payload["system"]

    rows = [comparison_row(label, payload) for label, payload in systems.items()]
    metrics = {
        "run": _run_metadata(config, seed, note_budget, max_tokens, guidance_scales),
        "probe_metadata": probe_metadata,
        "reference_counts": {name: len(rows_) for name, rows_ in reference.items()},
        "systems": systems,
        "comparison_table": rows,
        "comparison_table_markdown": comparison_table_markdown(rows),
        "end_to_end": {
            key: value for key, value in end_to_end_payload.items() if key != "system"
        },
        "notes": _interpretation_notes(),
    }

    output_dir = resolve_path(output_config.get("directory", "artifacts/evaluation"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "comparison_table.csv").write_text(comparison_table_csv(rows), encoding="utf-8")
    print("\n" + comparison_table_markdown(rows))
    print(f"\nwrote {output_dir / 'metrics.json'}")
    return metrics


def _generate_conditioned(
    generator: Any,
    tokenizer: MusicTokenizer,
    clips_per_quadrant: int,
    max_tokens: int,
    temperature: float,
    top_k: Any,
    guidance_scale: float,
    batch_size: int,
    seed: int,
    note_budget: int,
    torch_module: Any,
) -> list[GeneratedClip]:
    """Generate a balanced set of clips, forcing each quadrant in turn."""
    requests = [
        quadrant.id for quadrant in EMOPIA_QUADRANTS for _ in range(clips_per_quadrant)
    ]
    return _sample_requests(
        generator,
        tokenizer,
        requests,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        guidance_scale=guidance_scale,
        batch_size=batch_size,
        seed=seed,
        note_budget=note_budget,
        torch_module=torch_module,
    )


def _generate_unconditional(
    generator: Any,
    tokenizer: MusicTokenizer,
    total_clips: int,
    max_tokens: int,
    temperature: float,
    top_k: Any,
    batch_size: int,
    seed: int,
    note_budget: int,
    torch_module: Any,
) -> list[GeneratedClip]:
    """Sample from the null emotion embedding, with guidance necessarily switched off."""
    null_id = getattr(generator.model, "null_emotion_id", None)
    if null_id is None:
        return []
    requests = [int(null_id)] * total_clips
    clips = _sample_requests(
        generator,
        tokenizer,
        requests,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        guidance_scale=1.0,
        batch_size=batch_size,
        seed=seed + 1,
        note_budget=note_budget,
        torch_module=torch_module,
    )
    # There is no conditioning quadrant to compare against, so the label is left meaningless
    # and round-trip scoring is skipped for this row.
    for clip in clips:
        clip.emotion_id = -1
    return clips


def _sample_requests(
    generator: Any,
    tokenizer: MusicTokenizer,
    requests: Sequence[int],
    max_tokens: int,
    temperature: float,
    top_k: Any,
    guidance_scale: float,
    batch_size: int,
    seed: int,
    note_budget: int,
    torch_module: Any,
) -> list[GeneratedClip]:
    torch_module.manual_seed(seed)
    clips: list[GeneratedClip] = []
    for start in range(0, len(requests), max(1, batch_size)):
        chunk = list(requests[start : start + max(1, batch_size)])
        sequences = generator.model.sample_batch(
            emotion_ids=chunk,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=int(top_k) if top_k is not None else None,
            guidance_scale=guidance_scale,
            device=generator.device,
        )
        for emotion_id, token_ids in zip(chunk, sequences):
            clips.append(
                GeneratedClip(
                    emotion_id=int(emotion_id),
                    notes=crop_notes(tokenizer.decode_tokens(token_ids), note_budget),
                    token_ids=list(token_ids),
                    eos_token_id=tokenizer.eos_token_id,
                )
            )
    return clips


def _generate_random_piano(
    count: int, note_budget: int, tokenizer: MusicTokenizer
) -> list[GeneratedClip]:
    """Clips from the deterministic baseline generator, one per seed."""
    import tempfile

    from musemotion.music.random_generator import RandomPianoGeneratorConfig, generate_random_piano

    clips: list[GeneratedClip] = []
    with tempfile.TemporaryDirectory(prefix="musemotion-eval-random-") as temp_dir:
        for index in range(count):
            path = Path(temp_dir) / f"random_{index}.mid"
            try:
                generate_random_piano(RandomPianoGeneratorConfig(seed=260625 + index), path)
                notes = crop_notes(tokenizer.midi_to_notes(path), note_budget)
            except Exception as error:
                # This baseline needs pretty_midi, so failing here should drop the row rather
                # than the run. It is reported loudly and the partial result is discarded:
                # novelty, diversity, and fidelity are all sample-size sensitive, so scoring a
                # truncated baseline beside full-size rows would be worse than omitting it.
                print(
                    f"random-piano baseline omitted after {index} of {count} clips: {error}"
                )
                return []
            clips.append(
                GeneratedClip(
                    emotion_id=-1,
                    notes=notes,
                    token_ids=tokenizer.encode_notes(notes),
                    eos_token_id=tokenizer.eos_token_id,
                )
            )
    return clips


def _run_end_to_end(
    config: dict[str, Any],
    generator: Any,
    tokenizer: MusicTokenizer,
    probes: dict[str, Any],
    ceilings: dict[str, float | None],
    corpus: dict[str, CorpusIndex],
    reference_features: np.ndarray,
    max_tokens: int,
    temperature: float,
    top_k: Any,
    guidance_scale: float,
    batch_size: int,
    seed: int,
    note_budget: int,
    torch_module: Any,
    fidelity_cache: dict[str, Any] | None = None,
    balanced_accuracies: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Run the committed text-to-music path over the fixed prompt set.

    Uses the same classifier and generator the CLI and the Gradio app use. The only thing
    added is scoring the output.
    """
    from musemotion.inference.pipeline import BertEmotionClassifier

    end_to_end_config = config.get("end_to_end", {})
    # The config lookup is inside the guard too. A config with no `classifier:` section would
    # otherwise raise KeyError here - after the whole sweep has already been generated and
    # scored - and the run would end without writing metrics.json at all.
    try:
        classifier_dir = resolve_path(config["classifier"]["model_dir"])
        classifier = BertEmotionClassifier.from_pretrained(classifier_dir)
    except KeyError as error:
        return {"skipped": f"no classifier configured ({error}); end-to-end row omitted"}
    except Exception as error:
        # The BERT weights are fetched separately, so a missing checkpoint should degrade this
        # one row rather than discard the whole run.
        return {"skipped": f"classifier unavailable: {error}"}

    prompts = limited_prompts(end_to_end_config.get("max_prompts"))
    intended = [prompt.intended_emotion_id for prompt in prompts]
    predictions = [classifier.predict(prompt.text) for prompt in prompts]
    classified = [int(prediction["emotion_id"]) for prediction in predictions]

    clips = _sample_requests(
        generator,
        tokenizer,
        classified,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        guidance_scale=guidance_scale,
        batch_size=batch_size,
        seed=seed + 2,
        note_budget=note_budget,
        torch_module=torch_module,
    )
    # The row is scored against the quadrant the prompt was *meant* to express, not the one BERT
    # predicted. Scoring against BERT's output would give the only row named after the shipped
    # system exactly zero text-stage error by construction, and print it beside the guidance rows
    # as though the classifier were free. The conditioning-relative number is still computed
    # below, since that is the one comparable with the guidance rows.
    for clip, prompt, classified_id in zip(clips, prompts, classified):
        clip.source_text = prompt.text
        clip.classified_emotion_id = classified_id
        clip.emotion_id = prompt.intended_emotion_id

    system = _score_system(
        clips, probes, ceilings, corpus, reference_features, max_tokens, config,
        fidelity_cache=fidelity_cache,
    )
    system["round_trip_measures"] = "intended quadrant (full text-to-music error)"

    # Same clips, scored against what actually conditioned them: directly comparable with the
    # guidance rows, and the difference between the two is the text stage's cost.
    conditioned_view = [
        GeneratedClip(
            emotion_id=int(clip.classified_emotion_id),
            notes=clip.notes,
            token_ids=clip.token_ids,
            eos_token_id=clip.eos_token_id,
        )
        for clip in clips
    ]
    conditioned = _score_system(
        conditioned_view, probes, ceilings, corpus, reference_features, max_tokens, config,
        fidelity_cache=fidelity_cache,
    )
    system["round_trip_vs_conditioning"] = conditioned.get("round_trip")

    attribution: dict[str, Any] = {}
    for name, probe in probes.items():
        probabilities = _probe_probabilities(probe, clips)
        recovered = [int(value) for value in np.argmax(probabilities, axis=-1)]
        attribution[name] = stage_attribution(
            intended,
            classified,
            recovered,
            # The matching guidance row: the same generator over an equal number of clips per
            # quadrant. Comparing against that, rather than against its accuracy on the mix the
            # classifier happens to produce, is what makes the comparison an actual test.
            balanced_generator_accuracy=(balanced_accuracies or {}).get(name),
        )

    return {
        "system": system,
        "prompt_count": len(prompts),
        "guidance_scale": guidance_scale,
        "stage_attribution": attribution,
        "per_prompt": [
            {
                "text": prompt.text,
                "intended_quadrant": prompt.intended_quadrant,
                "classified_quadrant": quadrant_name(classified_id),
                "classifier_confidence": float(prediction["confidence"]),
            }
            for prompt, classified_id, prediction in zip(prompts, classified, predictions)
        ],
    }


def _score_system(
    clips: Sequence[GeneratedClip],
    probes: dict[str, Any],
    ceilings: dict[str, float | None],
    corpus: dict[str, CorpusIndex],
    reference_features: np.ndarray,
    max_tokens: int,
    config: dict[str, Any],
    score_round_trip: bool = True,
    eos_is_learned: bool = True,
    fidelity_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute every metric for one set of clips."""
    fidelity_cache = {} if fidelity_cache is None else fidelity_cache
    novelty_config = config.get("novelty", {})
    ngram_sizes = [int(value) for value in novelty_config.get("ngram_sizes", DEFAULT_NGRAM_SIZES)]
    diversity_sizes = [
        int(value) for value in novelty_config.get("diversity_sizes", DEFAULT_DIVERSITY_SIZES)
    ]
    headline_diversity_n = int(novelty_config.get("headline_diversity_n", 2))

    # Clips that decoded to nothing are counted, then excluded from the corpus-relative metrics.
    # Leaving them in flatters two columns at once and in the project's favour: an empty clip has
    # no n-grams so it scores a vacuous 1.0 novelty, and its all-zero feature row enters the
    # fidelity comparison as a fictional clip at pitch 0. The exclusion also restores symmetry
    # with the real row, which never contains empty clips because load_clip_splits drops
    # them. Round-trip and well-formedness still see every clip, because an empty generation is
    # a real failure of the system and must not be quietly dropped from its accuracy.
    scorable = [clip for clip in clips if clip.notes]
    payload: dict[str, Any] = {
        "clip_count": len(clips),
        "scored_clip_count": len(scorable),
        "excluded_empty_clips": len(clips) - len(scorable),
        "well_formedness": well_formedness(
            [clip.structure_record() for clip in clips], max_tokens, eos_is_learned=eos_is_learned
        ),
    }

    round_trips: dict[str, Any] = {}
    predictions_by_probe: dict[str, list[int]] = {}
    for name, probe in probes.items():
        probabilities = _probe_probabilities(probe, clips)
        if probabilities.size == 0:
            continue
        predictions_by_probe[name] = [int(value) for value in np.argmax(probabilities, axis=-1)]
        if score_round_trip:
            result = round_trip_result(
                name, [clip.emotion_id for clip in clips], probabilities, ceilings.get(name)
            )
            round_trips[name] = result.to_dict()
        else:
            # No conditioning quadrant exists for this row, so only the distribution of what
            # the probe assigns is meaningful.
            payload.setdefault("probe_label_distribution", {})[name] = _label_distribution(
                predictions_by_probe[name]
            )

    if round_trips:
        payload["round_trip_by_probe"] = round_trips
        primary = "feature" if "feature" in round_trips else next(iter(round_trips))
        payload["round_trip"] = round_trips[primary]
        payload["primary_probe"] = primary
    if len(predictions_by_probe) >= 2:
        names = list(predictions_by_probe)
        payload["probe_agreement"] = probe_agreement(
            predictions_by_probe[names[0]], predictions_by_probe[names[1]]
        )

    # symbolic_features is not cheap and was being recomputed for every consumer; computed
    # once here and passed to the three that need it.
    features = [symbolic_features(clip.notes) for clip in scorable]
    payload["novelty"] = _novelty_summary(scorable, corpus, ngram_sizes)
    payload["diversity"] = _diversity_summary(scorable, diversity_sizes, headline_diversity_n)
    payload["fidelity"] = _fidelity_summary(reference_features, features, fidelity_cache)
    payload["per_quadrant_features"] = _per_quadrant_features(scorable, features)
    return payload


def _probe_probabilities(probe: Any, clips: Sequence[GeneratedClip]) -> np.ndarray:
    if not clips:
        return np.zeros((0, len(EMOPIA_QUADRANTS)), dtype="float64")
    return np.asarray(probe.predict_proba_batch([clip.notes for clip in clips]), dtype="float64")


def _novelty_summary(
    clips: Sequence[GeneratedClip],
    corpus: dict[str, CorpusIndex],
    ngram_sizes: Sequence[int],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for view, to_symbols in SYMBOL_VIEWS.items():
        index = corpus[view]
        symbol_clips = [to_symbols(clip.notes) for clip in clips]  # type: ignore[operator]
        rates = {
            int(size): [index.novel_ngram_rate(symbols, int(size)) for symbols in symbol_clips]
            for size in ngram_sizes
        }
        runs = [index.longest_copied_run(symbols) for symbols in symbol_clips]
        duplicates = sum(1 for symbols in symbol_clips if index.contains_exact(symbols))
        summary[view] = {
            "mean_novel_ngram_rate": {
                str(size): float(np.mean(values)) if values else 0.0 for size, values in rates.items()
            },
            "mean_longest_copied_run": float(np.mean(runs)) if runs else 0.0,
            "max_longest_copied_run": int(np.max(runs)) if runs else 0,
            "exact_duplicate_rate": float(duplicates / len(clips)) if clips else 0.0,
        }
    pitch = summary.get("pitch", {})
    # Promote the headline numbers so the comparison table can read them without digging. The
    # size is whichever configured n is closest to 8 rather than a hardcoded "8": ngram_sizes is
    # configurable, and keying on a size that was not computed would silently blank the novelty
    # column in the CSV, the markdown, and the figure without anything raising.
    rates = pitch.get("mean_novel_ngram_rate", {})
    headline_size = min(rates, key=lambda size: (abs(int(size) - 8), int(size))) if rates else None
    summary["headline_ngram_size"] = int(headline_size) if headline_size is not None else None
    summary["mean_novel_8gram_rate"] = rates.get(headline_size) if headline_size else None
    summary["mean_longest_copied_run"] = pitch.get("mean_longest_copied_run")
    summary["max_longest_copied_run"] = pitch.get("max_longest_copied_run")
    return summary


def _diversity_summary(
    clips: Sequence[GeneratedClip],
    sizes: Sequence[int],
    headline_n: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"by_view": {}}
    for view, to_symbols in SYMBOL_VIEWS.items():
        symbol_clips = [to_symbols(clip.notes) for clip in clips]  # type: ignore[operator]
        summary["by_view"][view] = {
            str(size): set_diversity(symbol_clips, view, int(size)).to_dict() for size in sizes
        }
    # As with novelty, fall back to the nearest computed size rather than blanking the column
    # when headline_diversity_n is not one of the configured diversity_sizes.
    by_size = summary["by_view"].get("pitch", {})
    key = str(headline_n)
    if key not in by_size and by_size:
        key = min(by_size, key=lambda size: (abs(int(size) - headline_n), int(size)))
    headline = by_size.get(key, {})
    summary["mean_pairwise_diversity"] = headline.get("mean_pairwise_diversity")
    summary["mean_pairwise_cosine_diversity"] = headline.get("mean_pairwise_cosine_diversity")
    summary["headline_n"] = int(key) if key and key in by_size else headline_n
    return summary


def _fidelity_summary(
    reference_features: np.ndarray,
    features: Sequence[dict[str, float]],
    cache: dict[str, Any],
) -> dict[str, Any]:
    names = dist.comparable_feature_names()
    candidate = (
        np.vstack([[row[name] for name in names] for row in features])
        if features
        else np.zeros((0, len(names)))
    )
    if candidate.size == 0 or reference_features.size == 0:
        return {}
    overlaps = dist.per_feature_overlap(reference_features, candidate, names)
    # The reference set's own spread is identical for every row, so it is computed once for the
    # whole run rather than rebuilt from an 862-square broadcast per system.
    distances, computed = dist.set_distances(
        reference_features, candidate, intra_reference=cache.get("intra_reference")
    )
    cache["intra_reference"] = computed["intra_reference"]
    if "reference_feature_means" not in cache:
        cache["reference_feature_means"] = dist.feature_means(reference_features, names)
    return {
        "feature_names": names,
        "per_feature_overlap": overlaps,
        "mean_overlap": float(np.mean(list(overlaps.values()))),
        "distances": distances.to_dict(),
        "inter_over_intra": distances.inter_over_intra,
        # Reuses the distance matrix set_distances just built instead of recomputing it.
        "nearest_neighbour": dist.nearest_neighbour_from_matrix(computed["inter_matrix"]),
        "feature_means": dist.feature_means(candidate, names),
        "reference_feature_means": cache["reference_feature_means"],
    }


def _per_quadrant_features(
    clips: Sequence[GeneratedClip], features: Sequence[dict[str, float]]
) -> dict[str, dict[str, float]]:
    """Mean features per conditioning quadrant, for the distribution figure."""
    grouped: dict[str, list[dict[str, float]]] = {}
    for clip, row in zip(clips, features):
        if clip.emotion_id < 0:
            continue
        grouped.setdefault(quadrant_name(clip.emotion_id), []).append(row)
    summary: dict[str, dict[str, float]] = {}
    for quadrant, rows in grouped.items():
        keys = rows[0].keys()
        summary[quadrant] = {
            key: float(np.mean([row[key] for row in rows])) for key in keys
        }
        summary[quadrant]["clip_count"] = float(len(rows))
    return summary


def _label_distribution(predictions: Sequence[int]) -> dict[str, float]:
    total = len(predictions)
    if not total:
        return {}
    counts = {quadrant.name: 0 for quadrant in EMOPIA_QUADRANTS}
    for value in predictions:
        if 0 <= int(value) < len(EMOPIA_QUADRANTS):
            counts[quadrant_name(int(value))] += 1
    return {name: float(count / total) for name, count in counts.items()}


def _balanced_accuracies(
    systems: dict[str, Any], guidance: float
) -> dict[str, float]:
    """Per-probe round-trip accuracy from the guidance row matching the end-to-end run.

    Balanced across quadrants by construction, which is the reference the end-to-end row has
    to be compared against for the comparison to mean anything.

    ``generation.guidance_scale`` and ``generation.guidance_scales`` are separate config keys and
    nothing forces the first to appear in the second. When it does not, there is no matching row
    and the end-to-end row loses its only real test - so the miss is reported rather than
    returning an empty mapping and letting the run look complete.
    """
    label = f"guidance={guidance:g}"
    if label not in systems:
        available = ", ".join(name for name in systems if name.startswith("guidance=")) or "none"
        print(
            f"warning: no {label} row to compare the end-to-end result against (swept: "
            f"{available}). Add {guidance:g} to generation.guidance_scales, or the end-to-end "
            "row will report no quadrant-mix ratio."
        )
        return {}
    row = systems[label].get("round_trip_by_probe", {})
    return {
        name: value["overall"]["accuracy"]
        for name, value in row.items()
        if value.get("overall", {}).get("accuracy") is not None
    }


def _probe_ceilings(probe_metadata: dict[str, Any]) -> dict[str, float | None]:
    """Each probe's accuracy on real held-out clips, which bounds its round-trip accuracy."""
    ceilings: dict[str, float | None] = {}
    for name, values in probe_metadata.get("probes", {}).items():
        accuracy = values.get("test", {}).get("accuracy")
        ceilings[name] = float(accuracy) if accuracy is not None else None
    return ceilings


def _run_metadata(
    config: dict[str, Any],
    seed: int,
    note_budget: int,
    max_tokens: int,
    guidance_scales: Sequence[float],
) -> dict[str, Any]:
    """Stamp the run so two metrics files can be compared without ambiguity."""
    return {
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "seed": seed,
        "note_budget": note_budget,
        "max_tokens": max_tokens,
        "guidance_scales": list(guidance_scales),
        "prompt_set_size": len(EVALUATION_PROMPTS),
        "config": config,
    }


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(Path(__file__).resolve().parents[3]),
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def _interpretation_notes() -> dict[str, str]:
    """Caveats recorded alongside the numbers, so the file is not read without them."""
    return {
        "quality": (
            "No metric here measures musical quality or pleasantness. These measure "
            "controllability, novelty, and distributional similarity to real EMOPIA. "
            "Distributional similarity is the closest available proxy for quality and it is "
            "only a proxy; answering whether the music is good requires human listeners."
        ),
        "ceiling": (
            "Round-trip accuracy must be read against the probe ceiling, which is the same "
            "probe's accuracy on real held-out EMOPIA clips. The controllability ratio does "
            "that division."
        ),
        "end_to_end_bound": (
            "The end_to_end row is scored against the quadrant each prompt was meant to express, "
            "so it carries the text stage's error as well as the generator's. Its "
            "round_trip_vs_conditioning field holds the same clips scored against the quadrant "
            "that actually conditioned them, which is the number directly comparable with the "
            "guidance rows. Note that conditional_product (text accuracy times generation given "
            "correct text) is an algebraic identity with end-to-end accuracy whenever no clip is "
            "recovered by accident - the text term cancels - so their agreement is not evidence "
            "about the stages. The real comparison is quadrant_mix_ratio, which divides observed "
            "end-to-end accuracy by text accuracy times the generator's accuracy over a balanced "
            "set of quadrants; above 1.0 means the classifier's predictions land on quadrants the "
            "generator renders more legibly than average."
        ),
        "fidelity_has_two_views": (
            "mean_feature_overlap averages per-feature histogram overlaps, so it describes the "
            "marginal distributions; inter_over_intra is a distance in the joint 28-dimensional "
            "feature space. They can disagree, and on this generator they do: across the guidance "
            "sweep the marginals hold roughly flat (0.795, 0.810, 0.803) while the joint distance "
            "grows monotonically (1.036, 1.097, 1.186). Guidance keeps each feature in range while "
            "moving their combinations away from real music. Read both. "
            "inter_over_intra in particular is fooled by a tightly clustered set near the centre of "
            "the reference distribution: the random-piano baseline scores 0.975 on it, nominally "
            "closer to real EMOPIA than real held-out clips are, while its marginal overlap of "
            "0.324 correctly identifies it as the worst row in the table."
        ),
        "diversity_saturation": (
            "Jaccard diversity saturates at 1.0 for n >= 4 even on real EMOPIA clips, where no "
            "two clips share a single 4-note pattern. At those sizes it works only as a "
            "collapse tripwire; the graded signal is at n = 1 and n = 2, and in the feature "
            "distances."
        ),
        "clip_length": (
            "Most generated clips run to the token cap rather than emitting an ending, so clip "
            "length is largely a sampling hyperparameter here; length-dependent features are "
            "excluded from the fidelity comparison for that reason. The EOS rate is not zero "
            "though, and it rises monotonically with guidance (measured over 200 clips per "
            "setting: 0.055, 0.085, 0.145, 0.305 at guidance 1, 2, 3 and 5), with mean note count "
            "falling 126 to 108 across the same range - guidance shortens clips as well as "
            "sharpening their emotion. The "
            "real and random_piano rows report no EOS rate at all: those clips are re-encoded "
            "through the tokenizer, which appends EOS unconditionally."
        ),
    }


__all__ = ["GeneratedClip", "run_evaluation"]
