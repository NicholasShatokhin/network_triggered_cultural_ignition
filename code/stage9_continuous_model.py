from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import numpy as np
from numba import njit

# Fixed population architecture retained from Stage 8 for comparability.
N_GROUPS = 16
GROUP_SIZE = 20
N_AGENTS = N_GROUPS * GROUP_SIZE
N_DOMAINS = 6
MAX_LEVEL = 5
GENERATIONS = 420
YEARS_PER_GENERATION = 25
EVENT_START = 120
BASE_EXTERNAL_MODEL_PROB = 0.002
BASE_MIGRATION_RATE = 0.0005

# Parameter vector indices.
I_LEARN_BASE = 0
I_LEARN_COG = 1
I_LEVEL_DIFFICULTY = 2
I_TEACHING_MAX = 3
I_INNOV_RATE = 4
I_NON_SPEC_FACTOR = 5
I_SKILL_BENEFIT = 6
I_BRAIN_COST = 7
I_MUT_SD = 8
I_EXTRA_MODELS_MAX = 9
I_RECOMB_STRENGTH = 10
I_MIGRATION_SCALE = 11
I_BASE_MODELS = 12
I_TEACH_MID = 13
I_TEACH_SCALE = 14
I_INFRA_STEEPNESS = 15

DEFAULT_PARAMS = np.array([
    0.65,   # learn base
    0.32,   # cognition contribution
    0.055,  # level difficulty
    0.32,   # maximum model-specific teaching bonus
    0.58,   # innovation rate
    0.001,  # non-speciality innovation factor
    0.070,  # fitness benefit per skill level
    0.52,   # quadratic brain cost
    0.015,  # cognitive mutation SD
    5.0,    # maximum smoothly added cultural models
    6.0,    # recombinational innovation strength
    0.05,   # migration response to contact
    3.0,    # baseline number of models
    12.0,   # repertoire midpoint for teacher quality
    2.5,    # repertoire scale for teacher quality
    12.0,   # steepness of smooth learning-infrastructure response
], dtype=np.float64)

# Ablation bit flags.
ABL_NO_SPECIALISATION = 1
ABL_NO_RECOMBINATION = 2
ABL_NO_MIGRATION = 4
ABL_NO_TEACHING = 8
ABL_NO_GENE_CULTURE = 16
ABL_NO_INFRA_FEEDBACK = 32
ABL_NO_CONTACT_EVENT = 64


@njit(cache=True)
def sigmoid(x: float) -> float:
    if x > 60.0:
        return 1.0
    if x < -60.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


@njit(cache=True)
def weighted_choice(weights: np.ndarray) -> int:
    total = 0.0
    for i in range(weights.shape[0]):
        total += weights[i]
    if total <= 0.0:
        return np.random.randint(weights.shape[0])
    draw = np.random.random() * total
    acc = 0.0
    for i in range(weights.shape[0]):
        acc += weights[i]
        if acc >= draw:
            return i
    return weights.shape[0] - 1


@njit(cache=True)
def group_metrics(skills: np.ndarray, start: int, params: np.ndarray) -> Tuple[float, float, float]:
    """Continuous diversity, teaching-pool and learning-infrastructure scores."""
    mean_total = 0.0
    teacher_pool = 0.0
    domain_presence = 0.0
    for j in range(GROUP_SIZE):
        idx = start + j
        total = 0.0
        for d in range(N_DOMAINS):
            total += skills[idx, d]
        mean_total += total
        teacher_pool += sigmoid((total - params[I_TEACH_MID]) / params[I_TEACH_SCALE])
    mean_total /= GROUP_SIZE
    teacher_pool /= GROUP_SIZE

    for d in range(N_DOMAINS):
        mean_level = 0.0
        for j in range(GROUP_SIZE):
            mean_level += skills[start + j, d]
        mean_level /= GROUP_SIZE
        # Smoothly counts a domain as socially represented around level 1.5.
        domain_presence += sigmoid((mean_level - 1.5) / 0.45)
    diversity = domain_presence / N_DOMAINS
    complexity = mean_total / (N_DOMAINS * MAX_LEVEL)
    raw = 0.42 * complexity + 0.38 * diversity + 0.20 * teacher_pool
    infrastructure = sigmoid(params[I_INFRA_STEEPNESS] * (raw - 0.47))
    return diversity, teacher_pool, infrastructure


@njit(cache=True)
def classify_agent(skills: np.ndarray, idx: int, total_threshold: int, domain_threshold: int) -> bool:
    total = 0
    domains = 0
    for d in range(N_DOMAINS):
        level = skills[idx, d]
        total += level
        if level >= 2:
            domains += 1
    return total >= total_threshold and domains >= domain_threshold


@njit(cache=True)
def run_model(
    seed: int,
    event_external_probability: float,
    event_duration: int,
    params: np.ndarray,
    ablation_flags: int = 0,
    topology_mode: int = 0,
    return_trace: bool = False,
):
    np.random.seed(seed)

    cognition = np.empty(N_AGENTS, dtype=np.float64)
    skills = np.zeros((N_AGENTS, N_DOMAINS), dtype=np.int8)

    for group in range(N_GROUPS):
        speciality = group % N_DOMAINS
        start = group * GROUP_SIZE
        for j in range(GROUP_SIZE):
            idx = start + j
            cognition[idx] = min(0.80, max(0.10, 0.32 + 0.04 * np.random.randn()))
            if np.random.random() < 0.50:
                skills[idx, speciality] = 1

    mean_rep_trace = np.zeros(GENERATIONS, dtype=np.float64)
    primary_trace = np.zeros(GENERATIONS, dtype=np.float64)
    cognition_trace = np.zeros(GENERATIONS, dtype=np.float64)
    infra_trace = np.zeros(GENERATIONS, dtype=np.float64)

    low_good = 0
    primary_good = 0
    strict_good = 0
    first_primary = -1

    for generation in range(GENERATIONS):
        in_event = EVENT_START <= generation < EVENT_START + event_duration
        if ablation_flags & ABL_NO_CONTACT_EVENT:
            external_probability = BASE_EXTERNAL_MODEL_PROB
        else:
            external_probability = event_external_probability if in_event else BASE_EXTERNAL_MODEL_PROB
        if external_probability > 1.0:
            external_probability = 1.0

        migration_rate = BASE_MIGRATION_RATE + params[I_MIGRATION_SCALE] * external_probability
        if ablation_flags & ABL_NO_MIGRATION:
            migration_rate = 0.0

        next_cognition = np.empty_like(cognition)
        next_skills = np.zeros_like(skills)
        mean_infra = 0.0

        for group in range(N_GROUPS):
            start = group * GROUP_SIZE
            speciality = group % N_DOMAINS
            diversity, teacher_pool, infrastructure = group_metrics(skills, start, params)
            if ablation_flags & ABL_NO_INFRA_FEEDBACK:
                infrastructure = 0.0
            mean_infra += infrastructure

            fitness = np.empty(GROUP_SIZE, dtype=np.float64)
            for j in range(GROUP_SIZE):
                idx = start + j
                total_skill = 0.0
                for d in range(N_DOMAINS):
                    total_skill += skills[idx, d]
                if ablation_flags & ABL_NO_GENE_CULTURE:
                    # Remove the cultural benefit of cognition while retaining its metabolic cost.
                    fitness[j] = math.exp(
                        - params[I_BRAIN_COST] * cognition[idx] * cognition[idx]
                    )
                else:
                    fitness[j] = math.exp(
                        params[I_SKILL_BENEFIT] * total_skill
                        - params[I_BRAIN_COST] * cognition[idx] * cognition[idx]
                    )

            expected_models = params[I_BASE_MODELS] + params[I_EXTRA_MODELS_MAX] * infrastructure
            n_models = int(expected_models)
            if np.random.random() < expected_models - n_models:
                n_models += 1
            if n_models < 1:
                n_models = 1
            if n_models > 12:
                n_models = 12

            for learner_j in range(GROUP_SIZE):
                parent = start + weighted_choice(fitness)
                learner_cognition = cognition[parent] + params[I_MUT_SD] * np.random.randn()
                learner_cognition = min(0.95, max(0.10, learner_cognition))

                best_level = np.zeros(N_DOMAINS, dtype=np.int8)
                best_teacher_quality = np.zeros(N_DOMAINS, dtype=np.float64)
                chosen_model = -1
                chosen_model_total = -1
                observed_domains = np.zeros(N_DOMAINS, dtype=np.int8)

                for model_i in range(n_models):
                    if np.random.random() < external_probability:
                        if topology_mode == 1:
                            source_group = (group + (1 if np.random.random() < 0.5 else -1)) % N_GROUPS
                        else:
                            source_group = np.random.randint(N_GROUPS - 1)
                            if source_group >= group:
                                source_group += 1
                        model = source_group * GROUP_SIZE + np.random.randint(GROUP_SIZE)
                    else:
                        model = start + np.random.randint(GROUP_SIZE)

                    model_total = 0
                    for d in range(N_DOMAINS):
                        model_total += skills[model, d]
                        if skills[model, d] >= 2:
                            observed_domains[d] = 1
                    quality = sigmoid((model_total - params[I_TEACH_MID]) / params[I_TEACH_SCALE])

                    if ablation_flags & ABL_NO_RECOMBINATION:
                        if model_total > chosen_model_total:
                            chosen_model_total = model_total
                            chosen_model = model
                    else:
                        for d in range(N_DOMAINS):
                            if skills[model, d] > best_level[d]:
                                best_level[d] = skills[model, d]
                                best_teacher_quality[d] = quality

                if ablation_flags & ABL_NO_RECOMBINATION:
                    if chosen_model < 0:
                        chosen_model = start + np.random.randint(GROUP_SIZE)
                    model_total = 0
                    for d in range(N_DOMAINS):
                        best_level[d] = skills[chosen_model, d]
                        model_total += skills[chosen_model, d]
                    quality = sigmoid((model_total - params[I_TEACH_MID]) / params[I_TEACH_SCALE])
                    for d in range(N_DOMAINS):
                        best_teacher_quality[d] = quality

                observed_diversity = 0.0
                for d in range(N_DOMAINS):
                    observed_diversity += observed_domains[d]
                observed_diversity /= N_DOMAINS

                for d in range(N_DOMAINS):
                    copied_level = 0
                    for level in range(1, best_level[d] + 1):
                        teaching_bonus = 0.0
                        if not (ablation_flags & ABL_NO_TEACHING):
                            teaching_bonus = params[I_TEACHING_MAX] * best_teacher_quality[d] * (0.35 + 0.65 * infrastructure)
                        copy_probability = (
                            params[I_LEARN_BASE]
                            + params[I_LEARN_COG] * learner_cognition
                            - params[I_LEVEL_DIFFICULTY] * level
                            + teaching_bonus
                        )
                        copy_probability = min(0.995, max(0.02, copy_probability))
                        if np.random.random() < copy_probability:
                            copied_level = level
                        else:
                            break
                    next_skills[start + learner_j, d] = copied_level

                if ablation_flags & ABL_NO_SPECIALISATION:
                    innovation_domain = np.random.randint(N_DOMAINS)
                    # Match the full model's mean innovation opportunity while removing domain specialisation.
                    speciality_factor = 0.875
                else:
                    if np.random.random() < 0.85:
                        innovation_domain = speciality
                    else:
                        innovation_domain = np.random.randint(N_DOMAINS)
                    speciality_factor = 1.0 if innovation_domain == speciality else params[I_NON_SPEC_FACTOR]

                current_level = next_skills[start + learner_j, innovation_domain]
                innovation_multiplier = 1.0
                # The recombination ablation disables both cross-model skill
                # assembly and the diversity-dependent innovation synergy.
                if not (ablation_flags & ABL_NO_RECOMBINATION) and not (ablation_flags & ABL_NO_INFRA_FEEDBACK):
                    innovation_multiplier += params[I_RECOMB_STRENGTH] * observed_diversity * infrastructure
                innovation_probability = (
                    params[I_INNOV_RATE]
                    * speciality_factor
                    * learner_cognition
                    * math.exp(-0.18 * current_level)
                    * innovation_multiplier
                )
                if innovation_probability > 0.95:
                    innovation_probability = 0.95
                if np.random.random() < innovation_probability and current_level < MAX_LEVEL:
                    next_skills[start + learner_j, innovation_domain] += 1

                next_cognition[start + learner_j] = learner_cognition

        mean_infra /= N_GROUPS

        n_swaps = int(migration_rate * N_AGENTS / 2.0)
        for _ in range(n_swaps):
            ga = np.random.randint(N_GROUPS)
            gb = np.random.randint(N_GROUPS - 1)
            if gb >= ga:
                gb += 1
            a = ga * GROUP_SIZE + np.random.randint(GROUP_SIZE)
            b = gb * GROUP_SIZE + np.random.randint(GROUP_SIZE)
            tmpc = next_cognition[a]
            next_cognition[a] = next_cognition[b]
            next_cognition[b] = tmpc
            for d in range(N_DOMAINS):
                tmps = next_skills[a, d]
                next_skills[a, d] = next_skills[b, d]
                next_skills[b, d] = tmps

        cognition = next_cognition
        skills = next_skills

        total_rep = 0.0
        n_low = 0
        n_primary = 0
        n_strict = 0
        for idx in range(N_AGENTS):
            total = 0
            domains = 0
            for d in range(N_DOMAINS):
                level = skills[idx, d]
                total += level
                if level >= 2:
                    domains += 1
            total_rep += total
            if total >= 14 and domains >= 4:
                n_low += 1
            if total >= 16 and domains >= 5:
                n_primary += 1
            if total >= 20 and domains >= 6:
                n_strict += 1

        mean_rep = total_rep / N_AGENTS
        frac_low = n_low / N_AGENTS
        frac_primary = n_primary / N_AGENTS
        frac_strict = n_strict / N_AGENTS
        if first_primary < 0 and frac_primary > 0.10:
            first_primary = generation

        if generation >= GENERATIONS - 50:
            if frac_low > 0.50 and mean_rep > 16.0:
                low_good += 1
            if frac_primary > 0.50 and mean_rep > 18.0:
                primary_good += 1
            if frac_strict > 0.50 and mean_rep > 22.0:
                strict_good += 1

        mean_rep_trace[generation] = mean_rep
        primary_trace[generation] = frac_primary
        cognition_trace[generation] = cognition.mean()
        infra_trace[generation] = mean_infra

    persistent_low = low_good >= 45
    persistent_primary = primary_good >= 45
    persistent_strict = strict_good >= 45

    return (
        persistent_low,
        persistent_primary,
        persistent_strict,
        first_primary,
        mean_rep_trace[-1],
        primary_trace[-1],
        cognition_trace[-1],
        infra_trace[-1],
        mean_rep_trace,
        primary_trace,
        cognition_trace,
        infra_trace,
    )


if __name__ == '__main__':
    # JIT warm-up and small smoke test.
    for contact in [0.02, 0.05, 0.08, 0.12, 0.20]:
        hits = 0
        for seed in range(20):
            r = run_model(seed, contact, 60, DEFAULT_PARAMS)
            hits += int(r[1])
        print(contact, hits / 20)
