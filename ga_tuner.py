# ga_tuner.py

import random
from enhance_image import image_enhancer
from transform_image import transform_paper_image
from grade_paper import ProcessPage


PARAM_BOUNDS = {
    "blur_ksize": (5, 11),
    "block_size": (51, 51),
    "C": (5, 10),
    "morph_kernel_size": (1, 1)
}


def make_odd(x):
    return x + 1 if x % 2 == 0 else x


def generate_individual():
    return [
        random.randint(*PARAM_BOUNDS["blur_ksize"]),
        random.randint(*PARAM_BOUNDS["block_size"]),
        random.randint(*PARAM_BOUNDS["C"]),
        random.randint(*PARAM_BOUNDS["morph_kernel_size"]),
    ]


def mutate(individual):
    i = random.randint(0, len(individual) - 1)
    if i in [0, 1]:
        min_val, max_val = list(PARAM_BOUNDS.values())[i]
        individual[i] = make_odd(random.randint(min_val, max_val))
    else:
        individual[i] = random.randint(*list(PARAM_BOUNDS.values())[i])
    return individual


def crossover(p1, p2):
    idx = random.randint(1, len(p1) - 2)
    return p1[:idx] + p2[idx:], p2[:idx] + p1[idx:]


def fitness(individual, image):
    blur_ksize, block_size, C, morph_kernel_size = individual
    try:
        enhanced = image_enhancer(
            image.copy(), blur_ksize, block_size, C, morph_kernel_size)
        detection_img, warped_paper, _, method_used, _ = transform_paper_image(
            enhanced)
        if method_used == "fallback" or warped_paper is None:
            return -999, None, None, None
        answers, graded_img, codes = ProcessPage(warped_paper)
        valid = sum(1 for a in answers if a != '?')
        return valid, enhanced, graded_img, detection_img
    except:
        return -999, None, None, None


def run_genetic_algorithm(image, pop_size=8, generations=3, mutation_rate=0.2, show_progress=lambda msg: None):
    population = [generate_individual() for _ in range(pop_size)]
    best_score = -999
    best_result = {}

    for gen in range(generations):
        fitness_scores = []

        for ind in population:
            score, enh, graded, detect = fitness(ind, image)
            fitness_scores.append((score, ind, enh, graded, detect))
            if score > best_score:
                best_score = score
                best_result = {
                    "score": score,
                    "params": ind,
                    "enhanced": enh,
                    "graded": graded,
                    "detection": detect
                }

        show_progress(f"Gen {gen + 1}: Best Score = {best_score}")

        fitness_scores.sort(key=lambda x: x[0], reverse=True)
        selected = [x[1] for x in fitness_scores[:pop_size // 2]]

        new_population = []
        while len(new_population) < pop_size:
            p1, p2 = random.sample(selected, 2)
            child1, child2 = crossover(p1, p2)
            if random.random() < mutation_rate:
                child1 = mutate(child1)
            if random.random() < mutation_rate:
                child2 = mutate(child2)
            new_population += [child1, child2]

        population = new_population[:pop_size]

    return best_result
