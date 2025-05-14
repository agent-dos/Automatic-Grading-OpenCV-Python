from grade_paper import ProcessPage
from transform_image import transform_paper_image


def detect_answer(image):
    image, paper, biggestContour = transform_paper_image(image)
    answers, paper, codes = ProcessPage(paper)
    return image, paper, biggestContour, answers, codes