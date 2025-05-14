from enhance_image import image_enhancer
from grade_paper import ProcessPage
from transform_image import transform_paper_image


def answer_detector(image):
    image = image_enhancer(image, block_size=51, blur_ksize=5, C=9, morph_kernel_size=1)
    image, paper, biggestContour = transform_paper_image(image)
    answers, paper, codes = ProcessPage(paper)
    return image, paper, biggestContour, answers, codes