import cv2
import argparse
from utils.fuzzy_match import levenshtein_distance
from utils.vision import draw_image, image_to_text
from utils.file_handler import file_writer, file_reader

ap = argparse.ArgumentParser()
ap.add_argument('-pi', '--parent_img', required=True,
                help='Path to the image file of parent doc')
ap.add_argument('-pt', '--parent_txt', required=True,
                help='Path to the text file of parent doc')
ap.add_argument('-di', '--docs_img', required=True,
                help='Path to the text file containing path to all documents images seperated by newline')
ap.add_argument('-dt', '--docs_txt', required=True,
                help='Path to the text file containing path to all documents text files seperated by newline')
args = vars(ap.parse_args())


p_doc_img = cv2.imread(args['parent_img'])
file_writer(args['parent_txt'], image_to_text(p_doc_img)[1])
pattern_doc = file_reader(args['parent_txt'])

docs_img_path = []
with open(args['docs_img'], 'r') as file:
    docs_img_path = file.readlines()

docs_text_path = []
with open(args['docs_txt'], 'r') as file:
    docs_text_path = file.readlines()

assert len(docs_img_path) == len(docs_text_path),\
    'number of paths in {} and {} should be same'.format(args['docs_img'],
                                                         args['docs_txt'])

docs = []
for doc_i_p, doc_t_p in zip(docs_img_path, docs_text_path):
    doc_img = cv2.imread(doc_i_p.rstrip('\n'))
    file_writer(doc_t_p.rstrip('\n'), image_to_text(doc_img)[1])
    docs.append(file_reader(doc_t_p.rstrip('\n')))


similarity_score = levenshtein_distance(pattern_doc, docs)
similarity_score = sorted(similarity_score.items(), key=lambda kv: kv[1])
print(similarity_score)
