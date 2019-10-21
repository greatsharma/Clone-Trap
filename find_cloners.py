import os
import argparse
from utils.fuzzy_match import get_cloners
from utils.file_handler import file_reader

# docs_i = ['this is a test for fuzzy wuzzy match', 'a test for fuzzy match', 'test fuzzy matching', 'this is a test for fuzzy wuzzy match',
#           'fuzz wuzz tester', "let's test this program", 'this a test fuzzy wuzzy match', 'this test for fuzzy wuzzy match', 'test fuzz matching',
#           'this is test for fuzy wuzy match', 'this is a for fuzzy wuzzy match', 'testing fuzz wuzz match', 'fuzzy wuzzy tester']

# for i in range(1, len(docs)+1):
#     try:
#         with open('fuzzy_docs/{}.txt'.format(i), 'w+') as file:
#             file.write(docs[i-1])
#     except Exception as e:
#         print(e)

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--docs_path', required=True,
                help='Path to the folder containing all the docs to be matched')
args = vars(ap.parse_args())

docs = []
files = os.listdir(args['docs_path'])
files = sorted([int(f.rstrip('.txt')) for f in files])

for file in files:
    docs.append(file_reader(args['docs_path'] + '/' + str(file) + '.txt'))

cloners = get_cloners(docs, thresh=0.85)
print('total cloner groups : ', len(cloners), end='\n\n')
print(cloners, end='\n\n')

# c_table = get_cloners_table(docs)

# print('    ', end='')
# for i in range(len(docs)+1):
#     print('d'+str(i), end='       ')

# for i in range(len(docs)+1):
#     print('\nd'+str(i), end='  ')

#     for j in range(len(docs)+1):
#         print(c_table[i][j], end='     ')

# print('\n\n')
# cloners = find_cloners_from_table(c_table, thresh=0.85)
# print('total cloner groups ', len(cloners), end='\n\n')
# print(cloners, end='\n')
