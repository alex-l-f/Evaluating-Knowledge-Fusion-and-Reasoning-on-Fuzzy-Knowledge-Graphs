import os
import sys

def split_lines(folderpath, filename, num_lines=3):
    result = []
    file_path = folderpath + filename
    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.split('\t', num_lines)
                result.append(parts)
    return result

#given a url string, grab the final part of the url
def get_url_tail(url):
    return url.rsplit('/', 1)[1].strip('\n')

#given a list of urls, grab the final part of each url
def get_url_tails(urls_list):
    p_list = []
    for urls in urls_list:
        fact_list = []
        for url in urls:
            fact_list.append(get_url_tail(url))
        p_list.append(fact_list)
    return p_list

#map each relation to a genric id
def map_generic(rels, gmap):
    final_list = []
    for rel in rels:
        fact = []
        if rel[0] in gmap:
            fact.append(gmap[rel[0]])
        else:
            fact.append(rel[0])
        
        if rel[2] in gmap:
            fact.append(gmap[rel[2]])
        else:
            fact.append(rel[2])

        final_list.append([fact[0], rel[1], fact[1]])
    return final_list

def map_generic_2(rels, gmap):
    final_list = []
    for rel in rels:
        fact = []
        if rel[0] in gmap:
            fact.append(gmap[rel[0]])
        else:
            fact.append(rel[0])
        
        if rel[1] in gmap:
            fact.append(gmap[rel[1]])
        else:
            fact.append(rel[1])

        final_list.append([fact[0], fact[1]])
    return final_list

folder_path = "./data/OpenEA/EN_DE_100K_V2/"#sys.argv[1]

#read in kg 1 relations
kg_rels_1 = get_url_tails(split_lines(folder_path, "rel_triples_1"))
#read in kg 2 relations
kg_rels_2 = get_url_tails(split_lines(folder_path, "rel_triples_2"))

#read in a mapping file that maps kg 1 entites to kg 2 entites
#rel_map = get_url_tails(split_lines(folder_path, "ent_links", 2))

#create a generic mapping for either direction
#gmap = {}
#for i, rel in enumerate(rel_map):
#    gmap[rel[0]] = i
#    gmap[rel[1]] = i

final_list = []
final_list.extend(kg_rels_1)

final_list.extend(kg_rels_2)
    
#read in test data for cross KG links, and create new fact with custom relation
#also process validation and test data
val_list = []
test_list = []
cust_rel = "link"
#for each folder in ./721_5fold
for folder in os.listdir(folder_path + '721_5fold/'):
    #if the folder is a number
    if folder.isdigit():
        test_data = get_url_tails(split_lines(folder_path + '721_5fold/' + folder + "/", "test_links", 2))
        #map test facts to generic ids
        for fact in test_data:
            final_list.append([fact[0], cust_rel, fact[1]])
        #load in val data
        val_data = get_url_tails(split_lines(folder_path + '721_5fold/' + folder + "/", "valid_links", 2))
        #map val facts to generic ids
        for fact in val_data:
            val_list.append([fact[0], cust_rel, fact[1]])
        #load in test data
        test_data = get_url_tails(split_lines(folder_path + '721_5fold/' + folder + "/", "test_links", 2))
        #map test facts to generic ids
        for fact in test_data:
            test_list.append([fact[0], cust_rel, fact[1]])

#write out the final list to a file, + the valid and test data
with open(folder_path + "/final_list.tsv", 'w') as file:
    for fact in final_list:
        file.write(f"{fact[0]}\t{fact[1]}\t{fact[2]}\t1.00\n")
with open(folder_path + "/valid_list.tsv", 'w') as file:
    for fact in val_list:
        file.write(f"{fact[0]}\t{fact[1]}\t{fact[2]}\t1.00\n")
with open(folder_path + "/test_list.tsv", 'w') as file:
    for fact in test_list:
        file.write(f"{fact[0]}\t{fact[1]}\t{fact[2]}\t1.00\n")