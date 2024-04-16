import os

#number of train links to extract
#-1 for all
num_train = -1
#repeate factor for each link
repeats = 1
#dataset fold to use
fold = 1

def split_lines(folderpath, filename, num_lines=3):
    result = []
    file_path = folderpath + filename
    if os.path.isfile(file_path):
        with open(file_path, 'r', errors="ignore") as file:
            for line in file:
                parts = line.split('\t', num_lines)
                result.append(parts)
    return result

#given a url string, grab the final part of the url
def get_url_tail(url):
    if '/' in url:
        return url.rsplit('/', 1)[1].strip('\n')
    return url.strip('\n')

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

folder_path = "./data/unprocessed/OpenEA/EN_DE_100K_V2/"#sys.argv[1]

#read in kg 1 relations
kg_rels_1 = get_url_tails(split_lines(folder_path, "rel_triples_1"))
#read in kg 2 relations
kg_rels_2 = get_url_tails(split_lines(folder_path, "rel_triples_2"))

#process attribute data
#read in kg 1 attributes
kg_atts = get_url_tails(split_lines(folder_path, "attr_triples_1"))
#read in kg 2 attributes
kg_atts.extend(get_url_tails(split_lines(folder_path, "attr_triples_2")))

#read in test data for cross KG links, and create new fact with custom relation
#also process validation and test data
#for each folder in ./721_5fold
for folder in os.listdir(folder_path + '721_5fold/'):
    val_list = []
    test_list = []
    link_list = []
    cust_rel = "link"
    added_facts = 0
    #if the folder is a number
    if folder.isdigit():
        test_data = get_url_tails(split_lines(folder_path + '721_5fold/' + folder + "/", "train_links", 2))
        #map test facts to generic ids
        for fact in test_data:
            if num_train != -1:
                if added_facts < num_train:
                    added_facts += 1
                else:
                    continue
            for i in range(repeats): link_list.append([fact[0], cust_rel, fact[1]])
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

        #write all other files to each folder for ease of use
        with open(folder_path + folder + "/kg1_list.tsv", 'w') as file:
            for fact in kg_rels_1:
                file.write(f"{fact[0]}\t{fact[1]}\t{fact[2]}\t1.00\n")
        with open(folder_path + folder + "/kg2_list.tsv", 'w') as file:
            for fact in kg_rels_2:
                file.write(f"{fact[0]}\t{fact[1]}\t{fact[2]}\t1.00\n")
        with open(folder_path + folder + "/link_list.tsv", 'w') as file:
            for fact in link_list:
                file.write(f"{fact[0]}\t{fact[1]}\t{fact[2]}\t1.00\n")
                #also write reverse link
                file.write(f"{fact[2]}\t{fact[1]}\t{fact[0]}\t1.00\n")
        with open(folder_path + folder + "/valid_list.tsv", 'w') as file:
            for fact in val_list:
                file.write(f"{fact[0]}\t{fact[1]}\t{fact[2]}\t1.00\n")
        with open(folder_path + folder + "/test_list.tsv", 'w') as file:
            for fact in test_list:
                file.write(f"{fact[0]}\t{fact[1]}\t{fact[2]}\t1.00\n")


        #write out the final list to a file
        with open(folder_path + folder + "/kg_atts.tsv", 'w') as file:
            for fact in kg_atts:
                file.write(f"{fact[0]}\thasAttribute\t{fact[1]}\t1.00\n")

#for a set of all entites in kg1 and kg2
test_data = get_url_tails(split_lines(folder_path + '721_5fold/' + folder + "/", "test_links", 2))
#map test facts to generic ids
test_list = []
for fact in test_data:
    test_list.append([fact[0], cust_rel, fact[1]])
ents1 = set()
ents2 = set()
for fact in test_list:
    ents1.add(fact[0])
    ents2.add(fact[2])

#save all entites and the corresponding id to a file
with open(folder_path + "ent_ids.tsv", 'w') as file:
    for ent in ents1:
        file.write(f"{ent}\t1\n")
    for ent in ents2:
        file.write(f"{ent}\t2\n")