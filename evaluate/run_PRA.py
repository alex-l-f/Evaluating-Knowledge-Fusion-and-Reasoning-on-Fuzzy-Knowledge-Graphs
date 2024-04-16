import random
from collections import defaultdict
from tqdm.auto import tqdm
import os

def load_knowledge_graph(tsv_file, reverse=False):
    knowledge_graph = defaultdict(list)
    with open(tsv_file, 'r') as f:
            #next(f)  # Skip header
            for line in f:
                head, relation, tail, _ = line.strip().split('\t')
                knowledge_graph[head].append((relation, tail))
                if reverse:
                    #be fair, make the rels invertable
                    knowledge_graph[tail].append((relation, head))
    return knowledge_graph

#laod multiple files into the same knowledge graph
def load_knowledge_graphs(tsv_files, reverse=False):
    knowledge_graph = defaultdict(list)
    for tsv_file in tsv_files:
        #if teh file is missing, skip it
        if not os.path.exists(tsv_file):
            continue
        with open(tsv_file, 'r') as f:
            #next(f)  # Skip header
            for line in f:
                head, relation, tail, _ = line.strip().split('\t')
                knowledge_graph[head].append((relation, tail))
                if reverse:
                    #be fair, make the rels invertable
                    knowledge_graph[tail].append((relation, head))
    return knowledge_graph


def path_ranking(knowledge_graph, start_entity, num_walks, walk_length):
    entity_distribution = defaultdict(int)

    for _ in range(num_walks):
            current_entity = start_entity
            for _ in range(walk_length):
                neighbors = knowledge_graph[current_entity]
                #restart if no neighbors
                if len(neighbors) == 0:
                     current_entity = start_entity
                     continue
                next_step = random.choice(neighbors)
                current_entity = next_step[1] 

                # Increment the count for the reached entity
                entity_distribution[current_entity] = entity_distribution.get(current_entity, 0) + 1

    # Normalize the distribution 
    total_counts = sum(entity_distribution.values())
    for entity in entity_distribution:
            entity_distribution[entity] = entity_distribution[entity] / total_counts

    return entity_distribution

def evaluate_test_data(knowledge_graph, test_data, num_walks, walk_length):
    all_results = []
    #get the number of head and tail eneites in the knowledge graph
    ent_set = set()
    for head in knowledge_graph.keys():
        for relation, tail in knowledge_graph[head]:
            ent_set.add(head)
            ent_set.add(tail)
    total_ents = len(ent_set)

    for head in tqdm(test_data.keys(), ncols=0):
        tail = test_data[head][0][1]
        entity_distribution = path_ranking(knowledge_graph, head, num_walks, walk_length)
        ranked_entities = sorted(entity_distribution.items(), key=lambda item: item[1], reverse=True)
        rank = 0
        final_rank = -1
        for entity, score in ranked_entities:
            rank += 1
            if entity == tail:
                final_rank = rank
                break
        if final_rank == -1:
             final_rank = total_ents
        mean_rank = final_rank
        reciprocal_rank = 1.0 / final_rank
        hits_at_1 = int(final_rank <= 1)
        hits_at_5 = int(final_rank <= 5)
        hits_at_10 = int(final_rank <= 10)
        hits_at_50 = int(final_rank <= 50)
        all_results.append({
        'head': head,
        'tail': tail,
        'mean_rank': mean_rank,
        'reciprocal_rank': reciprocal_rank,
        'hits@1': hits_at_1,
        'hits@5': hits_at_5,
        'hits@10': hits_at_10,
        'hits@50': hits_at_50,
        })
    return all_results

dataset_paths = ['./data/psl50/OpenEA/EN_DE_100K_V2']
files = ['kg1_list.tsv', 'kg2_list.tsv', 'kg_atts.tsv', 'link_list.tsv', 'kg1_list_psl.tsv']
dataset_name = {dataset_paths[0]: 'PSL 50%'}

for fold in range(1,6):
    for dataset_path in dataset_paths:
        #if the directory does not exist, skip
        if not os.path.isdir(dataset_path + f"/{fold}/"):
            continue

        knowledge_graph = load_knowledge_graphs([dataset_path + f"/{fold}/" + file for file in files], reverse=True)
        test_data = load_knowledge_graph(dataset_path + f"/{fold}/" +'test_list.tsv')


        num_walks = 500
        walk_length = 10

        evaluation_results = evaluate_test_data(knowledge_graph, test_data, num_walks, walk_length)

        #average the results
        mean_rank = sum([result['mean_rank'] for result in evaluation_results]) / len(evaluation_results)
        reciprocal_rank = sum([result['reciprocal_rank'] for result in evaluation_results]) / len(evaluation_results)
        hits_at_1 = sum([result['hits@1'] for result in evaluation_results]) / len(evaluation_results)
        hits_at_5 = sum([result['hits@5'] for result in evaluation_results]) / len(evaluation_results)
        hits_at_10 = sum([result['hits@10'] for result in evaluation_results]) / len(evaluation_results)
        hits_at_50 = sum([result['hits@50'] for result in evaluation_results]) / len(evaluation_results)

        #log to tsv
        with open('pra_results.tsv', 'a') as file:
            #has 2 dummy values for MSE and MAE
            file.write(f'{dataset_name[dataset_path]}\\{fold}\t0.0\\0.0\\{hits_at_1}\t{hits_at_5}\t{hits_at_10}\t{hits_at_50}\t{mean_rank}\t{reciprocal_rank}\n')