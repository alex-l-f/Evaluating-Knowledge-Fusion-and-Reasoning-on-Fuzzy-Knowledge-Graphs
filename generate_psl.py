from collections import defaultdict
from tqdm import tqdm

#config
filename = "./data/NL27K/train.tsv"

#load data
index_ents = {}
index_rels = {}
lookup_ents = {}
lookup_rels = {}
def load_triples(filename, splitter='\t', line_end='\n'):
    triples = {} #h -> t -> r -> w
    r_triples = {} #r -> h -> t -> w
    last_e = -1
    last_r = -1

    for line in open(filename):
        line = line.rstrip(line_end).split(splitter)
        if index_rels.get(line[1]) == None:
            last_r += 1
            index_rels[line[1]] = last_r
            lookup_rels[last_r] = line[1]
        if index_ents.get(line[0]) == None:
            last_e += 1
            index_ents[line[0]] = last_e
            lookup_ents[last_e] = line[0]
        if index_ents.get(line[2]) == None:
            last_e += 1
            index_ents[line[2]] = last_e
            lookup_ents[last_e] = line[2]
        h = index_ents[line[0]]
        r = index_rels[line[1]]
        t = index_ents[line[2]]
        w = float(line[3])

        if h not in triples:
            triples[h] = {}
        if t not in triples[h]:
            triples[h][t] = {}
        if r not in r_triples:
            r_triples[r] = {}
        if h not in r_triples[r]:
            r_triples[r][h] = {}

        triples[h][t][r] = w
        r_triples[r][h][t] = w
    return triples, r_triples
triples, r_triples = load_triples(filename)
heads = set(triples.keys())

#three relations make a rule
r_hit_rate = {}
#two relations in a walk
r_count = {}
print("Finding rules")
#for each relation in relation set
for h in tqdm(heads):
    #compute hit rate accross each relevant tuple for (h,r,t)^(t,r',t')->(h,r'',t')
    hit_rate = 0
    count = 0
    for t in triples[h].keys():
        if t in triples.keys():
            for t_prime in triples[t].keys():
                for r in triples[h][t].keys():
                    for r_prime in triples[t][t_prime].keys():
                        if t_prime in triples[h].keys():
                            #extract the final relation for our rule
                            for r_double_prime in triples[h][t_prime]:
                                if r not in r_hit_rate.keys():
                                    r_hit_rate[r] = {}
                                if r_prime not in r_hit_rate[r].keys():
                                    r_hit_rate[r][r_prime] = {}
                                if r_double_prime not in r_hit_rate[r][r_prime].keys():
                                    r_hit_rate[r][r_prime][r_double_prime] = 0
                                r_hit_rate[r][r_prime][r_double_prime] += 1
                        if r not in r_count.keys():
                            r_count[r] = {}
                        if r_prime not in r_count[r].keys():
                            r_count[r][r_prime] = 0
                        r_count[r][r_prime] += 1

#compute hit rate for each rule, while keeping track of teh number of occuanrces
r_hit_rate_final = {}
for r in r_hit_rate.keys():
    r_hit_rate_final[r] = {}
    for r_prime in r_hit_rate[r].keys():
        r_hit_rate_final[r][r_prime] = {}
        for r_double_prime in r_hit_rate[r][r_prime].keys():
            r_hit_rate_final[r][r_prime][r_double_prime] = r_hit_rate[r][r_prime][r_double_prime]/r_count[r][r_prime]

#flatten rules and sort by hit rate
r_hit_rate_final_flat = []
for r in r_hit_rate_final.keys():
    for r_prime in r_hit_rate_final[r].keys():
        for r_double_prime in r_hit_rate_final[r][r_prime].keys():
            r_hit_rate_final_flat.append((r,r_prime,r_double_prime,r_hit_rate_final[r][r_prime][r_double_prime]))
r_hit_rate_final_flat = sorted(r_hit_rate_final_flat, key=lambda x: x[3], reverse=True)

#filter out rules with low hit rate and count
r_hit_rate_final_flat = list(filter(lambda x: x[3] > 0.05 and r_count[x[0]][x[1]] > 100, r_hit_rate_final_flat))

#construct new tuples based on (h,r,t)^(t,r',t')->(h,r'',t') for missing tuples
print("Generating new triples")
triples_new = {}
#for each rule
for r in tqdm(r_hit_rate_final_flat):
    #get candidates for the first part of our logic statment
    c1 = r_triples[r[0]]
    #get candidates for the second part
    c2 = r_triples[r[1]]
    #get existing tuples for the third part
    c3 = r_triples[r[2]]
    # iterate over candidates for the first part
    for h in c1.keys():
        for t in c1[h].keys():
            # check if the second part of our logic statement has candidates for the current t
            if t in c2.keys():
                # iterate over candidates for the second part
                for t_prime in c2[t].keys():
                    # check if the third part of our logic statement does not have the current t_prime
                    if h not in c3.keys() or t_prime not in c3[h].keys():
                        # create a new tuple based on (h,r'',t') for missing tuples
                        if h not in triples_new.keys():
                            triples_new[h] = {}
                        if t_prime not in triples_new[h].keys():
                            triples_new[h][t_prime] = {}
                        #use PSL logic from the paper
                        val = max(0,c1[h][t] + c2[t][t_prime] - 1)
                        if val > 0:
                            triples_new[h][t_prime][r[2]] = val

#save tuples to tsv, with values truncated to 3 significant digits
print("Saving new triples")
with open(filename.replace(".tsv","_psl.tsv"), "w") as f:
    for h in triples_new.keys():
        for t in triples_new[h].keys():
            for r in triples_new[h][t].keys():
                f.write(f"{lookup_ents[h]}\t{lookup_rels[r]}\t{lookup_ents[t]}\t{round(triples_new[h][t][r],3)}\n")
print("Done")