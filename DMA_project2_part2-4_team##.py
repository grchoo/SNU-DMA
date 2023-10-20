# TODO: CHANGE THIS FILE NAME TO DMA_project2_team##.py
# EX. TEAM 1 --> DMA_project2_team01.py

# TODO: IMPORT LIBRARIES NEEDED FOR PROJECT 2
import mysql.connector
import os
import surprise
from surprise import Dataset
from surprise import Reader
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn import tree
import graphviz
from mlxtend.frequent_patterns import association_rules, apriori

np.random.seed(0)

# TODO: CHANGE GRAPHVIZ DIRECTORY
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# TODO: CHANGE MYSQL INFORMATION, team number 
HOST = 'localhost'
USER = 'USER'
PASSWORD = 'PASSWORD'
SCHEMA = 'SCHEMA'
team = 0 


# PART 2: Decision tree 
def part2():
    cnx = mysql.connector.connect(host=HOST, user=USER, password=PASSWORD)
    cursor = cnx.cursor()
    cursor.execute('SET GLOBAL innodb_buffer_pool_size=2*1024*1024*1024;')
    cursor.execute('USE %s;' % SCHEMA)
    
    # TODO: Requirement 2-1. MAKE best_film column
    
    
    # -------
    
    
    # TODO: Requirement 2-2. WRITE MYSQL QUERY AND EXECUTE. SAVE to .csv file
    
    
    fopen = open('DMA_project2_team%02d_part2.csv' % team, 'w', encoding='utf-8')
    
    fopen.close()
 
    
    # -------
    
    
    # TODO: Requirement 2-3. MAKE AND SAVE DECISION TREE
    # gini file name: DMA_project2_team##_part2_gini.pdf
    # entropy file name: DMA_project2_team##_part2_entropy.pdf
    
    
    # -------
    
    # TODO: Requirement 2-4. Don't need to append code for 2-4
    
    # -------
    
    cursor.close()
    

# PART 3: Association analysis
def part3():
    cnx = mysql.connector.connect(host=HOST, user=USER, password=PASSWORD)
    cursor = cnx.cursor()
    cursor.execute('SET GLOBAL innodb_buffer_pool_size=2*1024*1024*1024;')
    cursor.execute('USE %s;' % SCHEMA)
    
    # TODO: Requirement 3-1. CREATE VIEW AND SAVE to .csv file
    
    fopen = open('DMA_project2_team%02d_part3_tag.csv' % team, 'w', encoding='utf-8')
    
    fopen.close()
    
    # ------
    
    # TODO: Requirement 3-2. CREATE 2 VIEWS AND SAVE partial one to .csv file 
    
    fopen = open('DMA_project2_team%02d_part3_UTR.csv' % team, 'w', encoding='utf-8')
    
    fopen.close()
    
    # ------
    
    # TODO: Requirement 3-3. MAKE HORIZONTAL VIEW
    # file name: DMA_project2_team##_part3_horizontal.pkl
    
    # ------
    
    # TODO: Requirement 3-4. ASSOCIATION ANALYSIS
    # filename: DMA_project2_team##_part3_association.pkl (pandas dataframe )
    
    
    # ------
    
    cursor.close()
    

# TODO: Requirement 4-1. WRITE get_top_n 
def get_top_n(algo, testset, id_list, n, user_based=True):
    
    results = defaultdict(list)
    if user_based:
        # TODO: testset의 데이터 중에 user id가 id_list 안에 있는 데이터만 따로 testset_id로 저장 
        # Hint: testset은 (user_id, tag_id, default_rating)의 tuple을 요소로 갖는 list
        testset_id = []
        for i in testset:
            if i[0] in id_list:
                testset_id.append(i)

        predictions = algo.test(testset_id)
        for uid, iid, true_r, est, _ in predictions:
            # TODO: results는 user_id를 key로,  [(tag_id, estimated_rating)의 tuple이 모인 list]를 value로 갖는 dictionary
            results[uid].append((iid,est))
            pass
    else:
        # TODO: testset의 데이터 중 tag id가 id_list 안에 있는 데이터만 따로 testset_id라는 list로 저장
        # Hint: testset은 (user_id, tag_id, default_rating)의 tuple을 요소로 갖는 list
        testset_id = []
        for i in testset:
            if i[1] in id_list:
                testset_id.append(i)

        predictions = algo.test(testset_id)
        for uid, iid, true_r, est, _ in predictions:
            # TODO - results는 tag_id를 key로, [(user_id, estimated_rating)의 tuple이 모인 list]를 value로 갖는 dictionary(3점)
            results[iid].append((uid, est))
            pass
    for id_, ratings in results.items():
        # TODO: rating 순서대로 정렬하고 top-n개만 유지
        ratings.sort(key=lambda x:x[1], reverse=True)
        results[id_] = ratings[0:n]
        pass
    
    return results


# PART 4. Requirement 4-2, 4-3, 4-4
def part4():
    file_path = 'DMA_project2_team%02d_part3_UTR.csv' % team
    reader = Reader(line_format='user item rating', sep=',', rating_scale=(0, 10), skip_lines=1)
    data = Dataset.load_from_file(file_path, reader=reader)

    trainset = data.build_full_trainset()
    testset = trainset.build_anti_testset()

    # TODO: Requirement 4-2. User-based Recommendation
    uid_list = ['21433', '27211', '78970', '184449', '941658'] 
    # TODO - set algorithm for 4-2-1
    sim_opt={'name':'cosine', 'user_based':True}
    algo1 = surprise.KNNBasic(sim_options=sim_opt)

    algo1.fit(trainset)
    results = get_top_n(algo1, testset, uid_list, n=5, user_based=True)
    with open('4-2-1.txt', 'w') as f:
        for uid, ratings in sorted(results.items(), key=lambda x: x[0]):
            f.write('User ID %s top-5 results\n' % uid)
            for iid, score in ratings:
                f.write('Tag ID %s\tscore %s\n' % (iid, str(score)))
            f.write('\n')
            

    # TODO - set algorithm for 4-2-2
    sim_opt={'name':'pearson', 'user_based':True}
    algo2 = surprise.KNNWithMeans(sim_options=sim_opt)

    algo2.fit(trainset)
    results = get_top_n(algo2, testset, uid_list, n=5, user_based=True)
    with open('4-2-2.txt', 'w') as f:
        for uid, ratings in sorted(results.items(), key=lambda x: x[0]):
            f.write('User ID %s top-5 results\n' % uid)
            for iid, score in ratings:
                f.write('Tag ID %s\tscore %s\n' % (iid, str(score)))
            f.write('\n')

    # TODO - 4-2-3. Best Model
    sim_opt = {'name':'pearson_baseline', 'user_based':True, 'min_support':2}
    best_algo_ub = surprise.KNNBasic(sim_options=sim_opt)
    tmp = surprise.model_selection.cross_validate(algo1, data, measures=['RMSE', 'MSE', 'MAE'], cv=5, verbose=True)
    tmp = surprise.model_selection.cross_validate(algo2, data, measures=['RMSE', 'MSE', 'MAE'], cv=5, verbose=True)
    tmp = surprise.model_selection.cross_validate(best_algo_ub, data, measures=['RMSE', 'MSE', 'MAE'], cv=5, verbose=True)


    # TODO: Requirement 4-3. Item-based Recommendation
    tname_list = ['animation', 'sci-fi', 'romance', 'comedy', 'action']
    # TODO - set algorithm for 4-3-1
    sim_opt={'name':'cosine', 'user_based':False}
    algo1 = surprise.KNNBasic(sim_options=sim_opt)

    algo1.fit(trainset)
    results = get_top_n(algo1, testset, tname_list, n=10, user_based=False)
    with open('4-3-1.txt', 'w') as f:
        for iid, ratings in sorted(results.items(), key=lambda x: x[0]):
            f.write('Tag ID %s top-10 results\n' % iid)
            for uid, score in ratings:
                f.write('User ID %s\tscore %s\n' % (uid, str(score)))
            f.write('\n')

    # TODO - set algorithm for 4-3-2
    sim_opt={'name': 'pearson', 'user_based':False}
    algo2 = surprise.KNNWithMeans(sim_options=sim_opt)

    algo2.fit(trainset)
    results = get_top_n(algo2, testset, tname_list, n=10, user_based=False)
    with open('4-3-2.txt', 'w') as f:
        for iid, ratings in sorted(results.items(), key=lambda x: x[0]):
            f.write('Tag ID %s top-10 results\n' % iid)
            for uid, score in ratings:
                f.write('User ID %s\tscore %s\n' % (uid, str(score)))
            f.write('\n')

    # TODO - 4-3-3. Best Model
    s_o = {'name':'pearson_baseline', 'user_based':False, 'min_support':2}
    best_algo_ib = surprise.KNNBasic(sim_options=s_o, k=40)
    tmp=surprise.model_selection.cross_validate(algo1, data, measures=['RMSE', 'MSE', 'MAE'], cv=5, verbose=True)
    tmp=surprise.model_selection.cross_validate(algo2, data, measures=['RMSE', 'MSE', 'MAE'], cv=5, verbose=True)
    tmp=surprise.model_selection.cross_validate(best_algo_ib, data, measures=['RMSE', 'MSE', 'MAE'], cv=5, verbose=True)

    # TODO: Requirement 4-4. Matrix-factorization Recommendation
    # TODO - set algorithm for 4-4-1
    algo = surprise.SVD(n_factors=100, n_epochs=20, biased=False)

    algo.fit(trainset)
    results = get_top_n(algo, testset, uid_list, n=5, user_based=True)
    with open('4-4-1.txt', 'w') as f:
        for uid, ratings in sorted(results.items(), key=lambda x: x[0]):
            f.write('User ID %s top-5 results\n' % uid)
            for iid, score in ratings:
                f.write('Tag ID %s\tscore %s\n' % (iid, str(score)))
            f.write('\n')

    # TODO - set algorithm for 4-4-2
    algo = surprise.SVD(n_factors=200, n_epochs=20, biased=True)

    algo.fit(trainset)
    results = get_top_n(algo, testset, uid_list, n=5, user_based=True)
    with open('4-4-2.txt', 'w') as f:
        for uid, ratings in sorted(results.items(), key=lambda x: x[0]):
            f.write('User ID %s top-5 results\n' % uid)
            for iid, score in ratings:
                f.write('Tag ID %s\tscore %s\n' % (iid, str(score)))
            f.write('\n')

    # TODO - set algorithm for 4-4-3
    algo = surprise.SVDpp(n_factors=100, n_epochs=50)

    algo.fit(trainset)
    results = get_top_n(algo, testset, uid_list, n=5, user_based=True)
    with open('4-4-3.txt', 'w') as f:
        for uid, ratings in sorted(results.items(), key=lambda x: x[0]):
            f.write('User ID %s top-5 results\n' % uid)
            for iid, score in ratings:
                f.write('Tag ID %s\tscore %s\n' % (iid, str(score)))
            f.write('\n')

    # TODO - set algorithm for 4-4-4
    algo = surprise.SVDpp(n_factors=200, n_epochs=100)
    algo.fit(trainset)
    results = get_top_n(algo, testset, uid_list, n=5, user_based=True)
    with open('4-4-4.txt', 'w') as f:
        for uid, ratings in sorted(results.items(), key=lambda x: x[0]):
            f.write('User ID %s top-5 results\n' % uid)
            for iid, score in ratings:
                f.write('Tag ID %s\tscore %s\n' % (iid, str(score)))
            f.write('\n')

    # TODO - 4-4-5. Best Model
    best_algo_mf = surprise.SVD(n_factors=200, n_epochs=20, biased=True)
    tmp = surprise.model_selection.cross_validate(best_algo_mf, data, measures=['RMSE', 'MSE', 'MAE'], cv=5, verbose=True)
    
    

if __name__ == '__main__':
    part4()


# In[ ]:




