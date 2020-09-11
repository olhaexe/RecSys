import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight

class MainRecommender:
    """Рекомендации, которые можно получить из ALS

    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """

    # добавляю на вход item_features, чтобы получить словарь СТМ
    def __init__(self, data, item_features, weighting=True):

        # Топ покупок каждого юзера
        self.top_purchases = data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        self.top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.top_purchases = self.top_purchases[self.top_purchases['item_id'] != 9999999]

        # Топ покупок по всему датасету
        self.overall_top_purchases = data.groupby('item_id')['quantity'].count().reset_index()
        self.overall_top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.overall_top_purchases = self.overall_top_purchases[self.overall_top_purchases['item_id'] != 9999999]
        self.overall_top_purchases = self.overall_top_purchases.item_id.tolist()

        self.user_item_matrix = self.prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self.user_item_matrix)

        # Словарь {item_id: 0/1}. 0/1 - факт принадлежности товара к СТМ
        self.item_id_to_ctm = {}
        for i in range(item_features.shape[0]):
            if item_features['brand'][i] == 'Private':
                self.item_id_to_ctm[item_features['item_id'][i]] = 1
            else:
                self.item_id_to_ctm[item_features['item_id'][i]] = 0

        # Own recommender обучается до взвешивания матрицы
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)

        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T

        self.model = self.fit(self.user_item_matrix)

    @staticmethod
    def prepare_matrix(data):

        user_item_matrix = pd.pivot_table(data,
                                          index='user_id', columns='item_id',
                                          values='quantity',  # Можно пробовать другие варианты
                                          aggfunc='count',
                                          fill_value=0
                                          )
        user_item_matrix = user_item_matrix.astype(float)  # необходимый тип матрицы для implicit

        return user_item_matrix

    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())

        return own_recommender

    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""

        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads)
        model.fit(csr_matrix(user_item_matrix).T.tocsr())

        return model

    @staticmethod
    def fit_bpr(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает BPR"""

        model_bpr = BayesianPersonalizedRanking(factors=n_factors,
                                                regularization=regularization,
                                                iterations=iterations,
                                                num_threads=num_threads)
        model_bpr.fit(csr_matrix(user_item_matrix).T.tocsr())

        return model_bpr

    def get_similar_items_recommendation(self, user, filter_ctm=True, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        if filter_ctm:
            # ctm = item_features[item_features['brand'] == 'Private'].item_id.unique()
            ctm = [key for key, value in self.item_id_to_ctm.items() if value == 1]
            popularity = self.data[~self.data['item_id'].isin(ctm)].groupby([user, 'item_id'])[
                'quantity'].count().reset_index()
        else:
            popularity = self.data.groupby([user, 'item_id'])['quantity'].count().reset_index()
        popularity.sort_values('quantity', ascending=False, inplace=True)

        popularity = popularity[popularity['item_id'] != 9999999]

        popularity = popularity.groupby(user).head(N)
        popularity.sort_values(user, ascending=False, inplace=True)

        res = []
        for x in popularity['item_id']:
            if filter_ctm:
                recs = self.model.similar_items(self.itemid_to_id[x],
                                           N=50)  # Returns list of (itemid, score) tuples, sorted by score
                for item in recs:
                    if item[0] not in ctm:
                        recs.remove(item)
            else:
                recs = self.model.similar_items(self.itemid_to_id[x], N=2)

            top_rec = recs[1][0]
            res.append(self.id_to_itemid[top_rec])

        # your_code
        # Практически полностью реализовали на прошлом вебинаре
        # Не забывайте, что нужно учесть параметр filter_ctm

        return res

    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        recs = self.model.similar_users(self.userid_to_id[user], N=N+1)

        own_recommender_res = self.own_recommender

        res = [self.id_to_itemid[rec[0]] for rec in recs if rec not in own_recommender_res[0]]

        return res


#def get_recommendations(user, model, sparse_user_item, N=5):
    """Рекомендуем топ-N товаров"""

#    res = [id_to_itemid[rec[0]] for rec in
#           model.recommend(userid=userid_to_id[user],
#                           user_items=sparse_user_item,  # на вход user-item matrix
#                           N=N,
#                           filter_already_liked_items=False,
#                           filter_items=[itemid_to_id[9999999]],  # !!!
#                           recalculate_user=False)]
#    return res

#def get_rec(model, ctm, x):
    # Тут нет фильтрации по СТМ !! - в ДЗ нужно будет добавить
#    recs = model.similar_items(itemid_to_id[x], N=50)  # Returns list of (itemid, score) tuples, sorted by score
#    score = 0
#    for item in recs:
#        if item[0] not in ctm:
#            recs.remove(item)

#    top_rec = recs[1][0]

#    return id_to_itemid[top_rec]