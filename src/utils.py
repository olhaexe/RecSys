import pandas as pd
import numpy as np


def prefilter_items(data, take_n_popular=5000):
    """Предфильтрация товаров"""

    # 1. Удаление товаров, со средней ценой < 1$
    data = data[data['sales_value'] / data['quantity'] >= 1]

    # 2. Удаление товаров со средней ценой > 30$
    data = data[data['sales_value'] / data['quantity'] <= 30]

    # 3. Придумайте свой фильтр
    # Уберем товары со скидкой больше 30% (если купили не только из-за цены, то будут встречаться еще без дисконта)
    data = data[data['retail_disc'] > -30]

    # Уберем самые популярные товары (их и так купят)
    # popularity = data.groupby('item_id')['user_id'].nunique().reset_index() / data['user_id'].nunique()
    # popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)

    # top_popular = popularity[popularity['share_unique_users'] > 0.5].item_id.tolist()
    # data = data[~data['item_id'].isin(top_popular)]

    # Уберем самые НЕ популярные товары (их и так НЕ купят)
    # top_notpopular = popularity[popularity['share_unique_users'] < 0.01].item_id.tolist()
    # data = data[~data['item_id'].isin(top_notpopular)]

    # Уберем товары, которые не продавались за последние 12 месяцев
    # weeks = data.groupby('item_id')['week_no'].last().reset_index()
    # weeks = weeks[weeks['week_no'] < 42
    # year_sales_items = weeks['item_id'].tolist()
    # data = data[data['item_id'].isin(year_sales_items)]

    # 4. Выбор топ-N самых популярных товаров (N = take_n_popular)
    popularity_sales = data.groupby('item_id')['sales_value'].sum().reset_index()
    popularity_sales.sort_values('sales_value', ascending=False, inplace=True)
    n_popular = popularity_sales['item_id'][:take_n_popular].tolist()

    # Заведем фиктивный item_id (если юзер не покупал товары из топ-5000, то он "купил" такой товар)
    data.loc[~data['item_id'].isin(n_popular), 'item_id'] = 9999999
    n_popular.append(9999999)

    data = data[data['item_id'].isin(n_popular)]

    # Уберем не интересные для рекомендаций категории (department) - нужны на вход также item_features

    # ...

    return data