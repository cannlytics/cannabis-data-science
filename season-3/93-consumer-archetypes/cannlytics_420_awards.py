

    #--------------------------------------------------------------------------
    # The 420 Awards
    #--------------------------------------------------------------------------

    # # Who sold the most on 4/20/2022?
    # april_20 = stats.loc[stats['date'] == '2022-04-20']
    # max_april_20 = april_20.loc[april_20['total_price'] == april_20['total_price'].max()]
    # print('Who sold the most on 4/20/2022?', max_april_20.iloc[0]['licensee_id'])
    # print(f'${max_april_20.iloc[0]["total_price"]:,}')

    # # Who sold the most in 2022?
    # totals = stats.groupby('licensee_id').sum()
    # max_total = totals.loc[totals['total_price'] == totals['total_price'].max()]
    # print('Who sold the most in 2022?', max_total.iloc[0]['licensee_id'])

    # # Who had the highest single day sales?
    # max_sales = stats.loc[stats['total_price'] == stats['total_price'].max()]
    # print('Who had the highest single day sales?', max_sales.iloc[0]['licensee_id'])
