from basefunc import *


if __name__ == "__main__":
    delta, df = get_test_data_df()

    count = 128
    n_b = 8
    N = count
    dt = N / n_b
    print(len(delta) / count)

    csp = index_csum(delta, count)
    basis = genbasis_df(dt, count, n_b)

    lr = LinearRegression(fit_intercept=False)
    lr = BayesianRidge(fit_intercept=False)

    N_obs = 30
    optx = csp.loc[N_obs * count : (N_obs + 1) * count - 1]
    optx.index = basis.index
    print(optx)
    print(csp)
    # sys.exit()
    lr.fit(basis, optx)
    weights = lr.coef_
    weights = np.around(weights, 4)
    print(weights)
    cts = (weights * basis).sum(1)

    cts.plot(figsize=(11, 9))
    optx.plot()
    delta.iloc[
        N_obs * count : (N_obs + 1) * count - 1
    ].reset_index().delta.cumsum().plot()
    plt.show()
    # print(optx)
    # print(optx.shape)

    # print(csp.tail())
    # print(csp.shape)
    # csp[N_obs*count:(N_obs+1)*count].plot(figsize=(11, 9))
    # idxs[N_obs*count:(N_obs+1)*count].plot(alpha=0.6,color="red",secondary_y=True)
    # plt.show()
    # sys.exit()
    #
    # print(df.iloc[100:1000:50])
    # # print (df["sec"].value_counts())
    # daycs = df.groupby(str(df.index.strftime("%D"))).cumsum()
    # print(daycs.head())
    # print(df.index.day)
    # # sys.exit()
    #
    # print(df.resample("1D").mean())
    # print(df.resample("1D").mean().isnull().sum())
    # print(df.isnull().sum())
    # print(df.shape)
    # # sys.exit()
    #
    # delta.plot(figsize=(11, 9))
    # plt.show()
    #
    # print(df.head())
    # print(df.dtypes)
    # print(df.describe())
